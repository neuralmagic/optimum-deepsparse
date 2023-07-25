import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np 
import torch
import os
import onnx
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.file_utils import add_start_docstrings
import importlib

import deepsparse
from optimum.exporters import TasksManager
from optimum.exporters.onnx import OnnxConfig, main_export
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from optimum.onnx.utils import  _get_external_data_paths
    
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER
    )

from optimum.onnxruntime import ORTModel
from optimum.onnxruntime.utils import (
    _ORT_TO_NP_TYPE,
    ONNX_WEIGHTS_NAME,
    get_provider_for_device,
    parse_device,
    validate_provider_availability,
)
from optimum.utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors

from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer

import onnxruntime as ort

from optimum.exporters.onnx import OnnxConfig, main_export
from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin

logger = logging.getLogger(__name__)


def parse_shapes(shape_string: str) -> List[List[int]]:
    """
    Reduces a string representation of a list of shapes to an actual list of shapes.
    Examples:
        "[1,2,3]" -> input0=[1,2,3]
        "[1,2,3],[4,5,6],[7,8,9]" -> input0=[1,2,3] input1=[4,5,6] input2=[7,8,9]
    """
    if not shape_string:
        return None

    shapes_list = []
    if shape_string:
        matches = re.findall(r"\[(.*?)\],?", shape_string)
        if matches:
            for match in matches:
                # Clean up stray extra brackets
                value = match.replace("[", "").replace("]", "")
                # Parse comma-separated dims into shape list
                shape = [int(s) for s in value.split(",")]
                shapes_list.append(shape)
        else:
            raise Exception(f"Can't parse input shapes parameter: {shape_string}")

    return shapes_list


def override_onnx_input_shapes(
    onnx_filepath: str,
    input_shapes: Union[List[int], List[List[int]]],
    output_path: Optional[str] = None,
) -> str:
    """
    Rewrite input shapes of ONNX model, saving the modified model and returning its path

    :param onnx_filepath: File path to ONNX model. If the graph is to be
        modified in-place, only the model graph will be loaded and modified.
        Otherwise, the entire model will be loaded and modified, so that
        external data are saved along the model graph.
    :param input_shapes: Override for model's input shapes
    :param output_path: If None, overwrite the original model file inplace. Otherwise write to path
    :return: File path to modified ONNX model.
        If output_path is True,
        the modified model will be saved to the same path as the original
        model. Else the modified model will be saved to output_path.
    """
    inplace = output_path is None

    if input_shapes is None:
        return onnx_filepath

    model = onnx.load(onnx_filepath, load_external_data=not inplace)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [input for input in all_inputs if input.name not in initializer_input_names]

    # Input shapes should be a list of lists, even if there is only one input
    if not all(isinstance(inp, list) for inp in input_shapes):
        input_shapes = [input_shapes]

    # If there is a single input shape given and multiple inputs,
    # duplicate for all inputs to apply the same shape
    if len(input_shapes) == 1 and len(external_inputs) > 1:
        input_shapes.extend([input_shapes[0] for _ in range(1, len(external_inputs))])

    # Make sure that input shapes can map to the ONNX model
    assert len(external_inputs) == len(
        input_shapes
    ), "Mismatch of number of model inputs ({}) and override shapes ({})".format(
        len(external_inputs), len(input_shapes)
    )

    # Overwrite the input shapes of the model
    for input_idx, external_input in enumerate(external_inputs):
        assert len(external_input.type.tensor_type.shape.dim) == len(
            input_shapes[input_idx]
        ), "Input '{}' shape doesn't match shape override: {} vs {}".format(
            external_input.name,
            external_input.type.tensor_type.shape.dim,
            input_shapes[input_idx],
        )
        for dim_idx, dim in enumerate(external_input.type.tensor_type.shape.dim):
            dim.dim_value = input_shapes[input_idx][dim_idx]

    if inplace:
        logger.info(f"Overwriting in-place the input shapes of the model at {onnx_filepath}")
        onnx.save(model, onnx_filepath)
        return onnx_filepath
    else:
        logger.info(f"Saving new model with static input shapes at {output_path}")
        onnx.save(model, output_path)
        return output_path


@add_start_docstrings(
    """
    Base DeepSparse class.
    """,
)
class DeepSparseBaseModel(OptimizedModel):
    model_type = "onnx_model"
    auto_model_class = AutoModel
    export_feature = None

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return TasksManager.infer_task_from_model(auto_model_class)

    def shared_attributes_init(
        self,
        model: deepsparse.Engine,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        """
        Initializes attributes that may be shared among DeepSparse engines.
        """
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        # # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # # would end-up removing the directory containing the underlying ONNX model.
        # self._model_save_dir_tempdirectory_instance = None
        # if model_save_dir is None:
        #     self.model_save_dir = Path(model._model_path).parent
        # elif isinstance(model_save_dir, TemporaryDirectory):
        #     self._model_save_dir_tempdirectory_instance = model_save_dir
        #     self.model_save_dir = Path(model_save_dir.name)
        # elif isinstance(model_save_dir, str):
        #     self.model_save_dir = Path(model_save_dir)
        # else:
        #     self.model_save_dir = model_save_dir

        self.preprocessors = preprocessors if preprocessors is not None else []

        # Registers the DeepSparseModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating a pipeline
        # https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def __init__(
        self,
        model: str,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        input_shapes: Optional[Union[str, List]] = None,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        **kwargs,
    ):
        super().__init__(model, config)

        self.config = config
        self.model_save_dir = model_save_dir

        self.model = str(model)
        self.model_path = Path(self.model)
        self.model_name = self.model_path.name
        self.engine = None

        self.input_shapes = parse_shapes(input_shapes)
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict

    def _check_is_dynamic(self) -> bool:
        has_dynamic = False
        if isinstance(self.model, str) and self.model.endswith(".onnx"):
            model = onnx.load(self.model)
            has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in model.graph.input
            )

        return has_dynamic

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        `from_pretrained` class method. It will always save the file under model_save_dir/latest_model_name.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        src_paths = [self.model_path]
        dst_paths = [Path(save_directory) / self.model_path.name]

        # add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        from_onnx: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        model_path = Path(model_id)
        regular_onnx_filenames = ORTModel._generate_regular_names_for_filename(ONNX_WEIGHTS_NAME)

        if file_name is None:
            if model_path.is_dir():
                onnx_files = list(model_path.glob("*.onnx"))
            else:
                if isinstance(use_auth_token, bool):
                    token = HfFolder().get_token()
                else:
                    token = use_auth_token
                repo_files = map(Path, HfApi().list_repo_files(model_id, revision=revision, token=token))
                pattern = "*.onnx" if subfolder == "" else f"{subfolder}/*.onnx"
                onnx_files = [p for p in repo_files if p.match(pattern)]

            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any ONNX model file in {model_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Too many ONNX model files were found in {model_path}, specify which one to load by using the "
                    "file_name argument."
                )
            else:
                file_name = onnx_files[0].name

        if file_name not in regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {file_name} is not a regular name used in optimum.onnxruntime, the ORTModel might "
                "not behave as expected."
            )

        preprocessors = None
        # Load the model from local directory
        if model_path.is_dir():
            model = model_path / file_name
            new_model_save_dir = model_id
            preprocessors = maybe_load_preprocessors(model_id)
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

            # try download external data for >2GB models
            try:
                hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=file_name + "_data",
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                # model doesn't use external data
                pass

            model = model_cache_path
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(model, config=config, model_save_dir=model_save_dir, preprocessors=preprocessors, **kwargs)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        custom_onnx_configs: Optional[Dict[str, OnnxConfig]] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            custom_onnx_configs (`Optional[Dict[str, OnnxConfig]]`, defaults to `None`):
                Experimental usage: override the default ONNX config used for the given model. This argument may be useful for advanced users that desire a finer-grained control on the export. An example is available [here](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model).
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            custom_onnx_configs=custom_onnx_configs,
        )

        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            model_save_dir=save_dir,
            **kwargs,
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        return super().from_pretrained(
            model_id,
            export=export,
            force_download=force_download,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            **kwargs,
        )

    def compile(self, batch_size=1):
        """
        Compiles the model with DeepSparse
        """
        if self.engine is None:
            self.is_dynamic = self._check_is_dynamic()
            if self.is_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("Model is dynamic and has no shapes defined, skipping reshape..")
                    self.engine = deepsparse.Engine(model=self.model, batch_size=batch_size)
                    return
                self._reshape(self.model)

            logger.info("Compiling...")
            self.engine = deepsparse.Engine(model=self.model, batch_size=batch_size)
            logger.info(self.engine)

    def _reshape(
        self,
        model_path,
    ):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            model_path (`int`):
                Path to the model.
        """
        # Reset the engine so we know to recompile
        self.engine = None

        if not isinstance(model_path, str) or not model_path.endswith(".onnx"):
            raise ValueError("The model_path isn't a path to an ONNX '{model_path}'")

        if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
            raise ValueError(
                "The model provided has dynamic axes in input, please provide `input_shapes` for compilation!"
            )

        # static_model_path = str(Path(model_path).parent / ONNX_WEIGHTS_NAME_STATIC)
        static_model_path = model_path

        if self.input_shapes:
            self.model = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=static_model_path
            )
        else:
            from onnx import shape_inference
            from onnx.tools import update_model_dims

            model = onnx.load(model_path)
            updated_model = update_model_dims.update_inputs_outputs_dims(
                model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_model = shape_inference.infer_shapes(updated_model)

            onnx.save(inferred_model, static_model_path)
            self.model = static_model_path

    def reshape(
        self,
        input_shapes: Optional[Union[str, List]] = None,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
    ):
        if input_shapes:
            self.input_shapes = parse_shapes(input_shapes)

        if input_shape_dict:
            self.input_shape_dict = input_shape_dict

        if output_shape_dict:
            self.output_shape_dict = output_shape_dict

        self._reshape(self.model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class DeepSparseStableDiffusionPipelineBase(DeepSparseBaseModel):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def shared_attributes_init(
        self,
        vae_encoder_model: deepsparse.Engine,
        vae_decoder_model: deepsparse.Engine,
        text_encoder_model: deepsparse.Engine,
        text_2_encoder_model: deepsparse.Engine,
        unet_model: deepsparse.Engine,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        **kwargs,
    ):
        """
        Initializes attributes that may be shared among DeepSparse engines.
        """
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        self.preprocessors = preprocessors if preprocessors is not None else []

        # Registers the DeepSparseModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating a pipeline
        # https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def __init__(
        self,
        vae_decoder_model: str,
        vae_encoder_model: str,
        text_encoder_model: str,
        text_encoder_2_model: str,
        unet_model: str,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        input_shapes: Optional[Union[str, List]] = None,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
    ):
        """
        Args:
            vae_decoder_engine (`DeepSparse Engine`):
                The DeepSparse Engine ssociated to the VAE decoder.
            text_encoder_engine (`DeepSparse Engine`):
                The DeepSparse Engine associated to the text encoder.
            unet_session (`DeepSparse Engine`):
                The DeepSparse Engine associated to the U-NET.
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            vae_encoder_session (`Optional[ort.InferenceSession]`, defaults to `None`):
                The ONNX Runtime inference session associated to the VAE encoder.
            use_io_binding (`Optional[bool]`, defaults to `None`):
                Whether to use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`Optional[str]`, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """

        self._internal_dict = config
        self.model_save_dir = model_save_dir

        self.vae_encoder_engine =  None
        self.text_encoder_engine = None
        self.text_encoder_2_engine = None
        self.unet_engine = None
        self.vae_decoder_engine = None

        self.input_shapes = parse_shapes(input_shapes)
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict

        self.text_encoder_2_model = str(text_encoder_2_model)

        self.vae_encoder = DeepSparseModelVaeEncoder(self.vae_encoder_engine, self)
        self.vae_encoder_model = str(vae_encoder_model)
        
        self.unet = DeepSparseModelUnet(self.unet_engine, self)
        self.unet_model = str(unet_model)
        self.unet_model_path = Path(self.unet_model_path)
        self.unet_model_name = self.unet_model_path.name

        self.vae_decoder = DeepSparseModelVaeDecoder(self.vae_encoder_engine, self)
        self.vae_decoder_model = str(vae_decoder_model)
        self.vae_decoder_model_path = Path(self.vae_decoder_model)
        self.vae_decoder_model_name = self.vae_decoder_model_path.name
        
        self.text_encoder_model = str(text_encoder_model)


        if self.text_encoder_engine is not None:
            self.text_encoder_model_path = Path(self.text_encoder_model_path)
            self.text_encoder_model_name = Path(self.text_encoder_model_path.name)
            self.text_encoder = DeepSparseModelTextEncoder(self.text_encoder_engine, self)
        else:
            self.text_encoder_model_path = None
            self.text_encoder = None

        if self.vae_encoder_engine is not None:
            self.vae_encoder_model_path = Path(self.vae_encoder_model_path)
            self.vae_encoder_model_name = self.vae_encoder_model_path.name
            self.vae_encoder = DeepSparseModelVaeEncoder(self.vae_encoder_engine, self)
        else:
            self.vae_encoder_model_path = None
            self.vae_encoder = None

        if self.text_encoder_2_engine is not None:
            self.text_encoder_2_model_path = Path(self.text_encoder_2_model_path)
            self.text_encoder_2_model_name = Path(self.text_encoder_2_model_path.name)
            self.text_encoder_2 = DeepSparseModelTextEncoder(self.text_encoder_2_engine, self)
        else:
            self.text_encoder_2_model_path = None
            self.text_encoder_2 = None

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None

        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER: self.vae_encoder,
            DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER: self.text_encoder_2,
        }

        # Modify config to keep the resulting model compatible with diffusers pipelines
        for name in sub_models.keys():
            self._internal_dict[name] = (
                ("diffusers", "OnnxRuntimeModel") if sub_models[name] is not None else (None, None)
            )
        self._internal_dict.pop("vae", None)

        if "block_out_channels" in self.vae_decoder.config:
            self.vae_scale_factor = 2 ** (len(self.vae_decoder.config["block_out_channels"]) - 1)
        else:
            self.vae_scale_factor = 8

    def _check_vae_encoder_is_dynamic(self) -> bool:
        vae_encoder_has_dynamic = False

        if isinstance(self.vae_encoder_model, str) and self.vae_encoder_model.endswith(".onnx"):
            vae_encoder_model = onnx.load(self.vae_encoder_model)
            vae_encoder_has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in vae_encoder_model.graph.input
            )

        return vae_encoder_has_dynamic
    
    def _check_vae_decoder_is_dynamic(self) -> bool:
        vae_decoder_has_dynamic = False

        if isinstance(self.vae_decoder_model, str) and self.vae_decoder_model.endswith(".onnx"):
            vae_decoder_model = onnx.load(self.vae_decoder_model)
            vae_decoder_has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in vae_decoder_model.graph.input
            )

        return vae_decoder_has_dynamic
    
    def _check_text_encoder_is_dynamic(self) -> bool:
        text_encoder_has_dynamic = False

        if isinstance(self.text_encoder_model, str) and self.text_encoder_model.endswith(".onnx"):
            text_encoder_model = onnx.load(self.text_encoder_model)
            text_encoder_has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in text_encoder_model.graph.input
            )

        return text_encoder_has_dynamic
    
    def _check_text_2_encoder_is_dynamic(self) -> bool:
        text_encoder_2_has_dynamic = False

        if isinstance(self.text_encoder_2_model, str) and self.text_encoder_2_model.endswith(".onnx"):
            text_encoder_2_model = onnx.load(self.text_encoder_2_model)
            text_encoder_2_has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in text_encoder_2_model.graph.input
            )

        return text_encoder_2_has_dynamic
    
    def _check_unet_is_dynamic(self) -> bool:
        unet_has_dynamic = False

        if isinstance(self.unet_model, str) and self.unet_model.endswith(".onnx"):
            unet_model = onnx.load(self.unet_model)
            unet_has_dynamic = any(
                any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in unet_model.graph.input
            )

        return unet_has_dynamic

    def _save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        src_to_dst_path = {
            self.vae_decoder_model_path: save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.text_encoder_model_path: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            self.unet_model_path: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / ONNX_WEIGHTS_NAME,
        }

        sub_models_to_save = {
            self.vae_encoder_model_path: DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
            self.text_encoder_2_model_path: DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
        }
        for path, subfolder in sub_models_to_save.items():
            if path is not None:
                src_to_dst_path[path] = save_directory / subfolder / ONNX_WEIGHTS_NAME

        # TODO: Modify _get_external_data_paths to give dictionnary
        src_paths = list(src_to_dst_path.keys())
        dst_paths = list(src_to_dst_path.values())
        # Add external data paths in case of large models
        src_paths, dst_paths = _get_external_data_paths(src_paths, dst_paths)

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            config_path = src_path.parent / self.sub_component_config_name
            if config_path.is_file():
                shutil.copyfile(config_path, dst_path.parent / self.sub_component_config_name)

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        vae_decoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        unet_file_name: str = ONNX_WEIGHTS_NAME,
        vae_encoder_file_name: str = ONNX_WEIGHTS_NAME,
        text_encoder_2_file_name: str = ONNX_WEIGHTS_NAME,
        local_files_only: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        if provider == "TensorrtExecutionProvider":
            raise ValueError("The provider `'TensorrtExecutionProvider'` is not supported")

        model_id = str(model_id)
        patterns = set(config.keys())
        sub_models_to_load = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})

        if not os.path.isdir(model_id):
            patterns.update({"vae_encoder", "vae_decoder"})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    vae_decoder_file_name,
                    text_encoder_file_name,
                    unet_file_name,
                    vae_encoder_file_name,
                    text_encoder_2_file_name,
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
            )
        new_model_save_dir = Path(model_id)

        sub_models = {}
        for name in sub_models_to_load:
            library_name, library_classes = config[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    sub_models[name] = load_method(new_model_save_dir / name)
                else:
                    sub_models[name] = load_method(new_model_save_dir)

        vae_decoder, text_encoder, unet, vae_encoder, text_encoder_2 = cls.load_model(
            vae_decoder_path=new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            text_encoder_path=new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            unet_path=new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            vae_encoder_path=new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name,
            text_encoder_2_path=new_model_save_dir
            / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER
            / text_encoder_2_file_name,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        if use_io_binding:
            raise ValueError(
                "IOBinding is not yet available for stable diffusion model, please set `use_io_binding` to False."
            )

        return cls(
            vae_encoder_model=vae_decoder,
            text_encoder_model=text_encoder,
            unet_model=unet,
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            scheduler=sub_models.get("scheduler"),
            feature_extractor=sub_models.get("feature_extractor", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            vae_encoder_model=vae_encoder,
            text_encoder_2_model=text_encoder_2,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[ort.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "DeepSparseStableDiffusionPipeline":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        return cls._from_pretrained(
            save_dir_path,
            config=config,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
        )
    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        export: bool = False,
        force_download: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        config: Optional["PretrainedConfig"] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        return super().from_pretrained(
            model_id,
            export=export,
            force_download=force_download,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            subfolder=subfolder,
            config=config,
            local_files_only=local_files_only,
            **kwargs,
        )

    def compile(self, batch_size=1):
        """
        Compile the models with DeepSparse
        """
        if (
            self.vae_encoder_engine is None or 
            self.vae_decoder_engine is None or
            self.text_encoder_engine is None or 
            self.text_encoder_2_engine is None or 
            self.unet_engine is None
            ):
            self.vae_encoder_has_dynamic = self._check_vae_encoder_is_dynamic()
            self.vae_decoder_has_dynamic = self._check_vae_decoder_is_dynamic()
            self.unet_has_dynamic = self._check_unet_is_dynamic()
            self.text_encoder_has_dynamic = self._check_text_encoder_is_dynamic()
            self.text_encoder_2_has_dynamic = self._check_text_2_encoder_is_dynamic

            if self.vae_encoder_has_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("VAE encoder is dynamic and has no shapes defined, skipping reshape..")
                    self.vae_encoder_engine = deepsparse.Engine(model=self.vae_encoder_model, batch_size=batch_size)
                    return
                self._reshape(self.vae_encoder_model, model_type="vae_encoder")

            logger.info("Compiling...")
            self.vae_encoder_engine = deepsparse.Engine(model=self.vae_encoder_model, batch_size=batch_size)
            logger.info(self.vae_encoder_engine)

            if self.vae_decoder_has_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("VAE decoder is dynamic and has no shapes defined, skipping reshape..")
                    self.vae_decoder_engine = deepsparse.Engine(model=self.vae_decoder_model, batch_size=batch_size)
                    return
                self._reshape(self.vae_decoder_model, model_type="vae_decoder")

            logger.info("Compiling...")
            self.vae_decoder_engine = deepsparse.Engine(model=self.vae_decoder_model, batch_size=batch_size)
            logger.info(self.vae_decoder_engine)

            if self.unet_has_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("Unet is dynamic and has no shapes defined, skipping reshape..")
                    self.unet_engine = deepsparse.Engine(model=self.unet_model, batch_size=batch_size)
                    return
                self._reshape(self.unet_model, model_type="unet")

            logger.info("Compiling...")
            self.unet_engine = deepsparse.Engine(model=self.unet_model, batch_size=batch_size)
            logger.info(self.unet_engine)    

            if self.text_encoder_has_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("Text encoder is dynamic and has no shapes defined, skipping reshape..")
                    self.text_encoder_engine = deepsparse.Engine(model=self.text_encoder_model, batch_size=batch_size)
                    return
                self._reshape(self.text_encoder_model, model_type="text_encoder")

            logger.info("Compiling...")
            self.text_encoder_engine = deepsparse.Engine(model=self.text_encoder_model, batch_size=batch_size)
            logger.info(self.text_encoder_engine)      

            if self.text_encoder_2_has_dynamic:
                if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
                    logger.warn("Text encoder 2 is dynamic and has no shapes defined, skipping reshape..")
                    self.text_encoder_2_engine = deepsparse.Engine(model=self.text_encoder_2_model, batch_size=batch_size)
                    return
                self._reshape(self.text_encoder_2_model, model_type="text_encoder_2")

            logger.info("Compiling...")
            self.text_encoder_2_engine = deepsparse.Engine(model=self.text_encoder_model, batch_size=batch_size)
            logger.info(self.text_encoder_2_engine)  

    def _reshape(
        self,
        model_type,
    ):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            model_type (`str`):
                Model type.
        """
        model_path = ""
        if model_type == "vae_encoder":
            # Reset the engine so we know to recompile
            self.vae_encoder_engine = None 
            model_path = self.vae_encoder_model_path
            vae_encoder_static_model_path = model_path
        
        if model_type == "vae_decoder":
            self.vae_decoder_engine = None
            model_path = self.vae_decoder_model_path
            vae_decoder_static_model_path = model_path
        
        if model_type == "unet":
            self.unet_engine = None
            model_path = self.unet_model_path
            unet_static_model_path = model_path
        
        if model_type == "text_encoder":
            self.text_encoder_engine = None 
            model_path = self.text_encoder_model_path
            text_encoder_static_model_path = model_path
        
        if model_type == "text_encoder_2":
            self.text_encoder_2_engine = None 
            model_path = self.text_encoder_2_model_path
            text_2_encoder_static_model_path = model_path


        if not isinstance(model_path, str) or not model_path.endswith(".onnx"):
            raise ValueError("The model_path isn't a path to an ONNX '{model_path}'")

        if self.input_shapes is None and (self.input_shape_dict is None or self.output_shape_dict is None):
            raise ValueError(
                "The model provided has dynamic axes in input, please provide `input_shapes` for compilation!"
            )

        if self.input_shapes:
            self.vae_encoder_model = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=vae_encoder_static_model_path
            )
            self.vae_decoder_model = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=vae_decoder_static_model_path
            )
            self.unet_model = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=unet_static_model_path
            )
            self.text_encoder_model = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=text_encoder_static_model_path
            )
            self.text_2_encoder_static_model_path = override_onnx_input_shapes(
                model_path, input_shapes=self.input_shapes, output_path=text_2_encoder_static_model_path
            )
        else:
            from onnx import shape_inference
            from onnx.tools import update_model_dims

            vae_encoder_model = onnx.load(model_path)
            updated_vae_encoder_model = update_model_dims.update_inputs_outputs_dims(
                vae_encoder_model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_vae_encoder_model = shape_inference.infer_shapes(updated_vae_encoder_model)
            onnx.save(inferred_vae_encoder_model, vae_encoder_static_model_path)
            self.vae_encoder_model = vae_encoder_static_model_path

            vae_decoder_model = onnx.load(model_path)
            updated_vae_encoder_model = update_model_dims.update_inputs_outputs_dims(
                vae_decoder_model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_vae_decoder_model = shape_inference.infer_shapes(updated_vae_encoder_model)
            onnx.save(inferred_vae_decoder_model, vae_encoder_static_model_path)
            self.vae_decoder_model = vae_decoder_static_model_path

            unet_model = onnx.load(model_path)
            updated_unet_model = update_model_dims.update_inputs_outputs_dims(
                unet_model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_unet_model = shape_inference.infer_shapes(updated_unet_model)
            onnx.save(inferred_unet_model, unet_static_model_path)
            self.unet_model = unet_static_model_path

            text_encoder_model = onnx.load(model_path)
            updated_text_encoder_model = update_model_dims.update_inputs_outputs_dims(
                text_encoder_model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_text_encoder_model = shape_inference.infer_shapes(updated_text_encoder_model)
            onnx.save(inferred_text_encoder_model, text_encoder_static_model_path)
            self.text_encoder_model = text_encoder_static_model_path

            text_encoder_2_model = onnx.load(model_path)
            updated_text_encoder_2_model = update_model_dims.update_inputs_outputs_dims(
                text_encoder_2_model, self.input_shape_dict, self.output_shape_dict
            )
            inferred_text_encoder_2_model = shape_inference.infer_shapes(updated_text_encoder_2_model)
            onnx.save(inferred_text_encoder_2_model, text_encoder_static_model_path)
            self.text_encoder_2_model = text_2_encoder_static_model_path

    def reshape(
        self,
        input_shapes: Optional[Union[str, List]] = None,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
    ):
        if input_shapes:
            self.input_shapes = parse_shapes(input_shapes)

        if input_shape_dict:
            self.input_shape_dict = input_shape_dict

        if output_shape_dict:
            self.output_shape_dict = output_shape_dict

        self._reshape(self.vae_encoder_model)
        self._reshape(self.vae_decoder_model)
        self._reshape(self.text_encoder_model)
        self._reshape(self.text_encoder_2_model)
        self._reshape(self.unet_model)


    def forward(self, *args, **kwargs):
        raise NotImplementedError
    

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)


# TODO : Use ORTModelPart once IOBinding support is added
class _ORTDiffusionModelPart:
    """
    For multi-file ONNX models, represents a part of the model.
    It has its own `onnxruntime.InferenceSession`, and can perform a forward pass.
    """

    CONFIG_NAME = "config.json"

    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseBaseModel):
        self.engine = engine
        self.parent_model = parent_model
        self.input_names = {input_key: idx for idx, input_key in enumerate(self.engine.input_names)}
        self.output_names = {output_key: idx for idx, output_key in enumerate(self.engine.output_names)}
        # config_path = Path(session._model_path).parent / self.CONFIG_NAME
        # self.config = self.parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}
        # self.input_dtype = {inputs: _ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    @property
    def device(self):
        return self.parent_model.device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DeepSparseModelTextEncoder(_ORTDiffusionModelPart):
    def forward(self, input_ids: np.ndarray):
        onnx_inputs = {
            "input_ids": input_ids,
        }
        outputs = self.text_encoder_engine([onnx_inputs])
        return outputs


class DeepSparseModelUnet(_ORTDiffusionModelPart):
    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseStableDiffusionPipelineBase):
        super().__init__(engine, parent_model)

    def forward(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
    ):
        onnx_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            onnx_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            onnx_inputs["time_ids"] = time_ids

        outputs = self.unet_engine(list(np.expand_dims(onnx_inputs, axis=0)))
        return outputs


class DeepSparseModelVaeDecoder(_ORTDiffusionModelPart):
    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.vae_decoder_engine(list(np.expand_dims(onnx_inputs, axis=0)))
        return outputs


class DeepSparseModelVaeEncoder(_ORTDiffusionModelPart):
    def forward(self, sample: np.ndarray):
        onnx_inputs = {
            "sample": sample,
        }
        outputs = self.vae_encoder_engine(list(np.expand_dims(onnx_inputs, axis=0)))
        return outputs

