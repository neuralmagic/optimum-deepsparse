import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple, Union

import onnx
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.file_utils import add_start_docstrings

import deepsparse
from optimum.exporters import TasksManager
from optimum.exporters.onnx import main_export
from optimum.modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from optimum.onnx.utils import _get_external_data_paths
from optimum.onnxruntime import ORTModel
from optimum.onnxruntime.utils import ONNX_WEIGHTS_NAME
from optimum.utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors


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
