#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import importlib
import logging
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import numpy as np
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_start_docstrings

import deepsparse
from optimum.exporters.onnx import main_export
from optimum.onnx.utils import _get_external_data_paths
from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

from .modeling_base import DeepSparseBaseModel


ONNX_WEIGHTS_NAME = "model.onnx"

logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base DeepSparseStableDiffusionPipelineBase class.
    """,
)
class DeepSparseStableDiffusionPipelineBase(DeepSparseBaseModel):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    base_model_prefix = "onnx_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        static_shapes: bool = True,
        device: str = "CPU",
        vae_decoder_path: Optional[Union[str, Path, TemporaryDirectory]] = None,
        vae_encoder_path: Optional[Union[str, Path, TemporaryDirectory]] = None,
        text_encoder_path: Optional[Union[str, Path, TemporaryDirectory]] = None,
        unet_path: Optional[Union[str, Path, TemporaryDirectory]] = None,
        text_encoder_2_path: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        """
        Args:
            vae_decoder_path (`Optional[Union[str, Path, TemporaryDirectory]]`):
                Path to VAE encoder ONNX file.
            vae_encoder_path (`Optional[Union[str, Path, TemporaryDirectory]]`):
                Path to VAE decoder ONNX file.
            text_encoder_path (`Optional[Union[str, Path, TemporaryDirectory]]`):
                Path to text encoder ONNX file.
            unet_path (`Optional[Union[str, Path, TemporaryDirectory]]`):
                Path to UNET ONNX file.
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
            model_save_dir (`Optional[str]`, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """

        self._internal_dict = config
        self.model_save_dir = model_save_dir

        self.is_static = static_shapes

        self.vae_decoder_path = vae_decoder_path
        self.vae_encoder_path = vae_encoder_path
        self.text_encoder_path = text_encoder_path
        self.unet_path = unet_path
        self.text_encoder_2_path = text_encoder_2_path

        self._device = device.upper()

        self.compile(height=None, width=None)

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

    def compile(self, height, width, batch_size=1):
        os.environ["NM_DISABLE_BATCH_OVERRIDE"] = "1"
        unet_batch_size = 2
        if self.is_static:
            logger.info(f"Compiling model with height {height} and width {width}")
            # Create a dummy vae decoder in order to load config information
            self.temp_vae_decoder = DeepSparseModelVaeDecoder(None, self)
            if "block_out_channels" in self.temp_vae_decoder.config:
                self.vae_scale_factor = 2 ** (len(self.temp_vae_decoder.config["block_out_channels"]) - 1)
            else:
                self.vae_scale_factor = 8
            # Create a dummy Unet in order to load config information
            self.temp_unet = DeepSparseModelUnet(None, self)

            height = height or self.temp_unet.config.get("sample_size", 64) * self.vae_scale_factor
            width = width or self.temp_unet.config.get("sample_size", 64) * self.vae_scale_factor

            sample_size_height = height // self.vae_scale_factor
            sample_size_width = width // self.vae_scale_factor

            cross_attention_dim = self.temp_unet.config.get("cross_attention_dim")
            # CLIP is fixed to a max length of 77
            max_seq_len = 77
            vae_input_shapes = [[batch_size, 4, sample_size_height, sample_size_width]]
            text_encoder_input_shapes = [[batch_size, max_seq_len]]
            unet_input_shapes = [
                [unet_batch_size, 4, sample_size_height, sample_size_width],
                [1],
                [unet_batch_size, max_seq_len, cross_attention_dim],
            ]

            self.vae_decoder_engine = deepsparse.Engine(
                model=str(self.vae_decoder_path), batch_size=batch_size, input_shapes=vae_input_shapes
            )
            self.vae_encoder_engine = deepsparse.Engine(
                model=str(self.vae_encoder_path), batch_size=batch_size, input_shapes=vae_input_shapes
            )
            self.text_encoder_engine = deepsparse.Engine(
                model=str(self.text_encoder_path), batch_size=batch_size, input_shapes=text_encoder_input_shapes
            )
            self.unet_engine = deepsparse.Engine(
                model=str(self.unet_path), batch_size=unet_batch_size, input_shapes=unet_input_shapes
            )
            if os.path.exists(self.text_encoder_2_path):
                self.text_encoder_2_engine = deepsparse.Engine(
                    model=str(self.text_encoder_2_path), batch_size=batch_size, input_shapes=text_encoder_input_shapes
                )

        else:
            logger.info("Compiling model with dynamic shapes, not recommended for performance..")
            self.vae_decoder_engine = deepsparse.Engine(model=str(self.vae_decoder_path), batch_size=batch_size)
            self.vae_encoder_engine = deepsparse.Engine(model=str(self.vae_encoder_path), batch_size=batch_size)
            self.text_encoder_engine = deepsparse.Engine(model=str(self.text_encoder_path), batch_size=batch_size)
            self.unet_engine = deepsparse.Engine(model=str(self.unet_path), batch_size=unet_batch_size)
            if os.path.exists(self.text_encoder_2_path):
                self.text_encoder_2_engine = deepsparse.Engine(
                    model=str(self.text_encoder_2_path), batch_size=batch_size
                )

        self.vae_decoder = DeepSparseModelVaeDecoder(self.vae_decoder_engine, self)
        self.unet = DeepSparseModelUnet(self.unet_engine, self)
        self.text_encoder = DeepSparseModelTextEncoder(self.text_encoder_engine, self)
        self.vae_encoder = DeepSparseModelVaeEncoder(self.vae_encoder_engine, self)
        self.text_encoder_2 = (
            DeepSparseModelTextEncoder(
                self.text_encoder_2_engine, self, model_name=DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER
            )
            if os.path.exists(self.text_encoder_2_path)
            else None
        )

        logger.info("Done compiling models")

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
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
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

        vae_decoder_path = new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name
        text_encoder_path = new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name
        unet_path = new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name
        vae_encoder_path = new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / vae_encoder_file_name
        text_encoder_2_path = new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / text_encoder_2_file_name

        logger.info(f"Saving models to {model_save_dir}")

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            vae_decoder_path=vae_decoder_path,
            text_encoder_path=text_encoder_path,
            unet_path=unet_path,
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            scheduler=sub_models.get("scheduler"),
            feature_extractor=sub_models.get("feature_extractor", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            vae_encoder_path=vae_encoder_path,
            text_encoder_2_path=text_encoder_2_path,
            model_save_dir=Path(model_save_dir.name),
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
        task: Optional[str] = None,
    ):
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = Path("tmp")
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
            model_save_dir=save_dir,
        )

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    def to(self, device: str):
        self._device = device.upper()
        return self

    @property
    def device(self) -> str:
        return self._device.lower()


class _DeepSparseDiffusionModelPart:
    """
    For multi-file ONNX models, represents a part of the model.
    """

    CONFIG_NAME = "config.json"

    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseBaseModel, model_name: str = "encoder"):
        self.engine = engine
        self.parent_model = parent_model
        self._model_name = model_name
        config_path = parent_model.model_save_dir / model_name / self.CONFIG_NAME
        self.config = self.parent_model._dict_from_json_file(config_path) if config_path.is_file() else {}
        self.input_dtype = {"latent_sample": np.float64}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def device(self):
        return self.parent_model._device


class DeepSparseModelTextEncoder(_DeepSparseDiffusionModelPart):
    def __init__(
        self,
        engine: deepsparse.Engine,
        parent_model: DeepSparseBaseModel,
        model_name: str = "text_encoder",
    ):
        super().__init__(engine, parent_model, model_name)

    def forward(self, input_ids: np.ndarray):
        onnx_inputs = [np.int32(input_ids)]
        outputs = self.engine(onnx_inputs)
        return outputs


class DeepSparseModelUnet(_DeepSparseDiffusionModelPart):
    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseBaseModel):
        super().__init__(engine, parent_model, "unet")

    def forward(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
    ):
        onnx_inputs = [
            np.float32(sample),
            np.int64(timestep),
            np.float32(encoder_hidden_states),
        ]

        if text_embeds is not None:
            onnx_inputs.append(text_embeds)

        if time_ids is not None:
            onnx_inputs.append(time_ids)

        outputs = self.engine(onnx_inputs, val_inp=False)
        return outputs


class DeepSparseModelVaeDecoder(_DeepSparseDiffusionModelPart):
    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseBaseModel):
        super().__init__(engine, parent_model, "vae_decoder")

    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = [np.float32(latent_sample)]
        outputs = self.engine(onnx_inputs)
        return outputs


class DeepSparseModelVaeEncoder(_DeepSparseDiffusionModelPart):
    def __init__(self, engine: deepsparse.Engine, parent_model: DeepSparseBaseModel):
        super().__init__(engine, parent_model, "vae_encoder")

    def forward(self, sample: np.ndarray):
        onnx_inputs = [np.ascontiguousarray(sample)]
        outputs = self.engine(onnx_inputs)
        return outputs


class DeepSparseStableDiffusionPipeline(DeepSparseStableDiffusionPipelineBase, StableDiffusionPipelineMixin):
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        _height = height
        _width = width

        height = height or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        width = width or self.unet.config.get("sample_size", 64) * self.vae_scale_factor

        if _height and _width:
            logger.info("Custom dimensions provided.")
            self.compile(height=_height, width=_width)

        return StableDiffusionPipelineMixin.__call__(
            self,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )
