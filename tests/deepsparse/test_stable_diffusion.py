#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import unittest
from typing import Dict

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from parameterized import parameterized
from testing_utils import MODEL_DICT, SEED

from optimum.deepsparse import (
    DeepSparseModelTextEncoder,
    DeepSparseModelUnet,
    DeepSparseModelVaeDecoder,
    DeepSparseModelVaeEncoder,
    DeepSparseStableDiffusionPipeline,
)


def _generate_inputs(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


class DeepSparseStableDiffusionPipelineBaseTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}

    MODEL_CLASS = DeepSparseStableDiffusionPipeline

    def test_load_vanilla_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.MODEL_CLASS.from_pretrained(MODEL_DICT["bert"].model_id, export=True)

        self.assertIn(f"does not appear to have a file named {self.MODEL_CLASS.config_name}", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_callback(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id

        def callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
            callback_fn.has_been_called = True
            callback_fn.number_of_steps += 1

        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        callback_fn.has_been_called = False
        callback_fn.number_of_steps = 0
        inputs = self.generate_inputs(height=64, width=64)
        pipeline(**inputs, callback=callback_fn, callback_steps=1)
        self.assertTrue(callback_fn.has_been_called)
        self.assertEqual(callback_fn.number_of_steps, inputs["num_inference_steps"])

    def generate_inputs(self, height=64, width=64, batch_size=1):
        inputs = _generate_inputs(batch_size)
        inputs["height"] = height
        inputs["width"] = width
        return inputs


class DeepSparseStableDiffusionPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseStableDiffusionPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_diffusers(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        deepsparse_pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(deepsparse_pipeline.text_encoder, DeepSparseModelTextEncoder)
        self.assertIsInstance(deepsparse_pipeline.vae_encoder, DeepSparseModelVaeEncoder)
        self.assertIsInstance(deepsparse_pipeline.vae_decoder, DeepSparseModelVaeDecoder)
        self.assertIsInstance(deepsparse_pipeline.unet, DeepSparseModelUnet)
        self.assertIsInstance(deepsparse_pipeline.config, Dict)

        pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        pipeline.safety_checker = None
        batch_size, num_images_per_prompt, height, width = 1, 1, 64, 64

        latents = deepsparse_pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            deepsparse_pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(SEED),
        )

        kwargs = {
            "prompt": "sailing ship in storm by Leonardo da Vinci",
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            outputs = deepsparse_pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images
            # Compare model outputs
            self.assertTrue(np.allclose(outputs, outputs, atol=1e-4))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_image_reproducibility(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        inputs = _generate_inputs()
        height, width = 64, 64
        np.random.seed(SEED)
        outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(SEED)
        outputs_2 = pipeline(**inputs, height=height, width=width)
        outputs_3 = pipeline(**inputs, height=height, width=width)
        # Compare model outputs
        self.assertTrue(np.array_equal(outputs_1.images[0], outputs_2.images[0]))
        self.assertFalse(np.array_equal(outputs_1.images[0], outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_height_width_properties(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        prompt = "Surrealist painting of a floating island with giant clock gears, populated with mythical creatures."
        height = 64
        width = 64
        outputs = pipeline(
            prompt=prompt, num_inference_steps=10, guidance_scale=7.5, height=height, width=width
        ).images
        image_width = outputs[0].size[0]
        image_height = outputs[0].size[1]
        self.assertEqual(image_width, width)
        self.assertEqual(image_height, height)
