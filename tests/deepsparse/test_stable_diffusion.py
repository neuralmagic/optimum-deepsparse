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

import random
import tempfile
import unittest
from typing import Dict

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import floats_tensor, load_image
from parameterized import parameterized
from testing_utils import MODEL_DICT, SEED

from optimum.deepsparse.modeling_diffusion import (
    DeepSparseModelTextEncoder,
    DeepSparseModelUnet,
    DeepSparseModelVaeDecoder,
    DeepSparseModelVaeEncoder,
    DeepSparseStableDiffusionImg2ImgPipeline,
    DeepSparseStableDiffusionInpaintPipeline,
    DeepSparseStableDiffusionPipeline,
    DeepSparseStableDiffusionXLImg2ImgPipeline,
    DeepSparseStableDiffusionXLPipeline,
)
from optimum.onnxruntime import (
    ORTStableDiffusionImg2ImgPipeline,
    ORTStableDiffusionInpaintPipeline,
    ORTStableDiffusionXLImg2ImgPipeline,
    ORTStableDiffusionXLPipeline,
)


def _generate_inputs(batch_size=1):
    inputs = {
        "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
        "num_inference_steps": 3,
        "guidance_scale": 7.5,
        "output_type": "np",
    }
    return inputs


def _create_image(height=128, width=128):
    image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/in_paint/overture-creations-5sI6fQgYIuo.png"
    )
    return image.resize((width, height))


class DeepSparseStableDiffusionPipelineBaseTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}

    MODEL_CLASS = DeepSparseStableDiffusionPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertEqual(pipeline.vae_scale_factor, 2)
        self.assertEqual(pipeline.vae_decoder.config["latent_channels"], 4)
        self.assertEqual(pipeline.unet.config["in_channels"], 4)
        batch_size, height = 2, 128
        for width in [64, 128]:
            inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
            for num_images in [1, 3]:
                outputs = pipeline(**inputs, num_images_per_prompt=num_images).images
                self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

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

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_inputs(batch_size)
        inputs["height"] = height
        inputs["width"] = width
        return inputs


class DeepSparsetableDiffusionImg2ImgPipelineTest(DeepSparseStableDiffusionPipelineBaseTest):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseStableDiffusionImg2ImgPipeline
    ORT_MODEL_CLASS = ORTStableDiffusionImg2ImgPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        inputs = self.generate_inputs()
        inputs["prompt"] = "A painting of a squirrel eating a burger"
        np.random.seed(0)
        output = pipeline(**inputs).images[0, -3:, -3:, -1]
        # https://github.com/huggingface/diffusers/blob/v0.17.1/tests/pipelines/stable_diffusion/test_onnx_stable_diffusion_img2img.py#L71
        expected_slice = np.array([0.69643, 0.58484, 0.50314, 0.58760, 0.55368, 0.59643, 0.51529, 0.41217, 0.49087])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-1))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 2, 3, 128, 64
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        outputs = pipeline(**inputs, num_images_per_prompt=num_images, generator=np.random.RandomState(0)).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_inputs(batch_size)
        inputs["image"] = floats_tensor((batch_size, 3, height, width), rng=random.Random(SEED))
        inputs["strength"] = 0.75
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
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(pipeline.text_encoder, DeepSparseModelTextEncoder)
        self.assertIsInstance(pipeline.vae_encoder, DeepSparseModelVaeEncoder)
        self.assertIsInstance(pipeline.vae_decoder, DeepSparseModelVaeDecoder)
        self.assertIsInstance(pipeline.unet, DeepSparseModelUnet)
        self.assertIsInstance(pipeline.config, Dict)

        pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        pipeline.safety_checker = None
        batch_size, num_images_per_prompt, height, width = 1, 2, 64, 64

        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
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
            ov_outputs = pipeline(latents=latents, output_type=output_type, **kwargs).images
            self.assertIsInstance(ov_outputs, np.ndarray)
            with torch.no_grad():
                outputs = pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images
            # Compare model outputs
            self.assertTrue(np.allclose(ov_outputs, outputs, atol=1e-4))

        # Compare model devices
        self.assertEqual(pipeline.device.type, pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_image_reproducibility(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        inputs = _generate_inputs()
        height, width = 64, 64
        np.random.seed(0)
        outputs_1 = pipeline(**inputs, height=height, width=width)
        np.random.seed(0)
        outputs_2 = pipeline(**inputs, height=height, width=width)
        outputs_3 = pipeline(**inputs, height=height, width=width)
        # Compare model outputs
        self.assertTrue(np.array_equal(outputs_1.images[0], outputs_2.images[0]))
        self.assertFalse(np.array_equal(outputs_1.images[0], outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 3, 4, 128, 64
        prompt = "sailing ship in storm by Leonardo da Vinci"
        self.assertFalse(pipeline.is_dynamic)
        # Verify output shapes requirements not matching the static model don't impact the final outputs
        outputs = pipeline(
            [prompt] * batch_size,
            num_inference_steps=2,
            num_images_per_prompt=num_images,
            height=height + 8,
            width=width,
            output_type="np",
        ).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    # @parameterized.expand(SUPPORTED_ARCHITECTURES)
    # def test_height_width_properties(self, model_arch: str):
    #     model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
    #     model_id = model_info.model_id
    #     self.MODEL_CLASS.from_pretrained(model_id, export=True)
    #     # TO DO: Assert that image generate with given dimensions is of those dimensons


class DeepSparseStableDiffusionInpaintPipelineTest(DeepSparseStableDiffusionPipelineBaseTest):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseStableDiffusionInpaintPipeline
    ORT_MODEL_CLASS = ORTStableDiffusionInpaintPipeline

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_diffusers_pipeline(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 1, 1, 64, 64
        latents = pipeline.prepare_latents(
            batch_size * num_images,
            pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )
        inputs = self.generate_inputs(height=height, width=width)
        outputs = pipeline(**inputs, latents=latents).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

        ort_pipeline = self.ORT_MODEL_CLASS.from_pretrained(model_id, export=True)
        ort_outputs = ort_pipeline(**inputs, latents=latents).images
        self.assertTrue(np.allclose(outputs, ort_outputs, atol=1e-1))

        expected_slice = np.array([0.4692, 0.5260, 0.4005, 0.3609, 0.3259, 0.4676, 0.5593, 0.4728, 0.4411])
        self.assertTrue(np.allclose(outputs[0, -3:, -3:, -1].flatten(), expected_slice, atol=1e-1))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 1, 3, 128, 64
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        outputs = pipeline(**inputs, num_images_per_prompt=num_images, generator=np.random.RandomState(0)).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = super(DeepSparseStableDiffusionInpaintPipelineTest, self).generate_inputs(height, width, batch_size)
        inputs["image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo.png"
        ).resize((width, height))

        inputs["mask_image"] = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/in_paint/overture-creations-5sI6fQgYIuo_mask.png"
        ).resize((width, height))

        return inputs


class DeepSparsetableDiffusionXLPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion-xl",
    ]
    MODEL_CLASS = DeepSparseStableDiffusionXLPipeline
    ORT_MODEL_CLASS = ORTStableDiffusionXLPipeline
    PT_MODEL_CLASS = StableDiffusionXLPipeline
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_diffusers(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        self.assertIsInstance(pipeline.text_encoder, DeepSparseModelTextEncoder)
        self.assertIsInstance(pipeline.text_encoder_2, DeepSparseModelTextEncoder)
        self.assertIsInstance(pipeline.vae_encoder, DeepSparseModelVaeEncoder)
        self.assertIsInstance(pipeline.vae_decoder, DeepSparseModelVaeDecoder)
        self.assertIsInstance(pipeline.unet, DeepSparseModelUnet)
        self.assertIsInstance(pipeline.config, Dict)

        pt_pipeline = self.PT_MODEL_CLASS.from_pretrained(model_id)
        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            pipeline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        kwargs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "num_inference_steps": 1,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
        }

        for output_type in ["latent", "np"]:
            deepsparse_outputs = pipeline(latents=latents, output_type=output_type, **kwargs).images

            self.assertIsInstance(deepsparse_outputs, np.ndarray)
            with torch.no_grad():
                pt_outputs = pt_pipeline(latents=torch.from_numpy(latents), output_type=output_type, **kwargs).images

            # Compare model outputs
            self.assertTrue(np.allclose(deepsparse_outputs, pt_outputs, atol=1e-4))
        # Compare model devices
        self.assertEqual(pipeline.device.type, pipeline.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_image_reproducibility(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)

        # Verify every subcomponent is compiled by default
        for component in {"unet", "vae_encoder", "vae_decoder", "text_encoder", "text_encoder_2"}:
            pass
            # self.assertIsInstance(getattr(pipeline, component).request, CompiledModel)

        batch_size, num_images_per_prompt, height, width = 2, 3, 64, 128
        inputs = _generate_inputs(batch_size)
        np.random.seed(0)
        ov_outputs_1 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        np.random.seed(0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            pipeline = self.MODEL_CLASS.from_pretrained(tmp_dir)
        ov_outputs_2 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        ov_outputs_3 = pipeline(**inputs, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        self.assertTrue(np.array_equal(ov_outputs_1.images[0], ov_outputs_2.images[0]))
        self.assertFalse(np.array_equal(ov_outputs_1.images[0], ov_outputs_3.images[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 3, 4, 128, 64
        self.assertFalse(pipeline.is_dynamic)
        # Verify output shapes requirements not matching the static model don't impact the final outputs
        inputs = _generate_inputs(batch_size)
        outputs = pipeline(**inputs, num_images_per_prompt=num_images, height=height, width=width).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))


class DeepSparseStableDiffusionXLImg2ImgPipelineTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["stable-diffusion-xl", "stable-diffusion-xl-refiner"]
    MODEL_CLASS = DeepSparseStableDiffusionXLImg2ImgPipeline
    ORT_MODEL_CLASS = ORTStableDiffusionXLImg2ImgPipeline
    PT_MODEL_CLASS = StableDiffusionXLImg2ImgPipeline
    ARCH_MODEL_MAP = {}
    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}

    def test_inference(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
        pipeline = self.MODEL_CLASS.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)
            pipeline = self.MODEL_CLASS.from_pretrained(tmp_dir)

        inputs = self.generate_inputs()
        np.random.seed(0)
        output = pipeline(**inputs).images[0, -3:, -3:, -1]
        expected_slice = np.array([0.5675, 0.5108, 0.4758, 0.5280, 0.5080, 0.5473, 0.4789, 0.4286, 0.4861])
        self.assertTrue(np.allclose(output.flatten(), expected_slice, atol=1e-3))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_num_images_per_prompt_static_model(self, model_arch: str):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        pipeline = self.MODEL_CLASS.from_pretrained(model_id, export=True)
        batch_size, num_images, height, width = 2, 3, 128, 64
        inputs = self.generate_inputs(height=height, width=width, batch_size=batch_size)
        outputs = pipeline(**inputs, num_images_per_prompt=num_images, generator=np.random.RandomState(0)).images
        self.assertEqual(outputs.shape, (batch_size * num_images, height, width, 3))

    def generate_inputs(self, height=128, width=128, batch_size=1):
        inputs = _generate_inputs(batch_size)
        inputs["image"] = floats_tensor((batch_size, 3, height, width), rng=random.Random(SEED))
        inputs["strength"] = 0.75
        return inputs
