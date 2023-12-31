# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import unittest

import requests
import torch
from parameterized import parameterized
from PIL import Image
from testing_utils import SEED, TENSOR_ALIAS_TO_TYPE
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    PretrainedConfig,
    pipeline,
    set_seed,
)

from optimum.deepsparse import DeepSparseModelForImageClassification
from optimum.utils import (
    logging,
)


logger = logging.get_logger()


MODEL_DICT = {
    "mobilenet_v1": [
        "google/mobilenet_v1_0.75_192",
        "[1, 3, 192, 192]",
        {"pixel_values": [1, 3, 192, 192]},
        {"logits": [1, 1001]},
    ],
    "mobilenet_v2": [
        "hf-internal-testing/tiny-random-MobileNetV2Model",
        "[1, 3, 32, 32]",
        {"pixel_values": [1, 3, 32, 32]},
        {"logits": [1, 2]},
    ],
    "resnet": [
        "hf-internal-testing/tiny-random-resnet",
        "[1, 3, 224, 224]",
        {"pixel_values": [1, 3, 224, 224]},
        {"logits": [1, 1000]},
    ],
}


class DeepSparseModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "mobilenet_v1",
        "mobilenet_v2",
        "resnet",
    ]

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id, input_shapes, input_shape_dict, output_shape_dict = MODEL_DICT[model_arch]
        set_seed(SEED)
        nm_model1 = DeepSparseModelForImageClassification.from_pretrained(
            model_id, export=True, input_shapes=input_shapes
        )
        nm_model2 = DeepSparseModelForImageClassification.from_pretrained(
            model_id, export=True, input_shape_dict=input_shape_dict, output_shape_dict=output_shape_dict
        )
        self.assertIsInstance(nm_model1.config, PretrainedConfig)
        self.assertIsInstance(nm_model2.config, PretrainedConfig)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            nm_outputs1 = nm_model1(**inputs)
            nm_outputs2 = nm_model2(**inputs)
            print(nm_model1.engine)
            print(nm_model2.engine)
            self.assertTrue(nm_model1.engine.fraction_of_supported_ops >= 0.9)
            self.assertTrue(nm_model2.engine.fraction_of_supported_ops >= 0.9)
            self.assertIn("logits", nm_outputs1)
            self.assertIn("logits", nm_outputs2)
            self.assertIsInstance(nm_outputs1.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(nm_outputs2.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(nm_outputs1.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(torch.allclose(torch.Tensor(nm_outputs2.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id, input_shapes, input_shape_dict, output_shape_dict = MODEL_DICT[model_arch]
        model1 = DeepSparseModelForImageClassification.from_pretrained(
            model_id, export=True, input_shapes=input_shapes
        )
        model2 = DeepSparseModelForImageClassification.from_pretrained(
            model_id, export=True, input_shape_dict=input_shape_dict, output_shape_dict=output_shape_dict
        )
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe1 = pipeline("image-classification", model=model1, feature_extractor=preprocessor)
        pipe2 = pipeline("image-classification", model=model2, feature_extractor=preprocessor)
        outputs1 = pipe1("http://images.cocodataset.org/val2017/000000039769.jpg")
        outputs2 = pipe2("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertGreaterEqual(outputs1[0]["score"], 0.0)
        self.assertGreaterEqual(outputs2[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs1[0]["label"], str))
        self.assertTrue(isinstance(outputs2[0]["label"], str))
        gc.collect()
