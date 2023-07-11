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

import numpy as np
import pytest
import torch
from parameterized import parameterized
from testing_utils import MODEL_DICT, SEED
from transformers import (
    AutoModelForSequenceClassification,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor

import deepsparse
from optimum.deepsparse import DeepSparseModelForSequenceClassification
from optimum.utils import (
    logging,
)


logger = logging.get_logger()


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}


class DeepSparseModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        # "bart",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta_v2",
        "distilbert",
        # "ibert",
        # "mbart",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        # "squeezebert",
        # "xlm",
        # "xlm_roberta",
    ]

    ARCH_MODEL_MAP = {}

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.MODEL_CLASS.from_pretrained(MODEL_DICT["t5"].model_id, export=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        # model_args = {"test_name": model_arch, "model_arch": model_arch}
        # self._setup(model_args)

        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        input_shapes = model_info.input_shapes
        padding_kwargs = model_info.padding_kwargs
        # onnx_model = self.MODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        onnx_model = self.MODEL_CLASS.from_pretrained(model_id, export=True, input_shapes=input_shapes)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt", **padding_kwargs)
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type, **padding_kwargs)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_model.engine, deepsparse.Engine)
            self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.9)
            self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-1))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_nm_model(self, model_arch):
        # model_args = {"test_name": model_arch, "model_arch": model_arch}
        # self._setup(model_args)

        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        input_shapes = model_info.input_shapes
        padding_kwargs = model_info.padding_kwargs

        onnx_model = self.MODEL_CLASS.from_pretrained(model_id, export=True, input_shapes=input_shapes)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, **padding_kwargs)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)
        self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.9)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    def test_pipeline_zero_shot_classification(self):
        onnx_model = self.MODEL_CLASS.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", export=True  # , input_shapes=DEFAULT_TOKEN_SHAPES
        )
        tokenizer = get_preprocessor("typeform/distilbert-base-uncased-mnli")
        # TODO: padding doesn't work
        # pipe = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer, **DEFAULT_PADDING_KWARGS)
        pipe = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pipe(
            sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template
        )

        # compare model output class
        self.assertTrue(all(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(all(isinstance(label, str) for label in outputs["labels"]))
        # TODO: fix padding
        # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.9)
