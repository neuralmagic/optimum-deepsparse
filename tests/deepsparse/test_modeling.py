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

import pytest
import numpy as np
import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor

import deepsparse
from optimum.deepsparse import DeepSparseModelForImageClassification, DeepSparseModelForSequenceClassification
from optimum.utils import (
    logging,
)


SEED = 42

logger = logging.get_logger()

SEQLEN = 128
DEFAULT_TOKEN_SHAPES = f"[1,{SEQLEN}]"
DEFAULT_PADDING_KWARGS = {"padding":"max_length",
    "max_length": SEQLEN,
    "truncation":True,}

class ModelInfo:
    def __init__(self, model_id: str, input_shapes: str = None, padding_kwargs: dict = None):
        self.model_id = model_id
        self.input_shapes = input_shapes
        self.padding_kwargs = padding_kwargs

MODEL_DICT = {
    "albert": ModelInfo("hf-internal-testing/tiny-random-AlbertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "beit": ModelInfo("hf-internal-testing/tiny-random-BeitForImageClassification"),
    "bert": ModelInfo("hf-internal-testing/tiny-random-BertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "bart": ModelInfo("hf-internal-testing/tiny-random-bart", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "camembert": ModelInfo("hf-internal-testing/tiny-random-camembert", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "convbert": ModelInfo("hf-internal-testing/tiny-random-ConvBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "deberta": ModelInfo("hf-internal-testing/tiny-random-DebertaModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "deberta_v2": ModelInfo("hf-internal-testing/tiny-random-DebertaV2Model", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "deit": ModelInfo("hf-internal-testing/tiny-random-DeiTModel"),
    "convnext": ModelInfo("hf-internal-testing/tiny-random-convnext"),
    "detr": ModelInfo("hf-internal-testing/tiny-random-detr"),
    "distilbert": ModelInfo("hf-internal-testing/tiny-random-DistilBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "ibert": ModelInfo("hf-internal-testing/tiny-random-IBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "mbart": ModelInfo("hf-internal-testing/tiny-random-mbart", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "mobilebert": ModelInfo("hf-internal-testing/tiny-random-MobileBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "mobilenet_v1": ModelInfo("google/mobilenet_v1_0.75_192", "[1,3,192,192]"),
    "mobilenet_v2": ModelInfo("hf-internal-testing/tiny-random-MobileNetV2Model", "[1,3,32,32]"),
    "mobilevit": ModelInfo("hf-internal-testing/tiny-random-mobilevit"),
    "nystromformer": ModelInfo("hf-internal-testing/tiny-random-NystromformerModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "resnet": ModelInfo("hf-internal-testing/tiny-random-resnet", "[1,3,224,224]"),
    "roberta": ModelInfo("hf-internal-testing/tiny-random-RobertaModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "roformer": ModelInfo("hf-internal-testing/tiny-random-RoFormerModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "squeezebert": ModelInfo("hf-internal-testing/tiny-random-SqueezeBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "swin": ModelInfo("hf-internal-testing/tiny-random-SwinModel"),
    "t5": ModelInfo("hf-internal-testing/tiny-random-t5"),
    "vit": ModelInfo("hf-internal-testing/tiny-random-vit"),
    "yolos": ModelInfo("hf-internal-testing/tiny-random-YolosModel"),
    "xlm": ModelInfo("hf-internal-testing/tiny-random-XLMModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "xlm_roberta": ModelInfo("hf-internal-testing/tiny-xlm-roberta", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
}


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

    ARCH_MODEL_MAP = {
    }

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
            self.assertIsInstance(onnx_model.deepsparse_engine, deepsparse.Engine)
            self.assertTrue(onnx_model.deepsparse_engine.fraction_of_supported_ops >= 0.9)

            # compare tensor outputs
            # print("=====")
            # print(torch.Tensor(onnx_outputs.logits))
            # print("-----")
            # print(transformers_outputs.logits)
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
        self.assertTrue(onnx_model.deepsparse_engine.fraction_of_supported_ops >= 0.9)

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
            "typeform/distilbert-base-uncased-mnli", export=True#, input_shapes=DEFAULT_TOKEN_SHAPES
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
        # self.assertTrue(onnx_model.deepsparse_engine.fraction_of_supported_ops >= 0.9)
