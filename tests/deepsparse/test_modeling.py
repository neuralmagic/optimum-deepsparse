import gc
import unittest

import numpy as np
import pytest
import requests
import torch
from parameterized import parameterized
from PIL import Image
from testing_utils import MODEL_DICT, SEED, TENSOR_ALIAS_TO_TYPE
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor

import deepsparse
from optimum.deepsparse import (
    DeepSparseModelForAudioClassification,
    DeepSparseModelForImageClassification,
    DeepSparseModelForMaskedLM,
    DeepSparseModelForSequenceClassification,
    DeepSparseModelForCustomTasks,
)
from optimum.utils import (
    logging,
)


logger = logging.get_logger()


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
            self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

            # compare tensor outputs
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
        self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

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
        # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)


class DeepSparseModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "beit",
        "convnext",
        "deit",
        # "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "poolformer",
        "resnet",
        "segformer",
        # "swin", TODO(mgoin): Fix in nightly
        "vit",
    ]

    ARCH_MODEL_MAP = {}

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseModelForImageClassification
    TASK = "image-classification"

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

        onnx_model = self.MODEL_CLASS.from_pretrained(model_id, export=True, input_shapes=input_shapes)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            onnx_outputs = onnx_model(**inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_model.engine, deepsparse.Engine)
            self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.75)

            # compare tensor outputs
            print("MAX_DIFF: ", torch.max(torch.abs(torch.Tensor(onnx_outputs.logits) - trtfs_outputs.logits)))
            if model_arch not in ["deit", "poolformer", "segformer", "swin"]:
                self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=1e-3))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_nm_model(self, model_arch):
        # model_args = {"test_name": model_arch, "model_arch": model_arch}
        # self._setup(model_args)

        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        input_shapes = model_info.input_shapes

        onnx_model = self.MODEL_CLASS.from_pretrained(model_id, export=True, input_shapes=input_shapes)
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.75)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-classification")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))


class DeepSparseModelForAudioClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "audio_spectrogram_transformer",
        "data2vec_audio",
        "hubert",
        "sew",
        "sew_d",
        "unispeech",
        "unispeech_sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    ARCH_MODEL_MAP = {}

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseModelForAudioClassification
    TASK = "audio-classification"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

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
        # onnx_model = self.MODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        onnx_model = self.MODEL_CLASS.from_pretrained(
            model_id,
            export=True,
            #    input_shapes=input_shapes
        )

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(
            self._generate_random_audio_data(),
            return_tensors="pt",
            #  **padding_kwargs
        )

        with torch.no_grad():
            transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(
                self._generate_random_audio_data(),
                return_tensors=input_type,
                #   **padding_kwargs
            )
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            # self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-4))

            self.assertIsInstance(onnx_model.engine, deepsparse.Engine)
            # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_nm_model(self, model_arch):
        # model_args = {"test_name": model_arch, "model_arch": model_arch}
        # self._setup(model_args)

        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id

        onnx_model = self.MODEL_CLASS.from_pretrained(
            model_id,
            export=True,
            #   input_shapes=input_shapes
        )
        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline(
            "audio-classification",
            model=onnx_model,
            feature_extractor=processor,
            sampling_rate=220,
            # **padding_kwargs
        )
        outputs = pipe(data)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)
        # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("audio-classification")
        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)


class DeepSparseModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "camembert",
        "convbert",
        # "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        # "electra",
        # "flaubert",
        # "ibert",
        "mobilebert",
        # "perceiver",
        "roberta",
        "roformer",
        # "squeezebert",
        # "xlm",
        # "xlm_roberta",
    ]

    ARCH_MODEL_MAP = {}

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseModelForMaskedLM
    TASK = "fill-mask"

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
        # onnx_model = self.MODEL_CLASS.from_pretrained(self.onnx_model_dirs[model_arch])
        onnx_model = self.MODEL_CLASS.from_pretrained(
            model_id,
            export=True,
            #   input_shapes=input_shapes
        )

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(
            text,
            return_tensors="pt",
            #    **padding_kwargs
        )
        with torch.no_grad():
            transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(
                text,
                return_tensors=input_type,
                #    **padding_kwargs
            )
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_model.engine, deepsparse.Engine)
            # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

            # compare tensor outputs
            # self.assertTrue(torch.allclose(torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=1e-1))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_nm_model(self, model_arch):
        # model_args = {"test_name": model_arch, "model_arch": model_arch}
        # self._setup(model_args)

        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id

        onnx_model = self.MODEL_CLASS.from_pretrained(
            model_id,
            export=True,
            #    input_shapes=input_shapes
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline(
            "fill-mask",
            model=onnx_model,
            tokenizer=tokenizer,
            #  **padding_kwargs
        )
        MASK_TOKEN = tokenizer.mask_token
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)
        # self.assertTrue(onnx_model.engine.fraction_of_supported_ops >= 0.8)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("fill-mask")
        text = f"The capital of France is {pipe.tokenizer.mask_token}."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

class DeepSparseModelForCustomTasksMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
       "sbert"
    ]

    ARCH_MODEL_MAP = {}

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    MODEL_CLASS = DeepSparseModelForCustomTasks

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.MODEL_CLASS.from_pretrained(MODEL_DICT["t5"].model_id, export=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_model_call(self, model_arch):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        model = self.MODEL_CLASS.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            outputs = model(**tokens)
            self.assertIsInstance(outputs.pooler_output, self.TENSOR_ALIAS_TO_TYPE[input_type])

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_info = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_DICT[model_arch]
        model_id = model_info.model_id
        onnx_model = self.MODEL_CLASS.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))



