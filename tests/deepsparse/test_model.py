import os
import shutil
import tempfile
import unittest

from huggingface_hub.constants import default_cache_path
from testing_utils import MODEL_DICT
from transformers import (
    PretrainedConfig,
)

import deepsparse
from optimum.deepsparse import DeepSparseModel, DeepSparseModelForSequenceClassification
from optimum.onnxruntime import (
    ONNX_WEIGHTS_NAME,
)
from optimum.utils import CONFIG_NAME


class DeepSparseModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.LOCAL_MODEL_PATH = "assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.TINY_ONNX_MODEL_ID = "fxmarty/resnet-tiny-beans"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"

    def test_load_model_from_local_path(self):
        model = DeepSparseModel.from_pretrained(self.LOCAL_MODEL_PATH)
        model.compile()
        self.assertIsInstance(model.engine, deepsparse.Engine)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = DeepSparseModel.from_pretrained(self.ONNX_MODEL_ID)
        model.compile()
        self.assertIsInstance(model.engine, deepsparse.Engine)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        # does not pass with DeepSparseModel as it does not have export_feature attribute
        model = DeepSparseModelForSequenceClassification.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder", subfolder="my_subfolder", export=True
        )
        model.compile()
        self.assertIsInstance(model.engine, deepsparse.Engine)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = DeepSparseModel.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-onnx-subfolder", subfolder="my_subfolder"
        )
        model.compile()
        self.assertIsInstance(model.engine, deepsparse.Engine)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = DeepSparseModel.from_pretrained(self.TINY_ONNX_MODEL_ID)  # caching

        model = DeepSparseModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)
        model.compile()
        self.assertIsInstance(model.engine, deepsparse.Engine)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_MODEL_ID.replace("/", "--"))

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = DeepSparseModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

    def test_load_model_from_hub_without_onnx_model(self):
        with self.assertRaises(FileNotFoundError):
            DeepSparseModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)

    # @require_hf_token
    # def test_load_model_from_hub_private(self):
    #     model = DeepSparseModel.from_pretrained(self.ONNX_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
    #     model.compile()
    #     self.assertIsInstance(model.engine, deepsparse.Engine)
    #     self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = DeepSparseModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and ONNX exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_save_load_ort_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
            model = DeepSparseModelForSequenceClassification.from_pretrained(MODEL_DICT["bert"].model_id, export=True)
            model.save_pretrained(tmpdirname)
            print(model.model)

            # verify external data is exported
            print(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)
            self.assertIn(ONNX_WEIGHTS_NAME + "_data", folder_contents)
            # verify loading from local folder works
            model = DeepSparseModelForSequenceClassification.from_pretrained(tmpdirname, export=False)
            os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")

    # @require_hf_token
    # def test_save_model_from_hub(self):
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         model = DeepSparseModel.from_pretrained(self.LOCAL_MODEL_PATH)
    #         model.save_pretrained(
    #             tmpdirname,
    #             use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
    #             push_to_hub=True,
    #             repository_id=self.HUB_REPOSITORY,
    #             private=True,
    #         )

    # @require_hf_token
    # def test_push_ort_model_with_external_data_to_hub(self):
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         os.environ["FORCE_ONNX_EXTERNAL_DATA"] = "1"  # force exporting small model with external data
    #         model = DeepSparseModelForSequenceClassification.from_pretrained(MODEL_DICT["bert"].model_id, export=True)
    #         model.save_pretrained(
    #             tmpdirname + "/onnx",
    #             use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
    #             repository_id=MODEL_DICT["bert"].model_id.split("/")[-1] + "-onnx",
    #             private=True,
    #             push_to_hub=True,
    #         )

    #         # verify loading from hub works
    #         model = DeepSparseModelForSequenceClassification.from_pretrained(
    #             MODEL_DICT["bert"].model_id + "-onnx",
    #             export=False,
    #             use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
    #         )
    #         os.environ.pop("FORCE_ONNX_EXTERNAL_DATA")
