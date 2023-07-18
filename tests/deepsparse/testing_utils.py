import os
import unittest

import numpy as np
import torch


SEED = 42


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}

SEQLEN = 128
DEFAULT_TOKEN_SHAPES = f"[1,{SEQLEN}]"
DEFAULT_PADDING_KWARGS = {
    "padding": "max_length",
    "max_length": SEQLEN,
    "truncation": True,
}
DEFAULT_IMAGENET_SHAPES = "[1,3,224,224]"


class ModelInfo:
    def __init__(self, model_id: str, input_shapes: str = None, padding_kwargs: dict = None):
        self.model_id = model_id
        self.input_shapes = input_shapes
        self.padding_kwargs = padding_kwargs


MODEL_DICT = {
    "albert": ModelInfo("hf-internal-testing/tiny-random-AlbertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "beit": ModelInfo("hf-internal-testing/tiny-random-BeitForImageClassification", "[1,3,30,30]"),
    "bert": ModelInfo("hf-internal-testing/tiny-random-BertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "bart": ModelInfo("hf-internal-testing/tiny-random-bart", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "camembert": ModelInfo("hf-internal-testing/tiny-random-camembert", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "convbert": ModelInfo(
        "hf-internal-testing/tiny-random-ConvBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "deberta": ModelInfo("hf-internal-testing/tiny-random-DebertaModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "deberta_v2": ModelInfo(
        "hf-internal-testing/tiny-random-DebertaV2Model", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "deit": ModelInfo("hf-internal-testing/tiny-random-DeiTModel", "[1,3,30,30]"),
    "convnext": ModelInfo("hf-internal-testing/tiny-random-convnext", DEFAULT_IMAGENET_SHAPES),
    "detr": ModelInfo("hf-internal-testing/tiny-random-detr"),
    "distilbert": ModelInfo(
        "hf-internal-testing/tiny-random-DistilBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "ibert": ModelInfo("hf-internal-testing/tiny-random-IBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "levit": ModelInfo("hf-internal-testing/tiny-random-LevitModel", "[1,3,64,64]"),
    "mbart": ModelInfo("hf-internal-testing/tiny-random-mbart", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "mobilebert": ModelInfo(
        "hf-internal-testing/tiny-random-MobileBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "mobilenet_v1": ModelInfo("google/mobilenet_v1_0.75_192", "[1,3,192,192]"),
    "mobilenet_v2": ModelInfo("hf-internal-testing/tiny-random-MobileNetV2Model", "[1,3,32,32]"),
    "mobilevit": ModelInfo("hf-internal-testing/tiny-random-mobilevit", "[1,3,256,256]"),
    "nystromformer": ModelInfo(
        "hf-internal-testing/tiny-random-NystromformerModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "poolformer": ModelInfo("hf-internal-testing/tiny-random-PoolFormerModel", "[1,3,64,64]"),
    "resnet": ModelInfo("hf-internal-testing/tiny-random-resnet", DEFAULT_IMAGENET_SHAPES),
    "roberta": ModelInfo("hf-internal-testing/tiny-random-RobertaModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "roformer": ModelInfo(
        "hf-internal-testing/tiny-random-RoFormerModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "segformer": ModelInfo("hf-internal-testing/tiny-random-SegformerModel", "[1,3,64,64]"),
    "squeezebert": ModelInfo(
        "hf-internal-testing/tiny-random-SqueezeBertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS
    ),
    "swin": ModelInfo("hf-internal-testing/tiny-random-SwinModel", "[1,3,32,32]"),
    "t5": ModelInfo("hf-internal-testing/tiny-random-t5"),
    "vit": ModelInfo("hf-internal-testing/tiny-random-vit", "[1,3,30,30]"),
    "yolos": ModelInfo("hf-internal-testing/tiny-random-YolosModel", "[1,3,30,30]"),
    "xlm": ModelInfo("hf-internal-testing/tiny-random-XLMModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "xlm_roberta": ModelInfo("hf-internal-testing/tiny-xlm-roberta", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    # audio models
    "audio_spectrogram_transformer": ModelInfo("Ericwang/tiny-random-ast", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "data2vec_audio": ModelInfo("Ericwang/tiny-random-ast", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "hubert": ModelInfo("hf-internal-testing/tiny-random-HubertModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "sew": ModelInfo("hf-internal-testing/tiny-random-SEWModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "sew_d": ModelInfo("hf-internal-testing/tiny-random-SEWDModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "unispeech": ModelInfo("hf-internal-testing/tiny-random-unispeech", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "unispeech_sat": ModelInfo("hf-internal-testing/tiny-random-UnispeechSatModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "wavlm": ModelInfo("hf-internal-testing/tiny-random-WavlmModel", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "wav2vec2": ModelInfo("hf-internal-testing/tiny-random-Wav2Vec2Model", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),
    "wav2vec2-conformer": ModelInfo("hf-internal-testing/tiny-random-wav2vec2-conformer", DEFAULT_TOKEN_SHAPES, DEFAULT_PADDING_KWARGS),

}


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if use_auth_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case
