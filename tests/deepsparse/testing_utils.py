import os


SEED = 42

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

def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if use_auth_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case