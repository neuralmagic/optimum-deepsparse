import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import tqdm
import transformers
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    EvalPrediction,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    ImageClassifierOutput,
)

from deepsparse import Engine

from .modeling_base import DeepSparseBaseModel


logger = logging.getLogger(__name__)


_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

MODEL_START_DOCSTRING = r"""
    This model inherits from [`optimum.deepsparse.DeepSparseBaseModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`onnx`): an onnx model.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
        deepsparse_config (`Optional[Dict]`, defaults to `None`):
            The dictionnary containing the informations related to the model compilation.
"""

IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.Tensor`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""


class DeepSparseModel(DeepSparseBaseModel):
    auto_model_class = AutoModel

    def __init__(
        self,
        model,
        config: transformers.PretrainedConfig = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        label_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(model, config, **kwargs)
        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)

        # Evaluation args
        self.compute_metrics = compute_metrics
        self.label_names = ["labels"] if label_names is None else label_names

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def evaluation_loop(self, dataset: Dataset):
        """
        Run evaluation and returns metrics and predictions.

        Args:
            dataset (`datasets.Dataset`):
                Dataset to use for the evaluation step.
        """
        logger.info("***** Running evaluation *****")

        from transformers import EvalPrediction
        from transformers.trainer_pt_utils import nested_concat
        from transformers.trainer_utils import EvalLoopOutput

        all_preds = None
        all_labels = None
        for step, inputs in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            inputs = [np.array([inputs[key]]) for k, key in enumerate(self.input_names) if key in inputs]

            preds = None
            if len(preds) == 1:
                preds = preds[0].numpy()
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))


IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example of image classification using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, input_shape_dict="dict('pixel_values': [1, 3, 224, 224])", output_shape_dict="dict("logits": [1, 1000])",)
    >>> pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> outputs = pipe(url)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a ImageClassifierOutput for image classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForImageClassification(DeepSparseModel):
    export_feature = "image-classification"
    auto_model_class = AutoModelForImageClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.input_names = ["pixel_values"]

    @add_start_docstrings_to_model_forward(
        IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="DeepSparseModelForImageClassification",
            checkpoint="microsoft/resnet50",
        )
    )
    def forward(
        self,
        pixel_values: None,
        **kwargs,
    ):
        if not self.deepsparse_engine:
            self.deepsparse_engine = Engine(model=self.model, batch_size=1)

        np_inputs = isinstance(pixel_values, np.ndarray)
        if not np_inputs:
            pixel_values = np.array(pixel_values)

        outputs = self.deepsparse_engine(list(np.expand_dims(pixel_values, axis=0)))
        logits = torch.from_numpy(outputs[0]) if not np_inputs else outputs[0]
        return ImageClassifierOutput(logits=logits)
