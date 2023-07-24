import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
import transformers
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    EvalPrediction,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    ImageClassifierOutput,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SemanticSegmenterOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .modeling_base import DeepSparseBaseModel


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"
_PROCESSOR_FOR_DOC = "AutoProcessor"

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

TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
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
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(pixel_values, torch.Tensor)
        if use_torch:
            pixel_values = pixel_values.cpu().detach().numpy()

        outputs = self.engine(list(np.expand_dims(pixel_values, axis=0)))
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return ImageClassifierOutput(logits=logits)


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.deepsparse import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 2]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> nm_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    >>> text = "Hello, my dog is cute"
    >>> pred = nm_classifier(text)
    ```

    Example using zero-shot-classification `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> model = {model_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> nm_z0 = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    >>> sequence_to_classify = "Who are you voting for in 2020?"
    >>> candidate_labels = ["Europe", "public health", "politics", "elections"]
    >>> pred = nm_z0(sequence_to_classify, candidate_labels, multi_label=True)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForSequenceClassification(DeepSparseModel):
    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForSequenceClassification",
            checkpoint="distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            attention_mask = attention_mask.cpu().detach().numpy()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(logits=logits)


AUDIO_CLASSIFICATION_EXAMPLE = r"""
    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}
     >>> from datasets import load_dataset, Audio
    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> audio_classifier = pipeline("audio-classification", model=model, feature_extractor=processor)

    >>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
    >>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> audio_file = dataset[0]["audio"]["path"]
    >>> pred = audio_classifier(audio_file)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a audio classification
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForAudioClassification(DeepSparseModel):
    auto_model_class = AutoModelForAudioClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + AUDIO_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="DeepSparseModelForAudioClassification",
            checkpoint="optimum/hubert-base-superb-ks",
        )
    )
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_values, torch.Tensor)
        if use_torch:
            input_values = input_values.cpu().detach().numpy()

        inputs = [input_values]

        outputs = self.engine(inputs)
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(logits=logits)


MASKEDLM_EXAMPLE = r"""
    Example of feature extraction:
    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 8, 28996]
    ```
    Example using `transformers.pipeline`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    >>> text = "The capital of France is [MASK]."
    >>> pr
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model for MaskedLM
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForMaskedLM(DeepSparseModel):
    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKEDLM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForMaskedLM",
            checkpoint="optimum/bert-base-uncased-for-fill-mask",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            attention_mask = attention_mask.cpu().detach().numpy()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return MaskedLMOutput(logits=logits)


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of mutliple choice:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> num_choices = 4
    >>> first_sentence = ["Members of the procession walk down the street holding small horn brass instruments."] * num_choices
    >>> second_sentence = [
    ...     "A drum line passes by walking down the street playing their instruments.",
    ...     "A drum line has heard approaching the city.",
    ...     "A drum line arrives and they're outside dancing and asleep.",
    ...     "A drum line turns the lead singer watches the performance."
    ... ]
    >>> inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

    # Unflatten the inputs values expanding it to the shape [batch_size, num_choices, seq_length]
    >>> for k, v in inputs.items():
    ...     inputs[k] = [v[i: i + num_choices] for i in range(0, len(v), num_choices)]
    >>> inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForMultipleChoice(DeepSparseModel):
    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForMultipleChoice",
            checkpoint="ehdwns1516/bert-base-uncased_SWAG",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            attention_mask = attention_mask.cpu().detach().numpy()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return MultipleChoiceModelOutput(logits=logits)


FEATURE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction with DeepSparse:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 12, 384]
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> pred = onnx_extractor(text)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with BaseModelOutput for feature-extraction tasks.
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForFeatureExtraction(DeepSparseModel):
    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            if attention_mask is None:
                attention_mask = np.ones_like(input_ids)
            else:
                attention_mask = attention_mask.cpu().detach().numpy()

            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        last_hidden_state = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state=last_hidden_state)


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.deepsparse import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question = "What is DeepSparse?"
    >>> context = "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application."
    >>> inputs = tokenizer(question, context, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question = "What is DeepSparse?"
    >>> context = "DeepSparse is sparsity-aware inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application."

    >>> qa = pipeline("text-classification", model=model, tokenizer=tokenizer)

     >>> pred = qa(question = question, context = context)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model for question answering
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForQuestionAnswering(DeepSparseModel):
    auto_model_class = AutoModelForQuestionAnswering

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForQuestionAnswering",
            checkpoint="distilbert-base-uncased",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            attention_mask = attention_mask.cpu().detach().numpy()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        start_logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]
        end_logits = torch.from_numpy(outputs[1]) if use_torch else outputs[1]

        # converts output to namedtuple for pipelines post-processing
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


CUSTOM_TASKS_EXAMPLE = r"""
    Example of custom tasks(e.g. a sentence transformers taking `pooler_output` as output):

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("I love burritos!", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output
    ```

    Example using `transformers.pipelines`(only if the task is supported):

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "I love burritos!"
    >>> pred = onnx_extractor(text)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model for for any custom tasks
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForCustomTasks(DeepSparseModel):
    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + CUSTOM_TASKS_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForCustomTasks",
            checkpoint="optimum/sbert-all-MiniLM-L6-with-pooler",
        )
    )
    def forward(self, **kwargs):
        self.compile()

        use_torch = isinstance(next(iter(kwargs.values())), torch.Tensor)
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = self._prepare_onnx_inputs(use_torch=use_torch, **kwargs)
        onnx_outputs = self.engine(onnx_inputs)
        outputs = self._prepare_onnx_outputs(onnx_outputs, use_torch=use_torch)

        # converts output to namedtuple for pipelines post-processing
        return ModelOutput(outputs)

    def _prepare_onnx_inputs(self, use_torch: bool, **kwargs):
        onnx_inputs = {}
        inputs = []
        # converts pytorch inputs into numpy inputs for onnx
        for input in self.engine.input_names:
            onnx_inputs[input] = kwargs.pop(input)

            if use_torch:
                onnx_inputs[input] = onnx_inputs[input].cpu().detach().numpy()

        for key in onnx_inputs.keys():
            inputs.append(onnx_inputs[key])

        return inputs

    def _prepare_onnx_outputs(self, onnx_outputs, use_torch: bool):
        outputs = {}
        # converts onnxruntime outputs into tensor for standard outputs
        for idx, output in enumerate(self.engine.output_names):
            outputs[output] = onnx_outputs[idx]
            if use_torch:
                outputs[output] = torch.from_numpy(outputs[output])

        return outputs


SEMANTIC_SEGMENTATION_EXAMPLE = r"""
    Example of semantic segmentation using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, input_shape_dict="dict('pixel_values': [1, 3, 224, 224])", output_shape_dict="dict("logits": [1, 1000])",)
    >>> pipe = pipeline("semantic-segmentation", model=model, feature_extractor=preprocessor)
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> outputs = pipe(url)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a SemanticSegmenterOutput for semantic segmentation tasks.
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForSemanticSegmentation(DeepSparseModel):
    export_feature = "semantic-segmentation"
    auto_model_class = AutoModelForSemanticSegmentation

    @add_start_docstrings_to_model_forward(
        IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + SEMANTIC_SEGMENTATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="DeepSparseModelForSemanticSegmentation",
            checkpoint="nvidia/segformer-b0-finetuned-ade-512-512",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(pixel_values, torch.Tensor)
        if use_torch:
            pixel_values = pixel_values.cpu().detach().numpy()

        outputs = self.engine(list(np.expand_dims(pixel_values, axis=0)))
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return SemanticSegmenterOutput(logits=logits)


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.deepsparse import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("We are flying from Texas to California", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 2]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.deepsparse import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> nm_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)

    >>> text = "We are flying from Texas to California"
    >>> pred = nm_classifier(text)
    ```
"""


@add_start_docstrings(
    """
    DeepSparse Model with a token classification head on top (a linear layer on top of the
    pooled output)
    """,
    MODEL_START_DOCSTRING,
)
class DeepSparseModelForTokenClassification(DeepSparseModel):
    auto_model_class = AutoModelForTokenClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="DeepSparseModelForTokenClassification",
            checkpoint="distilbert-base-uncased",
        )
    )
    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_type_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        self.compile()

        use_torch = isinstance(input_ids, torch.Tensor)
        if use_torch:
            input_ids = input_ids.cpu().detach().numpy()
            attention_mask = attention_mask.cpu().detach().numpy()
            if token_type_ids is not None:
                token_type_ids = token_type_ids.cpu().detach().numpy()

        inputs = [input_ids, attention_mask]
        if token_type_ids is not None:
            inputs.append(token_type_ids)

        outputs = self.engine(inputs)
        logits = torch.from_numpy(outputs[0]) if use_torch else outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return TokenClassifierOutput(logits=logits)
