# optimum-deepsparse

Accelerated inference of 🤗 models on CPUs using the [DeepSparse Inference Runtime](https://github.com/neuralmagic/deepsparse).

[![DeepSparse Modeling / Python - Test](https://github.com/neuralmagic/optimum-deepsparse/actions/workflows/test_check.yaml/badge.svg)](https://github.com/neuralmagic/optimum-deepsparse/actions/workflows/test_check.yaml)
[![DeepSparse Modeling Nightly](https://github.com/neuralmagic/optimum-deepsparse/actions/workflows/test_nightly.yaml/badge.svg)](https://github.com/neuralmagic/optimum-deepsparse/actions/workflows/test_nightly.yaml)

## Install
Optimum DeepSparse is a fast-moving project, and you may want to install from source.

`pip install git+https://github.com/neuralmagic/optimum-deepsparse.git`

### Installing in developer mode

If you are working on the `optimum-deepsparse` code then you should use an editable install by cloning and installing `optimum` and `optimum-deepsparse`:

```
git clone https://github.com/huggingface/optimum
git clone https://github.com/neuralmagic/optimum-deepsparse
pip install -e optimum -e optimum-deepsparse
```

Now whenever you change the code, you'll be able to run with those changes instantly.


## How to use it?
To load a model and run inference with DeepSparse, you can just replace your `AutoModelForXxx` class with the corresponding `DeepSparseModelForXxx` class. 

```diff
import requests
from PIL import Image

- from transformers import AutoModelForImageClassification
+ from optimum.deepsparse import DeepSparseModelForImageClassification
from transformers import AutoFeatureExtractor, pipeline

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model_id = "microsoft/resnet-50"
- model = AutoModelForImageClassification.from_pretrained(model_id)
+ model = DeepSparseModelForImageClassification.from_pretrained(model_id, export=True, input_shapes="[1,3,224,224]")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
cls_pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
outputs = cls_pipe(image)
```

| Supported Task                              | Model Class |
| ------------------------------------------- | ------------- |
| "image-classification"                      | DeepSparseModelForImageClassification  |
| "text-classification"/"sentiment-analysis"  | DeepSparseModelForSequenceClassification  |
| "audio-classification"                      | DeepSparseModelForAudioClassification  |
| "question-answering"                        | DeepSparseModelForQuestionAnswering  |
| "image-segmentation"                        | DeepSparseModelForSemanticSegmentation  |

If you find any issue while using those, please open an issue or a pull request.
