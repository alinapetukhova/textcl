# TextCL

[![Build Status](https://travis-ci.com/alinapetukhova/textcl.svg?branch=master)](https://travis-ci.com/github/alinapetukhova/textcl)
[![codecov](https://codecov.io/gh/alinapetukhova/textcl/branch/master/graph/badge.svg?token=jgYuXyGGjS)](https://codecov.io/gh/alinapetukhova/textcl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

The **TextCL** package aims to clean text data for later use in Natural Language Processing tasks. It can be used as an initial step in text analysis as well as in predictive, classification or text generation models.

The quality of the models strongly depends on the quality of the input data. Common problems in the data sets include:

- If data are coming from a optical character recognition (OCR) platform, text in tables and columns is usually not processed correctly and will add noise to the models.
- Some parts of large texts scopes may contain sentences from different languages rather than the target language of the model and have to be filtered out.
- Real-world texts often have duplicated sentences due to the use of templates. In text generation tasks, this can cause model overfitting and duplications in generated texts or summaries.
- Data sets may contain text that is different from the main topic, such as a weather forecast in an accounting report.

## Features

The **TextCL** package allows the user to perform the following text pre-processing tasks:

- Split texts into sentences.
- Language filtering, for removing sentences from text not in the target language.
- Perplexity filtering, for removing linguistically unconnected sentences, that can be produced by OCR modules. For example: `Sustainability Report 2019 36 3%?!353? 1. 5В°C 1} 33%.`
- Duplicate sentences filtering using Jaccard similarity, for removing duplicate sentences from the text.
- Unsupervised outlier detection for revealing texts that are outside of the main data set topic distribution. Four methods are included with package for this purpose:
  - TONMF: Block Coordinate Descent Framework
    ([source article](https://arxiv.org/pdf/1701.01325.pdf),
    [matlab implementation](https://github.com/ramkikannan/outliernmf))
  - RPCA: Robust Principal Component Analysis
    ([source article](https://arxiv.org/pdf/0912.3599.pdf),
    [python implementation](https://github.com/dganguli/robust-pca))
  - SVD: Singular Value Decomposition
    (based on the [NumPy SVD implementation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html))

## Documentation

* [TextCL's API documentation](https://alinapetukhova.github.io/textcl/docs/)
* [Tutorial for the preprocessing functions](https://nbviewer.jupyter.org/github/alinapetukhova/textcl/blob/master/examples/text_preprocessing_example.ipynb)
* [Tutorial for the outlier detection functions](https://nbviewer.jupyter.org/github/alinapetukhova/textcl/blob/master/examples/outlier_detection_functions_plots_example.ipynb)
* [Developer's guide](https://github.com/alinapetukhova/textcl/blob/master/doc/devguide.md)

## Requirements

- Python >= 3.6
- flair >= 0.7
- langdetect >= 1.0.8
- numpy >= 1.16.5, < 1.20.0
- pandas >= 1.0.3
- lxml >= 4.6.2
- protobuf >= 3.14.0
- nltk >= 3.4.5

## How to install

### From PyPI

```text
pip install textcl
```

### From source/GitHub

```text
pip install git+https://github.com/alinapetukhova/textcl.git#egg=textcl
```

## License

[MIT License](LICENSE)