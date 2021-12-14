<img src="../../citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# en_2021-11-15 #

##  Model Details ##

This is an English language model for the [Citron Quote Extraction and Attribution System](https://github.com/bbc/citron) produced by [BBC R&D](https://www.bbc.co.uk/rd). 

| Parameter     |                  |     
|---------------|------------------|
| Language      | English          |
| Creation date | 15 November 2021 |

For technical details see: ["Quote Extraction and Analysis for News"](https://github.com/bbc/citron/docs/DSJM_2018_paper_1.pdf). For more information please contact: [chris.newell@bbc.co.uk](mailto:chris.newell@bbc.co.uk).

## Licence ##

Licensed under the Creative Commons [Attribution-NonCommercial-ShareAlike 4.0 International licence](./CC_BY-NC-SA_4.0.txt) and the [VerbNet 3.0 license](./verbnet-license.3.0.txt).

## Intended Use ##

This is an experimental model intended for research and evaluation.

## Factors ##

The training data is based on text extracts from the Wall Street Journal, an American business-focused, English-language international daily newspaper based in New York City. It may not perform as well with text from other domains.

## Metrics ##

The performance of the model is reported in the Quantitative Analysis below using [Exact Match and Overlap Match metrics](https://github.com/bbc/citron/tree/main/scripts/evaluate) for text spans.

## Training and Evaluation Data ##

The model was trained using the **train** and **dev** partitions of the [PARC 3.0 Corpus of Attribution Relations](https://aclanthology.org/L16-1619/). It also includes data extracted from [VerbNet 3.3](https://verbs.colorado.edu/verbnet).

The Quantitative Analysis shown below was obtained using the **test** partition of the [PARC 3.0 Corpus of Attribution Relations](https://aclanthology.org/L16-1619/).

The Coreference Resolver was trained and evaluated using equivalent partitions of the [CoNLL-2011 Shared Task dataset](https://conll.cemantix.org/2011/data.html) which covers a subset of the data in PARC 3.0.

PARC 3.0 and CoNLL-2011 are extensions of the [Penn Discourse Treebank](https://catalog.ldc.upenn.edu/LDC2008T05) and [Ontonotes](https://catalog.ldc.upenn.edu/LDC2013T19) corpora. These corpora are available under license from the [Linguistic Data Consortium](https://www.ldc.upenn.edu/).

## Ethical Considerations ##

Citron was developed under the [BBC's Machine Learning Engineering Principles](https://www.bbc.co.uk/rd/publications/responsible-ai-at-the-bbc-our-machine-learning-engine-principles) which comprises of six guiding principles and a self-audit checklist.

## Caveats and Recommendations ##

The performance of this model is reasonably good but there is a significant error rate. Extracted quotes should always be checked against the original text to confirm the accuracy of the text spans and correctness of the attribution.

## Quantitative Analysis ##

### Overall Performance ###

The overall performance of Citron using this model was measured using the [Citron Evaluate](https://github.com/bbc/citron/tree/main/scripts/evaluate) script. 

#### Cue Span ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 93.3% |
| Recall          | 73.6% |
| F1              | 82.3% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 96.5% |
| Recall          | 63.9% |
| F1              | 76.8% |

#### Source Spans ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 92.9% |
| Recall          | 73.3% |
| F1              | 82.0% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 97.4% |
| Recall          | 75.8% |
| F1              | 85.3% |

#### Content Spans ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 67.3% |
| Recall          | 53.1% |
| F1              | 59.3% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 93.2% |
| Recall          | 75.3% |
| F1              | 83.3% |

#### All Quote Spans ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 64.3% |
| Recall          | 50.7% |
| F1              | 56.7% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 74.7% |
| Recall          | 93.8% |
| F1              | 83.2% |

### Performance of the Individual Components ###

The performance of the individual components of Citron using this model was measured using the [build scripts](https://github.com/bbc/citron/tree/main/scripts/train).

#### Cue Classifier ####

| Metric          | Score |
|-----------------|-------|
| Precision       | 95.5% |
| Recall          | 72.9% |
| F1:             | 82.7% |

#### Source Classifier ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 92.5% |
| Recall          | 89.2% |
| F1:             | 90.8% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 89.3% |
| Recall          | 93.1% |
| F1:             | 91.1% |

#### Source Resolver ####

| Metric         |  Score |
|----------------|--------|
| Precision      | 100.0% |
| Recall         |  99.6% |
| F1:            |  99.8% |

#### Content Classifier ####

| Exact Metric    | Score |
|-----------------|-------|
| Precision       | 78.0% |
| Recall          | 72.0% |
| F1:             | 74.9% |

| Overlap Metric  | Score |
|-----------------|-------|
| Precision       | 89.9% |
| Recall          | 90.7% |
| F1:             | 90.3% |

#### Content Resolver ####

| Metric          | Score |
|-----------------|-------|
| Precision       | 99.5% |
| Recall          | 93.6% |
| F1:             | 96.5% |

#### Coreference Resolver ####

| Metric          | Score |
|-----------------|-------|
| Precision       | 86.4% |
| Recall          | 97.4% |
| F1:             | 91.6% |

## References ##

This document is adapted from ["Model Cards for Model Reporting", M. Mitchell et Al, 2018](https://arxiv.org/abs/1810.03993)

Copyright 2021 British Broadcasting Corporation.
