<img src="../../citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# en_2021-11-15 #

This English language model was trained using the **train** and **dev** partitions of PARC 3.0. The evaluation results below where obtained using the **test** partition. 

## Licence ##

Licensed under the Creative Commons [Attribution-NonCommercial-ShareAlike 4.0 International licence](./CC_BY-NC-SA_4.0.txt) and the [VerbNet 3.0 license](../../verbnet-license.3.0.txt).

## Evaluation Results ##

### citron_evaluate.py ###

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

### cue_classifier_builder.py ###

| Metric          | Score |
|-----------------|-------|
| Precision       | 95.5% |
| Recall          | 72.9% |
| F1:             | 82.7% |

### source_classifier_builder.py ###

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

### source_resolver_builder.py ###

| Metric         |  Score |
|----------------|--------|
| Precision      | 100.0% |
| Recall         |  99.6% |
| F1:            |  99.8% |

### content_classifier_builder.py ###

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

### content_resolver_builder.py ###

| Metric          | Score |
|-----------------|-------|
| Precision       | 99.5% |
| Recall          | 93.6% |
| F1:             | 96.5% |

### coreference_resolver_builder.py ###

| Metric          | Score |
|-----------------|-------|
| Precision       | 86.4% |
| Recall          | 97.4% |
| F1:             | 91.6% |

Copyright 2021 British Broadcasting Corporation.