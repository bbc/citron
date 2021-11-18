<img src="../../citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# Training Citron Models #

Each component in Citron has a script which can be used to train and/or evaluate its associated model. 

The scripts require data in [Citron's Annotation Format](../../docs/data_format.md).  Citron provides [pre-processing scripts](../preprocess) to extract suitable data from the [PARC 3.0 Corpus of Attribution Relations](https://aclanthology.org/L16-1619/). Alternatively, you can create your own data using the [Citron Annotator](../annotator) app.

- to train a model specify the *--train-path* parameter
- to evaluate a model specify the *--test-path* parameter

The *exact match metrics* require the predicted and correct spans to be an exact match. The *overlap metrics* are based on the number of predicted tokens that overlap with correct tokens, compared to the total number of predicted and correct tokens.

## Usage ##

All scripts require the Citron project directory in the PYTHONPATH

    $ export PYTHONPATH=$PYTHONPATH:/path/to/citron

All scripts share the following parameters:

        -h, --help    (Optional: show help message and exit)
        -v            (Optional: verbose mode)

### Cue Classifier ###

Training a Cue Classifier requires a copy of VerbNet 3.3.

    $ python3 cue_classifier_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)
        --verbnet-path    Path to VerbNet 3.3        (Optional: required to train)

### Source Classifier ###

    $ python3 source_classifier_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)

### Source Resolver ###

    $ python3 source_resolver_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)

### Content Classifier ###

    $ python3 content_classifier_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)

### Content Resolver ###

    $ python3 content_resolver_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)

### Coreference Resolver ###

    $ python3 coreference_resolver_builder.py
        --model-path      Path to model directory
        --train-path      Path to training data      (Optional: required to train)
        --test-path       Path to test data          (Optional: required to evaluate)

Copyright 2021 British Broadcasting Corporation.
