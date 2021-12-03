<img src="./citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# Citron #

Citron is an experimental quote extraction and attribution system created by [BBC R&D](https://www.bbc.co.uk/rd), based on a [paper](https://aclanthology.org/D13-1101/) and a [dataset](https://aclanthology.org/L16-1619/) developed by the School of Informatics at the University of Edinburgh.

It can be used to extract quotes from text documents, attributing them to the appropriate speaker and resolving pronouns where necessary. It supports direct and indirect quotes (with and without quotation marks respectively) and mixed quotes (which have direct and indirect parts). Note that there can be a significant number of errors and omissions. Extracted quotes should be checked against the input text.

You can run Citron using the [pre-trained model](./models/en_2021-11-15) or [train your own model](./scripts/train). You can also [evaluate its performance](./scripts/evaluate).

Training and evaluating models requires data using [Citron's Annotation Format](./docs/data_format.md). Citron provides [pre-processing scripts](./scripts/preprocess) to extract suitable data from the [PARC 3.0 Corpus of Attribution Relations](https://aclanthology.org/L16-1619/). Alternatively, you can create your own data using the [Citron Annotator](./scripts/annotator) app.

Technical details and potential applications are discussed in: ["Quote Extraction and Analysis for News"](./docs/DSJM_2018_paper_1.pdf).

## Installation ##
Requires Python 3.7.2 or above. The package versions shown should be installed when using the [pre-trained model](./models/en_2021-11-15).

- [Install scikit-learn (1.0.*)](https://scikit-learn.org/stable/install.html)
- [Install spaCy (3.*) and download a model](https://spacy.io/usage) &nbsp;&nbsp; (e.g. "en_core_web_sm")
- Download the source code: ```git clone git@github.com:bbc/citron.git```

Then from the citron root directory:

    python3 -m pip install -r requirements.txt

Then from python3:

    import nltk
    nltk.download("names")

## Usage  ##

Scripts to run Citron are available in the [bin/](./bin/) directory.

All scripts require the citron root directory in the PYTHONPATH.

    $ export PYTHONPATH=$PYTHONPATH:/path/to/citron_root_directory

### Run the Citron REST API and demonstration server ###
    
    $ citron-server
        --model-path   Path to Citron model directory
        --logfile      Path to logfile                   (Optional)
        --port         Port for the Citron API           (Optional: default is 8080)
        -v             Verbose mode                      (Optional)

### Run Citron on the command-line ###

    $ citron-extract
        --model-path     Path to Citron model directory
        --input-file     Path to input file                (Optional: Otherwise read from stdin)
        --output-file    Path to output file               (Optional: Otherwise write to stdout)
        -v               Verbose mode                      (Optional)

### Use Citron in Python ###

    from citron.citron import Citron
    from citron import utils
    
    nlp = utils.get_parser()
    citron = Citron(model_path, nlp)
    doc = nlp(text)
    quotes = citron.get_quotes(doc)

## Issues and Questions ##
Issues can be reported on the [issue tracker](https://github.com/bbc/citron/issues) and questions can be raised on the [discussion board](https://github.com/bbc/citron/discussions/categories/q-a).

## Contributing ##

Contributions would be welcome. Please refer to the [contributing guidelines](./CONTRIBUTING.md).

## License ##

Licensed under the [Apache License, Version 2.0](./LICENSE).

The [pre-trained model](./models/en_2021-11-15) is separately licensed under the Creative Commons [Attribution-NonCommercial-ShareAlike 4.0 International licence](./CC_BY-NC-SA_4.0.txt) and the [VerbNet 3.0 license](./verbnet-license.3.0.txt).

## Contact ##

For more information please contact: [chris.newell@bbc.co.uk](mailto:chris.newell@bbc.co.uk)

Copyright 2021 British Broadcasting Corporation.
