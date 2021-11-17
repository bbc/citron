# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This application trains and/or evaluates a Citron Source Classifier.
"""

import argparse
import logging

from citron.source import SourceClassifier
from citron.logger import logger
from citron import utils


def main():
    parser = argparse.ArgumentParser(
        description="Build and/or evaluate a Source Classifier model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v",
      action = "store_true",
      default = False,
      help = "Verbose mode"
    )
    parser.add_argument("--train-path", 
      metavar = "train_path",
      type = str,
      help = "Optional: Path to file or directory containing Citron format training data (default: no training)"
    )
    parser.add_argument("--test-path",
      metavar = "test_path",
      type = str, 
      help = "Optional: Path to file or directory containing Citron format test data (default: no testing)"
    )
    parser.add_argument("--model-path", 
      metavar = "model_path",
      type = str,
      required=True, 
      help = "Path to the Citron model directory"
    )
    args = parser.parse_args()
    
    if args.v:
        logger.setLevel(logging.DEBUG)
    
    nlp = utils.get_parser()
    
    if args.train_path:
        SourceClassifier.build_model(nlp, args.train_path, args.model_path)
    
    if args.test_path:
        source_classifier = SourceClassifier(args.model_path)
        source_classifier.evaluate(nlp, args.test_path)
        
    if not (args.train_path or args.test_path):
        logger.error("Must specify train_path and/or test_path")


if __name__ == "__main__":
    main()
