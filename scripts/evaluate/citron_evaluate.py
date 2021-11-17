# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This application evaluates the Citron quote extraction system.
"""

import argparse
import logging

from citron.citron import Citron
from citron import utils
from citron.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Citron',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-v',
      action = 'store_true',
      default = False,
      help = 'Verbose mode'
    )
    parser.add_argument('--model-path', 
      metavar = 'model_path',
      type = str,
      required=True,
      help = 'Path to the Citron model directory'
    )
    parser.add_argument("--test-path",
      metavar = "test_path",
      type = str,
      required=True,
      help = "Path to file or directory containing Citron annotation format test data (default: no testing)"
    )
    args = parser.parse_args()
    
    if args.v:
        logger.setLevel(logging.DEBUG)

    logger.info("Evaluating Citron using: %s", args.test_path)    
    nlp = utils.get_parser()
    citron = Citron(args.model_path, nlp)    
    citron.evaluate(args.test_path)


if __name__ == '__main__':
    main()
