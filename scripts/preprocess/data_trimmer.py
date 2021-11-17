# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This application removes sentences from Citron format data files which have missing
annotations. The sentences are identified by using a Cue Classifier to predict the
location of quote cues and then comparing these against the annotations.

The approach was suggested in the following paper to deal with missing annotations
data in the PARC 3.0 dataset:

Sylvia Pareti, Tim O'Keefe, Ioannis Konstas, James R Curran and Irena Koprinska. 2013.
“Automatically Detecting and Attributing Indirect Quotations”. Proceedings of the Conference
on Empirical Methods in Natural Language Processing (EMNLP), Seattle, U.S.
http://www.aclweb.org/anthology/D13-1101
"""

import argparse
import logging
import json
import sys
import os

from citron.data import CitronParser
from citron.cue import CueClassifier
from citron.logger import logger
from citron import utils

omitted_sentences = 0


def main():
    global omitted_sentences
    
    parser = argparse.ArgumentParser(
        description="Data trimmer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v",
      action = "store_true",
      default = False,
      help = "Verbose mode"
    )
    parser.add_argument("--input-path",
      metavar = "input_path",
      type = str,
      required=True,
      help = "Path to directory of Citron format files"
    )
    parser.add_argument("--output-path", 
      metavar = "output_path",
      type = str,
      required=True,
      help = "Path of output directory"
    )
    parser.add_argument("--model-path",
      metavar = "model_path",
      type = str,
      required=True,
      help = "Path to Citron model directory"
    )
    args = parser.parse_args()
    
    if args.v:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Input path:  %s", args.input_path)
    logger.info("Output path: %s", args.output_path)
    
    nlp = utils.get_parser()
    parser = CitronParser(nlp)
    cue_classifier = CueClassifier(args.model_path)
    
    input_file_paths = utils.get_files(args.input_path)
    logger.info("Found %s data files", len(input_file_paths))
    omitted_sentences = 0
    
    for input_file_path in input_file_paths:
        logger.debug("Input file: %s", input_file_path)
        subpath = input_file_path[len(args.input_path):]
        
        if len(subpath) == 0:
            output_file_path = args.output_path
        
        else:
            if subpath[0] == "/":
                subpath = subpath[1:]
            
            output_file_path = os.path.join(args.output_path, subpath)
            
        end = output_file_path.rfind("/")
        parent_directory = output_file_path[:end]
        
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
        
        data = parser.parse(input_file_path)
        old_to_new_index = get_old_to_new_index(data, cue_classifier)
        revise_datafile(input_file_path, output_file_path, old_to_new_index)
    
    logger.info("Omitted_sentences: %s", omitted_sentences)


def get_old_to_new_index(data, cue_classifier):
    """
    Get an index mapping original character indices to new indices.
    
    Args:
        data: data extracted from a Citron format data file.
        cue_classifier: A Cue Classifier object.
    
    Returns:
       A list of integers.
    """
    
    global omitted_sentences
    doc = data[0]
    quotes = data[1]
    
    actual_cue_labels = utils.get_cue_iob_labels(doc, quotes)
    actual_source_labels = utils.get_source_iob_labels(doc, quotes)
    actual_content_labels = utils.get_content_iob_labels(doc, quotes)
    
    predicted_cue_labels = cue_classifier.predict_cues_and_labels(doc)[1]
    old_to_new_index = [None] * len(doc.text)
    i = 0
    
    for sentence in doc.sents:
        has_actual_cue     = utils.span_has_labels(sentence, actual_cue_labels)
        has_actual_source  = utils.span_has_labels(sentence, actual_source_labels)
        has_actual_content = utils.span_has_labels(sentence, actual_content_labels)    
        has_predicted_cue  = utils.span_has_labels(sentence, predicted_cue_labels)
        
        has_annotations = has_actual_cue or has_actual_source or has_actual_content
        
        # Skip sentences with missing annotations
        if not has_annotations and has_predicted_cue:
            omitted_sentences += 1
            continue
        
        for j in range(0, len(sentence.text_with_ws)):
            old = sentence.start_char + j
            old_to_new_index[old] = i
            i += 1
    
    return old_to_new_index


def revise_datafile(input_file_path, output_file_path, old_to_new_index):
    """
    Revise a data file to remove sentences and correct the character indices 
    in the annotations.
    
    Args:
        input_file_path: The path of the input file.
        output_file_path: The path of the output file.
        old_to_new_index: A list of integers mapping character indices to new indices.
    """
    
    with open(input_file_path, encoding="utf-8") as infile:
        data = json.load(infile)
    
    corrected_data = {}
    text = ""
    
    for old, new in enumerate(old_to_new_index):
        if new is not None:
            text += data["text"][old]
    
    # Ignore files that are now empty.
    if len(text.strip()) == 0:
        return
    
    corrected_data["text"] = text
    corrected_quotes = []
    
    # Correct quote indices
    for quote in data["quotes"]:
        corrected_quote = {}
        corrected_quote["id"] = quote["id"]
        corrected_quote["cue"] = correct_span(quote["cue"], text, old_to_new_index)
        sources = []
        
        for source in quote["sources"]:
            sources.append(correct_span(source, text, old_to_new_index))
        
        corrected_quote["sources"] = sources
        contents = []
        
        for content in quote["contents"]:
            contents.append(correct_span(content, text, old_to_new_index)) 
        
        corrected_quote["contents"] = contents
        
        if "coreferences" in quote:
            corrected_quote["coreferences"] = correct_coreferences(quote["coreferences"], text, old_to_new_index)
        
        corrected_quotes.append(corrected_quote)
        
    corrected_data["quotes"] = corrected_quotes  
    
    # Correct coreference groups
    if "coreference_groups" in data:
        corrected_groups = []
        
        for coref_group in data["coreference_groups"]:
                corrected_group = correct_coreferences(coref_group, text, old_to_new_index)
                
                if len(corrected_group) > 1:
                    corrected_groups.append(corrected_group)
        
        corrected_data["coreference_groups"] = corrected_groups 
    
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(corrected_data, indent=4, sort_keys=False, ensure_ascii=False) + "\n")
        
    
def correct_span(span, text, old_to_new_index):
    """
    Correct the character indices in a span using the supplied index.
    
    Args:
        span: A dict representing a text span.
        text: The revised document text (string)
        old_to_new_index: A list of integers mapping character indices to new indices.
    
    Returns:
        A dict representing a text span.
    """
    
    start = old_to_new_index[span["start"]]
    end   = old_to_new_index[span["end"]]
    span_text = text[start : end]
    
    # Check new character indices refer to the same text.
    if span_text != span["text"]:
        logger.error("Text mismatch: old: %s new: %s", span["text"], text[start : end])
        logger.error("Old start: %s end: %s", span["start"], span["end"])
        logger.error("New start: %s end: %s", start, end)
        sys.exit(1)
    
    return {"start": start, "end": end, "text": span_text}


def correct_coreferences(coreferences, text, old_to_new_index):
    """
    Revise a list of coreferences using the supplied index.
    
    Args:
        coreferences: a list of dicts representing text spans.
        text: The revised document text (string)
        old_to_new_index: A list of integers mapping character indices to new indices.
    
    Returns:
        A list of dicts representing text spans.
    """
    
    corrected_coreferences = []
    
    for coreference in coreferences:
        start = old_to_new_index[coreference["start"]]
        end   = old_to_new_index[coreference["end"]] 
        span_text = text[start : end]
        
        # Ignore coreferences which are in omitted text 
        if start is None or end is None:
            continue
        
        # Check the new character indices refer to the same text   
        if span_text != coreference["text"]:
            print("Error mismatch!")
            print("Old text:", coreference["text"])
            print("New text:", text[start : end])
            print("Old start:", coreference["start"], " end:", coreference["end"])
            print("New start:", start, " end:", end)
            sys.exit()
        
        corrected_coreferences.append({"start": start, "end": end, "text": span_text})
    
    return corrected_coreferences


if __name__ == '__main__':
    main()
