# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
Application which iterates through a corpus of PARC v3.0 and PDTB v2.0 datafiles and
converts the data to Citron's internal JSON format.

Optionally, it can also add coreference data from the CoNLL-2011 Shared Task dataset, 
after it has been processed using the conll-extractor application. Note that the
CoNLL-2011 dataset does not provide data for all documents in PARC 3.0.
"""

import xml.etree.ElementTree as ET
import argparse
import logging
import json
import os

from citron.logger import logger

PARTITIONS = ["train", "dev", "test"]
PDTB_PREFIX_LENGTH = 9  # The length of the prefix found in all PDTB files.

SKIP_IDS = {
    "wsj_0118_Attribution_relation_level.xml_set_11",
    "wsj_0814_Attribution_relation_level.xml_set_35",
    "wsj_2454_Attribution_relation_level.xml_set_8"
}

OMIT_NESTED_ATTRIBUTIONS = True


def main():
    """
    Extract PARC 3.0 attributions and convert to Citron annotation format.
    """
    
    parser = argparse.ArgumentParser(
        description="PARC data extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v",
      action = "store_true",
      default = False,
      help = "Verbose mode"
    )
    parser.add_argument("--parc-path",
      metavar = "parc_path",
      type = str,
      required=True,
      help = "Path to: PARC3_complete directory"
    )
    parser.add_argument("--pdtb-path", 
      metavar = "pdtb_path",
      type = str,
      required=True,
      help = "Path to: pdtb_v2/data/raw/wsj/"
    )
    parser.add_argument("--conll-path", 
      metavar = "conll_path",
      type = str,
      help = "Optional: Path to output directory of conll_extractor.py"
    )
    parser.add_argument("--output-path", 
      metavar = "output_path",
      type = str,
      required=True,
      help = "Path of output directory"
    )
    args = parser.parse_args()
    
    if args.v:
        logger.setLevel(logging.DEBUG)
    
    logger.info("PARC path:   %s", args.parc_path)
    logger.info("PDTB path:   %s", args.pdtb_path)
    logger.info("CoNLL path:  %s", args.conll_path)
    logger.info("Output path: %s", args.output_path)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    data_source = ParcDataSource(args.parc_path, args.pdtb_path, args.conll_path)
    attribution_count = 0
    
    for filepath, text, quotes, coreference_groups in data_source:        
        relative_path = filepath[len(args.parc_path): -4] + ".json"
        
        if relative_path[0] == "/":
            relative_path = relative_path[1:]
            
        outpath = os.path.join(args.output_path, relative_path)
        end = outpath.rfind("/")
        parent_directory = outpath[:end]
        
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
        
        data = {}
        data["text"] = text
        data["quotes"] = quotes
        
        if coreference_groups is not None:
            data["coreference_groups"] = coreference_groups
        
        attribution_count += len(quotes)
        
        with open(outpath, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(data, indent=4, sort_keys=False, ensure_ascii=False) + "\n")
    
    logger.info("Attribution count: %s", attribution_count)


class ParcDataSource(object):
    """
    Class which provides an iterable source of data from PARC v3.0.
    """
    
    def __init__(self, parc_path, pdtb_path, conll_path=None):
        """
        Constructor.
        
        Args:
            parc_path: Path to PARC 3.0.
            pdtb_path: Path to PDTB v2.0 data (pdtb_v2/data/raw/wsj/)
            conll_path: Path to CoNLL-2011 or None.
        """
    
        self.parc_parser = ParcParser()        
        self.parc_files  = []
        self.pdtb_files  = []
        self.conll_files = []
        self.index = 0
        self.conll_file_count = 0
        
        for partition in PARTITIONS:
            parc_partition = os.path.join(parc_path, partition)
            self._add_directory(parc_partition, pdtb_path, conll_path)
            
        logger.info("PARC file count:  %s", len(self.parc_files))
        logger.info("CoNLL file count: %s", self.conll_file_count)
    
    
    def __iter__(self):
        """
        Iterate through the data files in the datasets.
        
        Yields:
            A tuple containing:
                the path of the PARC 3.0 file.
                the document text.
                a JSON serialisable list of attributions.
                a list of coreference_groups, or None.
        """
        
        while self.index < len(self.parc_files):
            parc_file = self.parc_files[self.index]
            logger.debug("parc_file: %s", parc_file)
            
            pdtb_file = self.pdtb_files[self.index]
            conll_file = self.conll_files[self.index]
            text, attributions = self.parc_parser.get_attributions(parc_file, pdtb_file)
            
            if conll_file is not None:
                coreference_parser = CoreferenceParser(conll_file)
                coreference_groups = coreference_parser.get_coreference_groups()
                
                for attribution in attributions:
                    coreference_parser.add_coreferences(attribution)
            
            else:
                coreference_groups = None
            
            self.index += 1
            yield parc_file, text, attributions, coreference_groups
    
    
    def _add_directory(self, parc_path, pdtb_path, conll_path):
        """
        Iterate through the directories in each dataset.
        
        Args:
            parc_path: The path of the PARC 3.0 file.
            pdtb_path: The path to the corresponding PDTB v2.0 file.
            conll_path: The path to the corresponding CoNLL-2011 file.
        """
        
        for filename in os.listdir(parc_path):
            parc_sub_path = os.path.join(parc_path, filename)
            
            if os.path.isfile(parc_sub_path):
                if filename.endswith(".xml"):
                    pdtb_sub_path = os.path.join(pdtb_path, filename[:-4])
                    
                    if conll_path is not None:
                        conll_sub_path = os.path.join(conll_path, filename[:-4] + ".json")
                    
                    else:
                        conll_sub_path = None
                    
                    self._add_file(parc_sub_path, pdtb_sub_path, conll_sub_path)       
            
            elif os.path.isdir(parc_sub_path):
                pdtb_sub_path  = os.path.join(pdtb_path, filename)
                
                if conll_path is not None:
                    conll_sub_path = os.path.join(conll_path, filename)
                else:
                    conll_sub_path = None
                           
                self._add_directory(parc_sub_path, pdtb_sub_path, conll_sub_path)
    
    
    def _add_file(self, parc_path, pdtb_path, conll_path):
        """
        Add file paths from each dataset.
        
        Args:
            parc_path: The path of the PARC 3.0 file.
            pdtb_path: The path to the corresponding PDTB v2.0 file.
            conll_path: The path to the corresponding CoNLL-2011 file.
        """
        
        if os.path.isfile(pdtb_path):
            self.parc_files.append(parc_path)
            self.pdtb_files.append(pdtb_path)
            
            if conll_path is not None and os.path.isfile(conll_path):
                self.conll_file_count += 1
                self.conll_files.append(conll_path)
            else:
                self.conll_files.append(None)
    

class ParcParser(object):
    """
    Class which parses PARC and PDTB files and extracts attributions.
    The ByteCount fields are used to denote the text spans. As PDTB uses ASCII 
    encoding this corresponds to the character offsets.
    """
    
    def get_attributions(self, parc_path, pdtb_path):
        """
        Get PARC attributions using spacy spans and tokens, omitting 
        those where the cue is in quotes
        
        Each attribution is three element array.
          - the first element is an array of source spans
          - the second element is the cue span              
          - the third element is an array of content spans
        
        Args:
            parc_path: The path to a PARC 3.0 file.
            pdtb_path: The path to the corresponding PDTB v2.0 file.
        
        Returns:
            A tuple containing:
                the document text (string).
                the a list of attributions.
        """
        
        text = self.read_pdtb_file(pdtb_path)
        attributions = self.get_parc_attributions(parc_path, text)
        attributions = self.filter_attributions(text, attributions)
        self.correct_inconsistent_spans(text, attributions)
        return text, attributions
    
    
    def filter_attributions(self, text, attributions):
        """
        Remove attributions where cue is in quotation marks or where a
        span has no alphanumeric content.
        
        Args:
            text: The document text (string).
            attributions: A list of attributions.
        
        Returns:
            A filtered list of attributions.
        """
        
        filtered_attributions = []
        inside_quotation_marks_labels = self.get_inside_quotation_marks_labels(text)
        
        for attribution in attributions:
            if not self.is_valid_attribution(attribution):
                continue
            
            # Ignore attributions where the cue is inside quotation marks
            start = attribution["cue"]["start"]
            end   = attribution["cue"]["end"]
            
            if inside_quotation_marks_labels[start] == 1 or inside_quotation_marks_labels[end] == 1:
                continue
            
            filtered_attributions.append(attribution)
        
        return filtered_attributions
    
    
    def is_valid_attribution(self, attribution):
        """
        Check the validity of an attribution.
        
        Args:
           attribution: The attribution.
        
        Returns:
           A boolean value.
        """
    
        cue = attribution["cue"]
        
        if self.has_no_alnum(cue):
            return False
        
        for source in attribution["sources"]:
            if self.has_no_alnum(source):
                return False
        
        for content in attribution["contents"]:
            if self.has_no_alnum(content):
                return False
        
        return True
    
    
    def correct_inconsistent_spans(self, text, attributions):
        """
        Correct minor inconsistencies found in PARC 3.0 so that the attribution 
        data is as consistent as possible.
        - some character indices are offset by one character, truncating words.
        - some spans include punctuation and prefixes which others omit.
        - the inclusion of quotation marks at the beginning and end of spans can be inconsistent.
        - some sources are followed by a comma and some descriptive text.
        
        Args:
            text: The document text.
            attributions: A list of attributions.
        """
        
        for attribution in attributions:
            cue = attribution["cue"]
            self.restore_truncated_words(text, cue)
            self.trim_preceding_punctuation(text, cue)
            self.strip(text, cue)
            self.remove_unwanted_prefix(text, cue)
            self.trim_trailing_punctuation(text, cue)
            
            for source in attribution["sources"]:
                self.restore_truncated_words(text, source)
                self.trim_bracketed_suffix(text, source)
                self.trim_preceding_punctuation(text, source)
                self.trim_from_comma_onwards(text, source)
                self.strip(text, source)
                self.trim_trailing_punctuation(text, source)
            
            for content in attribution["contents"]:
                self.restore_truncated_words(text, content)
                self.correct_inconsistent_quotation_marks(text, content)
                self.trim_preceding_punctuation(text, content)
                self.strip(text, content)
                self.trim_trailing_punctuation(text, content)
    
    
    def get_parc_attributions(self, parc_path, pdtb_text):
        """
        Returns a list of attributions, using character indices.        
        Each attribution is an list containing three elements:
        
        The first element contains a list of lists containing two integer fields:
          - source span start index
          - source span end index
        
        The second element contains a list containing two integer fields:
          - cue span start index
          - cue span end index
          
        The third element is a list of lists containing two integer fields:
          - content span start index
          - content span end index
        
        where each value is the PARC document ByteCount. Because PDTB is
        ASCII encoded this corresponds to the character index.
        
        Args:
            parc_path: The path to a PARC 3.0 file.
            pdtb_text: The text extracted from the corresponding PDTB v2.0 file, minus the 9 character prefix,
        
        Returns:
           A list of attributions.
        """
        
        attribution_map = self.parse(parc_path)

        # Check each attribution has a cue and at least one source and content span
        attributions = []
        
        for id_, attribution_tuple in attribution_map.items():
            source_tuples  = attribution_tuple[0]
            cue_tuple      = attribution_tuple[1]
            content_tuples = attribution_tuple[2]

            # Validate attribution
            if len(source_tuples) == 0:
                logger.debug("PARC %s has no sources", id_)
                continue

            if cue_tuple[0] is None or cue_tuple[1] is None:
                logger.debug("PARC %s has no cue", id_)
                continue
            
            if len(content_tuples) == 0:
                logger.debug("PARC %s has no contents", id_)
                continue
            
            # Create attribution object
            sources = []
            
            for source_tuple in source_tuples:
                source = self.get_span_json(source_tuple, pdtb_text)
                sources.append(source)
            
            cue = self.get_span_json(cue_tuple, pdtb_text)             
            contents = []
            
            for content_tuple in content_tuples:
                content = self.get_span_json(content_tuple, pdtb_text)
                contents.append(content)
            
            attribution = {}
            attribution["id"] = id_
            attribution["cue"] = cue
            attribution["sources"] = sources
            attribution["contents"] = contents
            attributions.append(attribution)
        
        return attributions
    
    
    def parse(self, parc_path):
        """
        Returns a map between attibution IDs and attributions.    
        Each attribution is an list containing three elements:
        
        The first element contains a list of lists, where each list contains two integer fields:
          - source span start index
          - source span end index
        
        The second element contains a list containing two integer fields:
          - cue span start index
          - cue span end index
          
        The third element is a list of lists where each list contains two integer fields:
          - content span start index
          - content span end index
        
        where each index is the PARC document ByteCount. PDTB is ASCII
        encoded so this corresponds to the character index.
        
        Args:
            parc_path: The path to a PARC 3.0 file.
        
        Returns:
            A dict object mapping attribution IDs to attributions.
        """
        
        tree = ET.parse(parc_path)
        attribution_map = {}
        self.in_source = False
        self.in_content = False
        
        # Process tree
        for sentence in tree.getroot():
            self.process_element(sentence, attribution_map)
            
        return attribution_map
    
            
    def process_element(self, element, attribution_map):
        """
        Process an xml.etree.ElementTree.Element
        
        Args:
            element: An Elemeny object.
            attribution_map: A dict object mapping attribution IDs to attributions.
        """
               
        if element.tag == "WORD":
            byte_count = element.attrib["ByteCount"]
            comma_index = byte_count.find(",")
            start = int(byte_count[0 : comma_index].strip())
            end   = int(byte_count[comma_index + 1 :].strip())
            role_value = None
            
            for child in element:
                if child.tag == "attribution":
                    attribution_id = child.attrib["id"]
                    
                    if attribution_id in SKIP_IDS:
                        continue
                    
                    if OMIT_NESTED_ATTRIBUTIONS and "_Nested_" in attribution_id:
                        continue
                    
                    if attribution_id in attribution_map:
                        attribution = attribution_map[attribution_id]
                    
                    else:
                        attribution = [[], [None] * 2, []]
                        attribution_map[attribution_id] = attribution
                        self.in_source = False
                        self.in_content = False
                    
                    for grandchild in child:
                        if grandchild.tag == "attributionRole":
                            role_value = grandchild.attrib["roleValue"]
                            
                            if role_value is not None:
                                if role_value == "source":
                                    sources = attribution[0]
                                    
                                    if not self.in_source or len(sources) == 0:
                                        source = [None] * 2
                                        sources.append(source)
                                        self.in_source = True
                                    else:
                                        source = sources[-1]
                                                    
                                    if source[0] is None:
                                        source[0] = start
                                        
                                    source[1] = end
                                
                                elif role_value == "cue":
                                    cue = attribution[1]
                                    if cue[0] is None:
                                        cue[0] = start
                                        
                                    cue[1] = end
                                    
                                elif role_value == "content":
                                    contents = attribution[2]
                                                                
                                    if not self.in_content or len(contents) == 0:
                                        content = [None] * 2
                                        contents.append(content)
                                        self.in_content = True
                                    else:
                                        content = contents[-1]
                                                    
                                    if content[0] is None:
                                        content[0] = start
                                    
                                    content[1] = end
            
            if role_value is None or role_value != "source":
                self.in_source = False
                
            if role_value is None or role_value != "content":
                self.in_content = False    
        
        else:
            for child in element:
                self.process_element(child, attribution_map)
    
    
    @staticmethod
    def read_pdtb_file(filename):
        """
        Read PDTB v2.0 file. The files have a 9 character prefix which this method removes.
        This affects the character indices loaded from PARC so they are adjusted later.
        
        Args:
           filename: The path to the PDTB v2.0 file.
        
        Returns:
           A string.
        """
        
        with open(filename, encoding="utf-8") as infile:
            text = infile.read()
            return text[PDTB_PREFIX_LENGTH :]
    
    
    @staticmethod        
    def get_span_json(span_tuple, pdtb_text):
        """
        Get a JSON serialisable representation of a text span.
        
        Args:
            span_tuple: a tuple containing start and end indices.
            pdtb_text: the text of the PDTB v2.0 file (string)
            
        Returns:
            A JSON serialisable object.
        """
        
        start = span_tuple[0] - PDTB_PREFIX_LENGTH
        end   = span_tuple[1] - PDTB_PREFIX_LENGTH
        text = pdtb_text[start : end]
        return {"start": start, "end": end, "text": text}
    
    
    @staticmethod
    def has_no_alnum(span):
        """
        Test whether the text of a span does not contain any 
        alphanumeric characters. 
        
        Args:
            span: an object representing a text span.
        
        Returns:
            A boolean value.
        """
        
        for char in span["text"]:
            if char.isalnum():
                return False
        
        return True
    
    
    @staticmethod
    def restore_truncated_words(text, span):
        """
        Some PARC character offsets appear to be offset by -1.
        This adds a space at the beginning of a span and truncates the
        character at the end.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]
        
        if not text[start].isspace():
            return
        
        if end < len(text):
            next_char = text[end]
            
            if next_char.isalnum():
                span["end"] = end + 1
                span["text"] = text[start : end + 1]
        
    
    @staticmethod
    def correct_inconsistent_quotation_marks(text, span):
        """
        Correct inconsistent inclusion of quotation marks in spans.
        The aim is to start and end the span with quotation marks if they
        are present at the boundaries of the span.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]
        
        if end < len(text):
            next_char = text[end]
            
            if next_char == "\"":
                span["end"] = end + 1
                span["text"] = text[start : end + 1]
                return
        
        if end + 1 < len(text) and text[end + 1] == "\"":
            next_char = text[end]
            if next_char == "," or next_char == "." or next_char == "!" or next_char == "," or next_char == "?":
                span["end"] = end + 2
                span["text"] = text[start : end + 2]
    
    
    @staticmethod
    def trim_bracketed_suffix(text, span):
        """
        Remove bracketed content at the end of the span
        Used for sources
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]

        # Remove bracketed content at the end of the span
        corrected_end = None
        
        for i in range(end - 1, start - 1, - 1):
            if text[i] == "(" or text[i] == "[":
                corrected_end = i
        
        if corrected_end is not None:
            span["end"] = corrected_end
            span["text"] = text[start : corrected_end]
    
    
    @staticmethod 
    def trim_from_comma_onwards(text, span):
        """
        Trim a text span from the comma onwards (if present)
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]
        comma_index = None
        
        for i in range(start , end):
            if text[i] == ",":
                comma_index = i
                break
        
        if comma_index is not None and comma_index > 4:
            span["end"] = comma_index
            span["text"] = text[start : comma_index]
    
    
    @staticmethod
    def strip(text, span):
        """
        Remove whitespace(s) from start and end of a span.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]
        
        for i in range(start, end):
            if not text[i].isspace():
                start = i
                break
        
        for i in range(end, start, - 1):
            if not text[i -1].isspace():
                end = i
                break
        
        if start != span["start"] or end != span["end"]:        
            span["start"] = start
            span["end"] = end
            span["text"] = text[start : end]
    
    
    def remove_unwanted_prefix(self, text, span):
        """
        Remove unwanted prefixes from a span.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end   = span["end"]
        span_text = text[start : end]
        
        if span_text.startswith("have "):
            start += 5
        
        elif span_text.startswith("also "):
            start += 5
        
        elif span_text.startswith("has also "):
            start += 9
        
        elif span_text.startswith("have been "):
            start += 10
        
        elif span_text.startswith("however, "):
            start += 9
            
        elif span_text.startswith("meanwhile, "):
            start += 11
        
        elif span_text.startswith("a letter, "):
            start += 10
            
        elif span_text.startswith("though, "):
            start += 8
            
        elif span_text.startswith("separately "):
            start += 11
        
        elif span_text.startswith("understand, "):
            start += 12
            
        elif span_text.startswith("perhaps understandably, "):
            start += 24
        
        # Suffixes
        if span_text.endswith("urge"):
            start = end - 4
        
        elif span_text.endswith("urged"):
            start = end - 5
        
        elif span_text.endswith("asking"):
            start = end - 6
        
        elif span_text.endswith("saying"):
            start = end - 6
                
        elif span_text.endswith("announcing"):
            start = end - 10
        
        if start != span["start"]:
            span["start"] = start
            span["text"] = text[start : end]
    

    @staticmethod
    def trim_preceding_punctuation(text, span):
        """
        Trim punctuation from the beginning of a span.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end = span["end"]
        first = text[start] 
        
        if first == "," or first == ":" or first == ";" or first == "!" or first == "?":
            span["start"] = start + 1
            span["text"] = text[start + 1 : end]
    
    
    @staticmethod
    def trim_trailing_punctuation(text, span):
        """
        Trim punctuation from the end of a span.
        
        Args:
            text: the text of the document (string).
            span: an object representing a text span.
        """
        
        start = span["start"]
        end = span["end"]
        last = text[end - 1] 
        
        if last == "," or last == ":" or last == ";" or last == "!" or last == "?":
            span["end"] = end - 1
            span["text"] = text[start : end - 1]
    
    
    @staticmethod
    def get_inside_quotation_marks_labels(text):
        """
        Get a list of binary labels indicating whether each character in a
        document is inside quotation marks.
        
        Args:
            text: the text of a document (string).
        
        Returns:
            A list containing a binary label for each character in the document.    
        """
        
        labels = [0] * len(text)
        in_quotes = False
        
        for i in range(0, len(text)):
            char = text[i]
            
            if in_quotes:
                labels[i] = 1
            
            else:
                labels[i] = 0
            
            if not in_quotes and char == "\"":
                in_quotes = True
            
            elif in_quotes and char == "\"":
                in_quotes = False
        
        return labels


class CoreferenceParser():
    """
    Class which provides methods to obtain coreferences from
    the CoNLL-2011 Shared Task dataset. Uses datafiles which 
    have been processed using the conll-extractor application.
    """
    
    def __init__(self, conll_path):
        """
        Parse a file of CoNLL-2011 data and extract the coreference groups.
        Each coreference group is a list of spans. Each span is a dict with 
        start, end and text fields.
        
        Args:
            conll_path: A path to a CoNLL-2011 file.
        """
        
        try:
            with open(conll_path, encoding="utf-8") as infile:
                coreference_groups = json.load(infile)
                sorted_groups = []
                
                for group in coreference_groups:
                    if len(group) <= 1:
                        continue
                    
                    # Correct indices
                    for member in group:
                        member["start"] = member["start"] - PDTB_PREFIX_LENGTH
                        member["end"]   = member["end"]   - PDTB_PREFIX_LENGTH
                    
                    sorted_group = sorted(group, key=lambda x: x["start"])
                    sorted_groups.append(sorted_group)
                
                self.coreference_groups =  sorted_groups
        
        except IOError as err:
            logger.error("Error loading: %s %s", conll_path, err)
    
    
    def get_coreference_groups(self):
        """
        Get the extracted coreference groups.
        
        Returns:
           A list of coreference groups.
        """
        
        return self.coreference_groups
    
    
    def add_coreferences(self, attribution):
        """
        Get the earliest coreferences for the sources of an Attribution.
        
        Args:
            span: An Attribution object.

        Returns:
            A dict (with start, end and text fields) representing a text span, or None.
        """
        
        coreferences = []
        
        for source in attribution["sources"]:
            coreference = self._get_coreference(source)
            
            if coreference is not None:
                coreferences.append(coreference)
        
        attribution["coreferences"] = coreferences
    
    
    def _get_coreference(self, span):
        """
        Get the earliest coreference for the supplied span.
        
        Args:
            span: A span dict with start, end and text fields.

        Returns:
            A span dict (with start, end and text fields) or None.
        """
        
        coreference_group = self._get_coreference_group(span)
        
        if coreference_group is not None:
            if span["text"] != coreference_group[0]["text"]:
                return coreference_group[0]
        
        return None
    
    
    def _get_coreference_group(self, span):
        """
        Get the coreference group for the supplied span.
        
        Args:
            span: A dict with start, end and text fields.

        Returns:
            A dict (with start, end and text fields) or None.
        """
        
        for coreference_group in self.coreference_groups:
            for coreference_span in coreference_group:
                if self._are_overlapping_spans(span, coreference_span):
                    return coreference_group
        
        return None
    
    
    @staticmethod
    def _are_overlapping_spans(span, other_span):
        """
        Test whether two spans overlap.
        
        Args:
            span: A span dict with start, end and text fields.
            other_span: A span dict with start, end and text fields.
        
        Returns:
            A boolean value.
        """
        
        return span["start"] < other_span["end"] and span["end"] > other_span["start"]


if __name__ == "__main__":
    main()
