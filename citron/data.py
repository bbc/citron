# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
Module providing functions to read and encapsulate data from files using
Citron's JSON data format.
"""

import json
import sys

from . import utils
from .logger import logger


class DataSource():
    """
    Class providing an iterable source of data for training or test purposes.
    """
    
    def __init__(self, nlp, input_path):
        """
        Constructor.
        
        Args:
            nlp: A spaCy Language object.
            input_path: The path (string) to a file or directory of Citron format JSON data files.
            
            Directories will be explored recursively.
        """
        
        self.parser = CitronParser(nlp)
        self.filenames = utils.get_files(input_path)
        logger.info("Loading %s data files", len(self.filenames))
        self.index = 0
    
    
    def __iter__(self):
        """
        Iterate through the data files.
        
        Yields:
            A tuple containing three fields:
            - doc: A spaCy Doc object.
            - quotes: A list of citron.data.Quote objects.
            - coref_groups: A list of coreference groups (or null, if coreference data is not available). 
                Each coreference group is a list of spaCy Span objects.
        """
        
        while self.index < len(self.filenames):
            path = self.filenames[self.index]
            self.index += 1
            yield self.parser.parse(path)
    
    
    def __len__(self):
        return len(self.filenames)


class CitronParser():
    """
    Class which parses files files using Citron's JSON data format.
    """
    
    def __init__(self, nlp):
        """
        Constructor.
        
        Args:
            nlp: A spaCy Language object.
        """
        self.nlp = nlp
    
    
    def parse(self, input_file):
        """
        Parse a file.
        
        Args:
            input_file: The file path (string). 
        
        Returns:
            A tuple containing three fields:
            - doc: A spaCy Doc object.
            - quotes: A list of citron.data.Quote objects.
            - coref_groups: A list of coreference groups (or null, if coreference data is not available).
            
            Each coreference group is a list of spaCy Span objects.
        """
        
        try:
            with open(input_file, encoding="utf-8") as infile:
                data = json.load(infile)
        
        except IOError:
            logger.error("Error reading: %s", input_file)
            sys.exit(0)
             
        except json.JSONDecodeError:
            logger.error("JSON decoding error : %s", input_file)
            sys.exit(0)
        
        doc = self.nlp(data["text"])
        char_to_span_index = self._get_character_index(doc)
        
        if "quotes" in data:
            quotes = []
            
            for quote in data["quotes"]:
                cue = quote["cue"]
                cue_span = self._get_span(doc, char_to_span_index, cue["start"], cue["end"])
                sources = []
                
                for source in quote["sources"]:
                    source_span = self._get_span(doc, char_to_span_index, source["start"], source["end"])
                    sources.append(source_span)

                contents = []
                
                for content in quote["contents"]:
                    content_span = self._get_span(doc, char_to_span_index, content["start"], content["end"])                    
                    contents.append(content_span)
                
                if "coreferences" in quote and quote["coreferences"] is not None:
                    corefs = []
                    
                    for coref in quote["coreferences"]:
                        coref_span = self._get_span(doc, char_to_span_index, coref["start"], coref["end"])
                        corefs.append(coref_span)
                
                else:
                    corefs = None
                
                spacy_quote = Quote(cue_span, sources, contents, corefs)
                quotes.append(spacy_quote)
        
        else:
            quotes = None
        
        if "coreference_groups" in data and data["coreference_groups"] is not None:
            coref_groups = []
            
            for entry in data["coreference_groups"]:
                coref_group = []
                
                for coref in entry:
                    coref_span = self._get_span(doc, char_to_span_index, coref["start"], coref["end"])
                    coref_group.append(coref_span)
                
                coref_groups.append(coref_group)
        
        else:
            coref_groups = None
        
        return doc, quotes, coref_groups
    
    
    def _get_span(self, doc, char_to_span_index, start_char, end_char):
        """
        Get a span corresponding the specified start and end character indices.
        
        Args:
            doc: A spaCy Doc object.
            char_to_span_index: an index obtained from the _get_character_index method.
            start_char: the character index of the start of the span.
            end_char: the character index of the end of the span.
        """
        
        start = char_to_span_index[start_char]
        end   = char_to_span_index[end_char - 1] + 1
        
        if (start < 0 or start > len(doc.text) or end < 1 or  end > len(doc) or end <= start):
            print("Invalid span start:", start, "end:", end)
            return None
        
        return doc[start : end]
    
    
    def _get_character_index(self, doc):
        """
        Get a list mapping character indices to the index of tokens within 
        a spaCy Doc object.
        
        Args:
            doc: A spaCy Doc object.
            
        Returns:
            A list of token indices, one for each character in the text of the document.
        """
        
        length = len(doc.text)
        char_to_span_index = [None] * length
        
        for token in doc:
            for i in range(0, len(token) + len(token.whitespace_)):
                char_to_span_index[token.idx + i] = token.i
        
        return char_to_span_index


class Quote(object):
    """
    Class representing a single quote.
    
    Each quote has a cue, sources and contents. It may also have coreferences
    for the sources and a confidence score.
    
    The value of the coreferences field may be None (when input data does not 
    provide coreference information) or an empty list (when no coreferences 
    exist for the sources) The dimensions of coreferences may not match the 
    dimensions of sources e.g. the source "they" could have the coreferences 
    "Morecambe" and "Wise".
    
    The value of the confidence score is a float value for predicted quotes, 
    otherwise None.  
    """
    
    def __init__(self, cue, sources, contents, coreferences=None, confidence=None):
        """
        Constructor.
        
        All spans are spaCy Span objects.
        
        Args:
            cue: A quote cue span 
            sources: A list of source spans
            contents: A list of content spans
            coreferences: A list of coreference spans or None.
            confidence: A confidence score (float) or None.
        """
        
        self.cue = cue
        self.sources = sources
        self.contents = contents
        self.coreferences = coreferences
        self.confidence = confidence
    
    
    def get_cue_length(self):
        """
        Get the length of the quote cue.
        
        Returns: The length (int)
        """
        
        return len(self.cue)
    
    
    def get_sources_length(self):
        """
        Get the combined length of the quote sources.
        
        Returns: The length (int)
        """
        
        return utils.get_spans_length(self.sources)
    
    
    def get_contents_length(self):
        """
        Get the combined length of the quote contents.
        
        Returns: The length (int)
        """
        
        return utils.get_spans_length(self.contents)
    
    
    def get_coreferences_length(self):
        """
        Get the combined length of the quote coreferences.
        
        Returns: The length (int)
        """
        
        if self.coreferences is None:
            return 0
        else:
            return utils.get_spans_length(self.coreferences)
    
    
    def get_cue_overlap(self, other_quote):
        """
        Get the length of the overlap between the cue of this quote and another quote.
        
        Returns: The overlap length (int)
        """
        
        return utils.get_overlap(self.cue, other_quote.cue)  
    
    
    def get_sources_overlap(self, other_quote):
        """
        Get the length of the overlap between the sources of this quote and another quote.
        
        Returns: The overlap length (int)
        """
        
        return utils.get_spans_overlap(self.sources, other_quote.sources)            
    
    
    def get_contents_overlap(self, other_quote):
        """
        Get the length of the overlap between the contents of this quote and another quote.
        
        Returns: The overlap length (int)
        """
        
        return utils.get_spans_overlap(self.contents, other_quote.contents)
    
    
    def get_coreferences_overlap(self, other_quote):
        """
        Get the length of the overlap between the coreferences of this quote and another quote.
        
        Returns: The overlap length (int)
        """
                      
        if self.coreferences is None or other_quote.coreferences is None:
            return 0
        else:
            return utils.get_spans_overlap(self.coreferences, other_quote.coreferences) 
    
    
    def to_json(self):
        """
        Get a JSON serialisable representation using character indices to define spans.
        
        Returns: A JSON serialisable object.
        """
        
        sources = []
        coreferences = []
        contents = []
        
        if self.coreferences is not None and len(self.coreferences) > 0:
            for coreference in self.coreferences:
                sources.append(coreference._.to_json())
                
            for source in self.sources:
                coreferences.append(source._.to_json())
        
        else:
            for source in self.sources:
                sources.append(source._.to_json())
        
        for content in self.contents:
            contents.append(content._.to_json())
        
        quote_json = {}
        quote_json["sources"] = sources
        quote_json["coreferences"] = coreferences
        quote_json["cue"] = self.cue._.to_json()
        quote_json["contents"] = contents
        
        if self.confidence is not None:
            quote_json["confidence"] = '{0:.4f}'.format(self.confidence)
        
        return quote_json
