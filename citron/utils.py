# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides utility functions.
"""

import os

import spacy
from spacy.tokens import Span

from .logger import logger


def get_parser():
    """
    Loads a spaCy pipeline and adds extension functions.
    
    Returns:
        A spaCy Language object.
    """
    
    logger.info("Loading spacy model")
    nlp = spacy.load("en_core_web_sm")
    to_json = lambda span: {"start": span.start, "end": span.end, "text": span.text}
    
    Span.set_extension("to_json", method=to_json)
    Span.set_extension("probability", default=None, force=True)
    Span.set_extension("is_plural",   default=None, force=True)
    Span.set_extension("gender",      default=None, force=True)
    return nlp


def get_files(path):
    """
    Get a list of JSON file paths using a recursive search of the supplied path.    
    
    Args:
        path: The path to search.
    
    Returns:
        A list of file path strings.
    """
    
    paths = []
    add_files(path, paths)
    return paths


def add_files(path, paths):
    """
    Add the paths of JSON files to the supplied list using a recursive search..
    
    Args:
        path:  The path to search.
        paths: The list to add to.
    """
    
    if os.path.isdir(path):
        for entry in os.listdir(path):
            sub_path = os.path.join(path, entry)
            add_files(sub_path, paths) 
            
    elif path.endswith(".json"):
        paths.append(path)


def conform_labels(input_labels):
    """
    Get a revised list of IOB labels which follow the conform to the
    sequence rules e.g. I cannot follow O.
    
    Args:
        input_labels: A list of IOB labels which may be non-conformant.
    
    Returns:
        output_labels: A conformant list of IOB labels.
    """
    
    output_labels = []
    last = "O"     
    
    for i in range(0, len(input_labels)):
        label = input_labels[i]    
        
        if last == "B":
            if label == "B":
                label = "I"
        
        elif last == "O":
            if label == "I":
                label = "B"
        
        last = label
        output_labels.append(label)
    
    return output_labels


def get_spans(doc, input_labels):
    """
    Get a list of span objects defined by the supplied IOB labels.
    
    Args:
        doc: A spaCy Doc object.
        input_labels: A list containing an IOB label for each token in the document.
    
    Returns:
        A list of spaCy Span objects.
    """
    
    spans = []
    start = None
       
    for i in range(0, len(input_labels)):
        label = input_labels[i]
        
        if label == "B":
            start = i
        elif label == "O":
            if start is not None:
                # Add span
                span = doc[start : i]
                start = None
                spans.append(span)
    
    return spans


def get_spans_or_entities(doc, input_labels):
    """
    Get a list of span objects conforming to the supplied labels using
    named entity spans from the doc when available.
    
    Args:
        doc: A spaCy Doc object.
        input_labels: A list containing an IOB label for each token in the document.
    
    Returns:
        A list of spaCy Span objects.
    """
    
    spans = []
    start = None
        
    for i in range(0, len(input_labels)):
        label = input_labels[i]
         
        if label == "B":
            start = i
             
        elif label == "O":
            if start is not None:
                # Add span
                span = doc[start : i]
                start = None
                entity = get_matching_entity(span)
                 
                if entity is not None:
                    spans.append(entity)
                else:
                    spans.append(span)
    
    return spans


def get_matching_entity(span):
    """
    Get a named entity span matching the supplied span when available.
    
    Args:
        span: A spaCy Span object.
    
    Returns:
        A spaCy Span object, or None.
    """
    
    for entity in span.doc.ents:
        if span.start == entity.start and span.end == entity.end:
            return entity
    
    return None


def is_same_sentence(span1, span2):
    """
    Test whether two spans are in the same sentence.
    
    Args:
        span1: A spaCy Span object.
        span2: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    return span1.sent == span2.sent


def is_ancestor_of(target_span, candidate_span):
    """
    Test whether any token in the candidate_span is an ancestor of a token in the target span.
    
    Args:
        target_span: A spaCy Span object.
        candidate_span: A spaCy Span object.
    
    Returns:
        A boolean value.        
    """
    
    for candidate_ancestor_token in candidate_span:
        for target_token in target_span:
            if candidate_ancestor_token.is_ancestor(target_token):
                return True
    
    return False


def span_has_labels(span, labels):
    """
    Test whether there are B or I labels within a span. 
    
    Args:
        span: A spaCy Span object.
        labels: A list containing an IOB label for each token in the document.
    
    Returns:
        A boolean value.
    """
    
    for i in range(span.start, span.end):
        if labels[i] != "O":
            return True
    
    return False


def are_matching_spans(span1, span2):
    """
    Test whether two spans are equivalent.
    
    Args:
        span1: A spaCy Span object.
        span2: A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    return span1.start == span2.start and span1.end == span2.end


def are_matching_span_lists(spans1, spans2):
    """
    Test whether two span lists are equivalent.
    
    Args:
        span1: A list of spaCy Span objects.
        span2: A list of spaCy Span objects.
    
    Returns:
        A boolean value.
    """
    
    if len(spans1) != len(spans2):
        return False
    
    for i in range(0, len(spans1)):
        span1 = spans1[i]
        span2 = spans2[i]

        if span1 is not None:
            if span2 is not None:
                if span1.start != span2.start or span1.end != span2.end:
                    return False
            else:
                return False
        
        elif span2 is not None:
            return False
    
    return True


def get_inside_quotation_marks_labels(doc):
    """
    Get a list of binary labels identifying tokens that are within 
    quotation marks.
    
    Args:
        doc: A spaCy Doc object.
    
    Returns:
        A list containing a binary label for each token in the document.
    """
    
    labels = [0] * len(doc)
    in_quotes = False
    
    for i in range(0, len(doc)):
        token = doc[i]
        
        if in_quotes:
            labels[i] = 1
        else:
            labels[i] = 0
        
        if token.pos_ == "PUNCT":
            if token.tag_ == "``":
                in_quotes = True
                labels[i] = 1
            
            elif token.tag_ in ("''", "\"\""):
                in_quotes = False
    
    return labels


def get_sentence_section_labels(doc):
    """
    Get a list of integer labels identifying sections within each sentence. 
    Sections are defined by commas or parentheses.
    
    Args:
        doc: A spaCy Doc object.
    
    Returns:
       A list containing an integer label for each token in the document.
    """
    
    labels = [0] * len(doc)
    in_commas = False
    label = 0
    
    for sentence in doc.sents:
        sentence_label = label
             
        for token in sentence:
            if token.text == "," or token.text == "(" or token.text == ")":
                if not in_commas:
                    label += 1
                
                in_commas = not in_commas
                labels[token.i] = label
            
            else:
                if in_commas:
                    labels[token.i] = label
                else:
                    labels[token.i] = sentence_label
        
        label += 1
     
    return labels


def is_inside_spans(index, spans):
    """
    Test whether a token index is inside any span in a span list.
    
    Args:
        index: A token index (int)
        spans: A list of spaCy Span objects.
        
    Returns:
        A boolean label.     
    """
    
    for span in spans:
        if is_inside_span(index, span):
            return True
    
    return False


def is_inside_span(index, span):
    """
    Test whether a token index is inside a span.
    
    Args:
        index: A token index (int)
        span: A spaCy Span object.
        
    Returns:
        A boolean label.
    """
    
    return span.start <= index < span.end


def are_overlapping_spans(span1, span2):
    """
    Test whether two spans overlap. 
        
    Args:
        span1: A spaCy Span object.
        span2: A spaCy Span object.
    
    Returns:
        A boolean label.
    """
    
    if span1.start <= span2.end and span1.end > span2.start:
        return True
        
    return False 


def are_overlapping_span_lists(spans1, spans2):
    """
    Test whether any span in spans1 overlaps with any span in spans2.
    
    Args:
        spans1: A list of spaCy Span objects.
        spans2: A list of spaCy Span objects.
    
    Returns:
        A boolean label.
    """
    
    for span1 in spans1:
        for span2 in spans2:
            if span1.start <= span2.end and span1.end > span2.start:
                return True
    
    return False


def get_overlap(span1, span2):
    """
    Get the number of tokens which overlap between two spans.
        
    Args:
        span1: A spaCy Span object.
        span2: A spaCy Span object.
    
    Returns:
        An integer value
    """
    
    start = min(span1.end, max(span1.start, span2.start))
    end = max(span1.start, min(span1.end, span2.end))
    return end - start


def get_spans_overlap(spans1, spans2):
    """
    Get the number of tokens which overlap between two lists of spans.
        
    Args:
        spans1: A list of spaCy Span objects.
        spans2: A list of spaCy Span objects.
    
    Returns:
        An integer value      
    """
    
    overlap = 0

    for span1 in spans1:
        for span2 in spans2:
            if span1 is not None and span2 is not None:
                overlap += get_overlap(span1, span2)
    
    return overlap


def get_spans_length(spans):
    """
    Get the total number of tokens in a list of spans.
        
    Args:
        spans: A list of spaCy Span objects.   
        
    Returns:
        An integer value.
    """
    
    length = 0
    
    for span in spans:
        if span is not None:
            length += len(span)
     
    return length


def get_spans_within_span(spans, span):
    """
    Get the spans which lie within another span. 
    
    Args:
        spans: A list of spaCy Span objects.
        span: A spaCy Span object.  
    
    Returns:
        A list of spaCy Span objects.
    """
    
    filtered_spans = []
    
    for candidate_span in spans:
        if candidate_span.start >= span.start and candidate_span.end <= span.end:
            filtered_spans.append(candidate_span)
    
    return filtered_spans   


def get_cues(quotes):
    """
    Get a list of cue spans from a list of quotes.     
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list of spaCy Span objects.
    """
    
    cues = []
    
    for quote in quotes:
        cues.append(quote.cue)
    
    return cues


def get_sources(quotes):
    """
    Get a list of source spans from a list of quotes.     
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list of spaCy Span objects.
    """
    
    source_spans = []
    
    for quote in quotes:
        for source in quote.sources:
            source_spans.append(source)
    
    return source_spans


def get_contents(quotes):
    """
    Get a list of content spans from a list of quotes.     
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list of spaCy Span objects.     
    """
    
    content_spans = []
       
    for quote in quotes:
        for content in quote.contents:
            content_spans.append(content)
               
    return content_spans


def get_cue_iob_labels(doc, quotes):
    """
    Get a list IOB labels from a list of quotes representing all the quote cues.
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list containing an IOB label for each token in the document.
    """
    
    labels = ["O"] * len(doc)
      
    for quote in quotes:
        labels[quote.cue.start] = "B"
        
        for i in range(quote.cue.start + 1, quote.cue.end):
            labels[i] = "I"
              
    return labels


def get_source_iob_labels(doc, quotes):
    """
    Get a list IOB labels from a list of quotes representing all the quote sources.
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list containing an IOB label for each token in the document. 
    """
    
    labels = ["O"] * len(doc)
       
    for quote in quotes:
        for source in quote.sources:
            start = source.start
            end = source.end
            labels[start] = "B"
               
            for i in range(start + 1, end):
                labels[i] = "I"
               
    return labels


def get_content_iob_labels(doc, quotes):
    """
    Get a list IOB labels from a list of quotes representing all the quote contents.
    
    Args:
        quotes: A list of citron.data.Quote objects.
    
    Returns:
        A list containing an IOB label for each token in the document.
    """
    
    labels = ["O"] * len(doc)
       
    for quote in quotes:
        for content in quote.contents:
            start = content.start
            end = content.end
            labels[start] = "B"
               
            for i in range(start + 1, end):
                labels[i] = "I"
               
    return labels


def is_person(span):
    """
    Test whether a span is a labelled as a person.
    
    Args:
        A spaCy Span object.
    
    Returns:
        A boolean value.
    """
        
    if span.label_ == "PERSON":
        return True
    
    for token in span:
        if token.ent_type_ == "PERSON":
            return True
    
    return False 


def is_organisation(span):
    """
    Test whether a span is a labelled as an organisation.
    
    Args:
        A spaCy Span object.
    
    Returns:
        A boolean value.
    """
    
    if span.label_ == "ORG":
        return True
    
    for token in span:
        if token.ent_type_ == "ORG":
            return True
        
    return False


def get_iob_labels_for_spans(doc, spans):
    """
    Get a list of IOB labels representing a list of spans.     
    
    Args:
        quotes: A list of spaCy Span objects.
    
    Returns:
        A list containing an IOB label for each token in the document.
    """
    
    labels = ["O"] * len(doc)
    
    for span in spans:
        start = span.start
        end = span.end
        labels[start] = "B"
            
        for i in range(start + 1, end):
            labels[i] = "I"
    
    return labels


def get_dependency_depth(token):
    """
    Get the dependency depth of a token.
    
    Args:
        token: a spaCy Token object.
    
    Returns:
        An integer value.
    """
    
    parent = token
    depth = 0
       
    while parent != parent.head:                    
        depth += 1
        parent = parent.head
    
    return depth


def get_dependency_root_and_depth(token):
    """
    Get the dependency root and depth of a token.
    
    Args:
        token: a spaCy Token object.
    
    Returns:
        A tuple containing:
            a token (the root ancestor).
            an integer value (the depth).
    """
    
    parent = token
    depth = 0
    
    while parent != parent.head:
        depth += 1
        parent = parent.head
       
    return parent, depth


def get_index_of_max(values):
    """
    Get the index in a list of the tuples, of the tuple with the highest value in its second element.

    Args:
        values: a list of tuples, where the second element of each tuple is a numerical value.
    
    Returns:
        An integer value.
    """
    
    index_of_max = 0
    max_value = 0.0
    
    for i in range(0, len(values)):
        value = values[i][1]
        
        if value > max_value:
            max_value = value
            index_of_max = i
            
    return index_of_max, max_value
