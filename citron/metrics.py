# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides metrics for measuring the overall performance of Citron
and the performance of its individual components.

Note that when components are used collectively, rather than individually, the 
performance is affected by the behaviour of components earlier in the pipeline. 
"""

from .data import DataSource    
from . import utils


def evaluate(citron, test_path):
    """
    Evaluate Citron and print exact and overlap metric scores.
    
    Args:
        test_path: The path (string) to a file or directory of Citron format JSON data files.
            Directories will be explored recursively.
    
    The exact match metrics require the predicted and correct spans to be an exact match.
    
    The overlap metrics are based on the number of predicted tokens that overlap with correct
    tokens compared to the total number of predicted and correct tokens.
    """
    
    total_predicted_quotes = 0
    total_actual_quotes = 0
    
    cue_exact_match_tp = 0
    cue_exact_match_fp = 0
    cue_exact_match_fn = 0

    source_exact_match_tp = 0
    source_exact_match_fp = 0
    source_exact_match_fn = 0

    content_exact_match_tp = 0
    content_exact_match_fp = 0
    content_exact_match_fn = 0
    
    quote_exact_match_tp = 0
    quote_exact_match_fp = 0
    quote_exact_match_fn = 0
    
    sum_of_cue_overlaps = 0
    sum_of_source_overlaps = 0
    sum_of_content_overlaps = 0
    
    sum_of_predicted_cue_lengths = 0
    sum_of_predicted_source_lengths = 0
    sum_of_predicted_content_lengths = 0
    
    sum_of_actual_cue_lengths = 0
    sum_of_actual_source_lengths = 0
    sum_of_actual_content_lengths = 0
    
    for doc, actual_quotes, _ in DataSource(citron.nlp, test_path):
        predicted_quotes = citron.get_quotes(doc, resolve_coreferences=False)
        total_predicted_quotes += len(predicted_quotes)
        total_actual_quotes += len(actual_quotes)
        
        scores = get_cue_exact_match_metrics(predicted_quotes, actual_quotes)
        cue_exact_match_tp += scores[0]
        cue_exact_match_fp += scores[1]
        cue_exact_match_fn += scores[2]
        
        scores = get_source_exact_match_metrics(predicted_quotes, actual_quotes)
        source_exact_match_tp += scores[0]
        source_exact_match_fp += scores[1]
        source_exact_match_fn += scores[2]
        
        scores = get_content_exact_match_metrics(predicted_quotes, actual_quotes)
        content_exact_match_tp += scores[0]
        content_exact_match_fp += scores[1]
        content_exact_match_fn += scores[2]
        
        scores = get_quote_exact_match_metrics(predicted_quotes, actual_quotes)
        quote_exact_match_tp += scores[0]
        quote_exact_match_fp += scores[1]
        quote_exact_match_fn += scores[2]
        
        scores = get_quote_overlap_counts_and_lengths(predicted_quotes, actual_quotes)
        sum_of_cue_overlaps                  += scores[0]
        sum_of_source_overlaps               += scores[1]
        sum_of_content_overlaps              += scores[2]
        sum_of_predicted_cue_lengths         += scores[3]
        sum_of_predicted_source_lengths      += scores[4]
        sum_of_predicted_content_lengths     += scores[5]
        sum_of_actual_cue_lengths            += scores[6]
        sum_of_actual_source_lengths         += scores[7]
        sum_of_actual_content_lengths        += scores[8]
    
    sum_of_quote_overlaps          = sum_of_cue_overlaps + sum_of_content_overlaps + sum_of_source_overlaps
    sum_of_quote_actual_lengths    = sum_of_actual_cue_lengths + sum_of_actual_content_lengths + sum_of_actual_source_lengths
    sum_of_quote_predicted_lengths = sum_of_predicted_cue_lengths + sum_of_predicted_content_lengths + sum_of_predicted_source_lengths
    
    cue_exact_scores   = get_exact_scores(cue_exact_match_tp, cue_exact_match_fp, cue_exact_match_fn)
    cue_overlap_scores = get_overlap_scores(sum_of_cue_overlaps, sum_of_predicted_cue_lengths, sum_of_actual_cue_lengths)
    
    source_exact_scores   = get_exact_scores(source_exact_match_tp, source_exact_match_fp, source_exact_match_fn)
    source_overlap_scores = get_overlap_scores(sum_of_source_overlaps, sum_of_predicted_source_lengths, sum_of_actual_source_lengths)
    
    content_exact_scores   = get_exact_scores(content_exact_match_tp, content_exact_match_fp, content_exact_match_fn)
    content_overlap_scores = get_overlap_scores(sum_of_content_overlaps, sum_of_predicted_content_lengths, sum_of_actual_content_lengths)
    
    quote_exact_scores   = get_exact_scores(quote_exact_match_tp, quote_exact_match_fp, quote_exact_match_fn)
    quote_overlap_scores = get_overlap_scores(sum_of_quote_overlaps, sum_of_quote_actual_lengths, sum_of_quote_predicted_lengths)
    
    print("================================================================================")
    print()
    print("Total predicted quotes:", total_predicted_quotes)
    print("Total actual quotes:   ", total_actual_quotes)
    print()
    print("---- Cue Span exact match metrics ----")
    print_metrics(*cue_exact_scores)
    print()
    print("---- Cue Span overlap metrics ----")
    print_metrics(*cue_overlap_scores)
    print()
    print("---- Source Spans exact match metrics ----")
    print_metrics(*source_exact_scores)
    print()
    print("--- Source Spans overlap metrics ----") 
    print_metrics(*source_overlap_scores)        
    print()
    print("---- Content Spans exact match metrics ----")
    print_metrics(*content_exact_scores)
    print()
    print("---- Content Spans overlap metrics ----")
    print_metrics(*content_overlap_scores)
    print()
    print("---- All Quote Spans exact match metrics ----")
    print_metrics(*quote_exact_scores)
    print()
    print("---- All Quote Spans overlap metrics ----")
    print_metrics(*quote_overlap_scores)


def get_cue_exact_match_metrics(predicted_quotes, actual_quotes):
    """
    Get the exact metric score for quote cues.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing:
            the true positive count (int)
            the false positive count (int)
            the false negative count (int)    
    """
        
    tp = 0
    fp = 0
    fn = 0

    for predicted_quote in predicted_quotes:
        for actual_quote in actual_quotes:   
            if utils.are_matching_spans(predicted_quote.cue, actual_quote.cue):
                tp += 1
                break
        else:
            fp += 1
             
    for actual_quote in actual_quotes:
        for predicted_quote in predicted_quotes:
            if utils.are_matching_spans(predicted_quote.cue, actual_quote.cue):
                break
        else:
            fn += 1
            
    return tp, fp, fn


def get_source_exact_match_metrics(predicted_quotes, actual_quotes):
    """
    Get the exact metric score for quote sources.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing:
            the true positive count (int)
            the false positive count (int)
            the false negative count (int)    
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    for predicted_quote in predicted_quotes:
        for actual_quote in actual_quotes:
            if utils.are_overlapping_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.sources, actual_quote.sources):
                    # Matching cue and source
                    tp += 1
                    break
        else:
            # No match
            fp += 1
    
    for actual_quote in actual_quotes:
        for predicted_quote in predicted_quotes:
            if utils.are_overlapping_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.sources, actual_quote.sources):
                    # Matching source and cue
                    break
        else:
            # No match
            fn += 1
            
    return tp, fp, fn


def get_content_exact_match_metrics(predicted_quotes, actual_quotes):
    """
    Get the exact metric score for quote contents.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing:
            the true positive count (int)
            the false positive count (int)
            the false negative count (int)    
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    for predicted_quote in predicted_quotes:
        for actual_quote in actual_quotes:
            if utils.are_overlapping_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.contents, actual_quote.contents):
                    # Matching cue and content 
                    tp += 1
                    break
            
        else:
            # No match
            fp += 1
    
    for actual_quote in actual_quotes:
        for predicted_quote in predicted_quotes:
            if utils.are_overlapping_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.contents, actual_quote.contents):
                    # Matching cue and content
                    break
        else:
            # No match
            fn += 1
            
    return tp, fp, fn


def get_quote_exact_match_metrics(predicted_quotes, actual_quotes): 
    """
    Get the exact metric score for quotes.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing:
            the true positive count (int)
            the false positive count (int)
            the false negative count (int)    
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    for predicted_quote in predicted_quotes:
        for actual_quote in actual_quotes:
            if utils.are_matching_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.sources, actual_quote.sources):
                    if utils.are_matching_span_lists(predicted_quote.contents, actual_quote.contents):
                        # Matching cue, source and span
                        tp += 1
                        break
        else:
            # No match
            fp += 1
    
    for actual_quote in actual_quotes:
        for predicted_quote in predicted_quotes:
            if utils.are_matching_spans(predicted_quote.cue, actual_quote.cue):
                if utils.are_matching_span_lists(predicted_quote.sources, actual_quote.sources):
                    if utils.are_matching_span_lists(predicted_quote.contents, actual_quote.contents):
                        # Matching source and span
                        break
        else:
            # No match
            fn += 1
    
    return tp, fp, fn


def get_quote_overlap_counts_and_lengths(predicted_quotes, actual_quotes):
    """
    Get the token overlap counts and lengths for quote cues, sources and contents.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing nine overlap counts and lengths (int)  
    """
    
    sum_of_cue_overlaps = 0
    sum_of_source_overlaps = 0
    sum_of_content_overlaps = 0
    sum_of_predicted_cue_lengths = 0
    sum_of_predicted_source_lengths = 0
    sum_of_predicted_content_lengths = 0
    sum_of_actual_cue_lengths = 0
    sum_of_actual_source_lengths = 0
    sum_of_actual_content_lengths = 0
        
    for predicted_quote in predicted_quotes:
        sum_of_predicted_cue_lengths     += predicted_quote.get_cue_length()
        sum_of_predicted_source_lengths  += predicted_quote.get_sources_length()
        sum_of_predicted_content_lengths += predicted_quote.get_contents_length()
        
        for actual_quote in actual_quotes:
            sum_of_cue_overlaps += predicted_quote.get_cue_overlap(actual_quote)
            sum_of_source_overlaps  += predicted_quote.get_sources_overlap(actual_quote)
            sum_of_content_overlaps += predicted_quote.get_contents_overlap(actual_quote)         
    
    for actual_quote in actual_quotes:
        sum_of_actual_cue_lengths     += actual_quote.get_cue_length()
        sum_of_actual_source_lengths  += actual_quote.get_sources_length()
        sum_of_actual_content_lengths += actual_quote.get_contents_length()
    
    return (sum_of_cue_overlaps,
            sum_of_source_overlaps,
            sum_of_content_overlaps,
            sum_of_predicted_cue_lengths, 
            sum_of_predicted_source_lengths,
            sum_of_predicted_content_lengths,
            sum_of_actual_cue_lengths,
            sum_of_actual_source_lengths,
            sum_of_actual_content_lengths)


def get_span_exact_match_metrics(actual_spans, predicted_spans):
    """
    Get the exact metric score for a list of spans.
    
    Args:
        predicted_quotes: A list of citron.data.Quote objects.
        actual_quotes: A list of citron.data.Quote objects.
    
    Returns:
        A tuple containing:
            the true positive count (int)
            the false positive count (int)
            the false negative count (int)    
    """
    
    tp = 0
    fp = 0
    fn = 0
    
    for predicted_span in predicted_spans:
        for actual_span in actual_spans:
            if actual_span.start == predicted_span.start and actual_span.end == predicted_span.end:
                tp += 1
                break
        else:
            fp += 1
    
    for actual_span in actual_spans:
        for predicted_span in predicted_spans:
            if actual_span.start == predicted_span.start and actual_span.end == predicted_span.end:
                break
        else:
            fn += 1
            
    return tp, fp, fn


def get_span_overlap_counts_and_lengths(predicted_spans, actual_spans):
    """
    Get the token overlap counts and lengths for quote cues, sources and contents.
    
    Args:
        predicted_quotes: A list of spaCy Span objects.
        actual_quotes: A list of spaCy Span objects.
    
    Returns:
        A tuple containing nine overlap counts and lengths (int)  
    """
    
    sum_of_overlaps          = utils.get_spans_overlap(predicted_spans, actual_spans)    
    sum_of_predicted_lengths = utils.get_spans_length(predicted_spans)
    sum_of_actual_lengths    = utils.get_spans_length(actual_spans)
    
    return sum_of_overlaps, sum_of_predicted_lengths, sum_of_actual_lengths


def get_exact_scores(tp, fp, fn):
    """
    Get the Precision, Recall and F1 metrics based on the input counts.
    
    Args:
        tp: the number of true positives.
        fp: the number of false positives.
        fn: the number of false negatives.
    
    Returns:
       A tuple containing
           the Precision (float)
           the Recall (float)
           the F1 score (float)
    """
    
    if tp + fp > 0:
        precision = float(tp) / (tp + fp)
    else:
        precision = 0.0
    
    if tp + fn > 0:
        recall = float(tp) / (tp + fn)
    else:
        recall = 0.0
    
    if precision + recall > 0:    
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1 


def get_overlap_scores(sum_of_overlaps, sum_of_predicted_lengths, sum_of_actual_lengths):
    """
    Get the Precision, Recall and F1 metrics based on the summed overlaps and lengths.
    
    Args:
        sum_of_overlaps: the total number of predicted tokens that overlap with actual tokens.
        sum_of_predicted_lengths: the total number of predicting tokens.
        sum_of_actual_lengths: the total number of actual tokens.
    
    Returns:
       A tuple containing
           the Precision (float)
           the Recall (float)
           the F1 score (float)
    """
    
    if sum_of_predicted_lengths > 0:
        precision = sum_of_overlaps / sum_of_predicted_lengths
    else:
        precision = 0.0
        
    if sum_of_actual_lengths > 0:
        recall = sum_of_overlaps / sum_of_actual_lengths
    else:
        recall = 0.0
        
    if precision + recall > 0:    
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def print_metrics(precision, recall, f1):
    """
    Print a set of metrics.
    
    Args:
        precision: float
        recall: float
        f1: float
    """
    
    print("Precision:", "{:1.4f}".format(precision))
    print("Recall:   ", "{:1.4f}".format(recall))
    print("F1:       ", "{:1.4f}".format(f1))
