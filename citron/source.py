# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides methods to identify quote sources and their associated cue.
"""

import datetime
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pycrfsuite

from .data import DataSource
from .logger import logger
from . import metrics
from . import utils


class SourceClassifier():
    """
    Classifier which identifies quote sources. A sentence-based approach is
    adopted as quote sources should not cross sentence boundaries.
    
    Based on the paper:
    "Automatically Detecting and Attributing Indirect Quotations"
    Silvia Pareti, Timothy O'Keefe, Ioannis Konstas, James R Curran, Irena Koprinska
    Proceedings of the Conference on Empirical Methods in Natural Language Processing
    (EMNLP), Seattle, U.S.
    """
    
    MODEL_FILENAME = "source-classifier.crfsuite"
    MAX_OFFSET = 5
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Source Classifier model: %s", filename)
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(filename)
    
    
    def predict_sources_and_labels(self, doc, cue_labels, content_labels):
        """
        Predict source spans and labels for a document.
        
        Args:
            doc: A spaCy Doc object.
            cue_labels: A list containing an IOB label for each token in the document.
            content_labels: A list containing an IOB label for each token in the document.
        
        Returns:
            A tuple containing:
                source_spans: A list of spaCy Span objects.
                source_labels: A list containing an IOB label for each token in the document.
        """
        
        doc_features_and_labels = self._get_features_and_labels(doc, cue_labels, content_labels)
        source_labels = []
        
        for sentence_features_and_labels in doc_features_and_labels:
            sentence_features = sentence_features_and_labels[0]
            sentence_source_labels = self._tagger.tag(sentence_features)
            source_labels.extend(sentence_source_labels)
        
        source_labels = self._remove_trailing_apostrophes(doc, source_labels)
        source_spans = utils.get_spans_or_entities(doc, source_labels)
        return source_spans, source_labels
    
    
    def evaluate(self, nlp, test_path):
        """
        Evaluate the Source Classifier.
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron format JSON data files.
            Directories will be explored recursively.
        """
        
        logger.info("Evaluating Source Classifier model using: %s", test_path)
        
        exact_tp = 0
        exact_fp = 0
        exact_fn = 0
        
        sum_of_overlaps = 0
        sum_of_predicted_lengths = 0
        sum_of_actual_lengths = 0
        
        for doc, quotes, _ in DataSource(nlp, test_path):
            cue_labels = utils.get_cue_iob_labels(doc, quotes)                   
            content_labels = utils.get_content_iob_labels(doc, quotes)
            source_count = 0
            content_count = 0
            
            for quote in quotes:
                source_count += len(quote.sources)
                content_count += len(quote.contents)
            
            doc_features_and_labels = self._get_features_and_labels(doc, cue_labels, content_labels, quotes)
            
            predicted_source_labels = []
            actual_source_labels = []
            
            for sentence_features_and_labels in doc_features_and_labels:
                sentence_features = sentence_features_and_labels[0]
                
                predicted_sentence_labels = self._tagger.tag(sentence_features)
                actual_sentence_labels = sentence_features_and_labels[1]
                predicted_source_labels.extend(predicted_sentence_labels)
                actual_source_labels.extend(actual_sentence_labels)
            
            actual_source_labels = self._remove_trailing_apostrophes(doc, actual_source_labels)
            actual_sources = utils.get_spans_or_entities(doc, actual_source_labels)
            
            predicted_source_labels = self._remove_trailing_apostrophes(doc, predicted_source_labels)
            predicted_sources = utils.get_spans_or_entities(doc, predicted_source_labels)
            
            tp, fp, fn = metrics.get_span_exact_match_metrics(actual_sources, predicted_sources)             
            exact_tp += tp
            exact_fp += fp
            exact_fn += fn
            
            scores = metrics.get_span_overlap_counts_and_lengths(actual_sources, predicted_sources)
            sum_of_overlaps += scores[0]
            sum_of_predicted_lengths += scores[1]
            sum_of_actual_lengths += scores[2]
        
        print("--------  Exact metrics  --------")
        exact_scores = metrics.get_exact_scores(exact_tp, exact_fp, exact_fn)
        metrics.print_metrics(*exact_scores)
        print()
        print("--------  Overlap metrics  --------")
        overlap_scores = metrics.get_overlap_scores(sum_of_overlaps, sum_of_predicted_lengths, sum_of_actual_lengths)
        metrics.print_metrics(*overlap_scores)
    
    
    @staticmethod
    def build_model(nlp, train_path, model_path):
        """
        Build and save a Source Classifier model.
        
        Args:
            nlp: A spaCy Language object.          
            train_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
            model_path: The path (string) to the Citron model directory.            
        """
        
        logger.info("Building Source Classifier model using: %s", train_path)
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        trainer = pycrfsuite.Trainer(verbose=False)
        
        for doc, quotes, _ in DataSource(nlp, train_path):
            cue_labels = utils.get_cue_iob_labels(doc, quotes)
            content_labels = utils.get_content_iob_labels(doc, quotes)
            
            doc_features_and_labels = SourceClassifier._get_features_and_labels(doc, cue_labels, content_labels, quotes)

            for sentence_features, sentence_labels in doc_features_and_labels:
                trainer.append(sentence_features, sentence_labels)
        
        trainer.set_params({
            "c1": 1.0,
            "c2": 1e-3,
            "max_iterations": 100,
            "feature.possible_transitions": True
        })
          
        logger.info("Training Source Classifier model")
        filename = os.path.join(model_path, SourceClassifier.MODEL_FILENAME)
        trainer.train(filename)
        logger.info("Training complete")
        logger.debug("Last_iteration: %s", trainer.logparser.last_iteration)
    
    
    @staticmethod
    def _get_features_and_labels(doc, cue_labels, content_labels, quotes=None):
        """
        Get the features and labels for all tokens in each sentence of a doc.
        
        Args:
            doc: A spaCy Doc object.
            cue_labels: A list containing an IOB label for each token in the document.
            content_labels: A list containing an IOB label for each token in the document.
            quotes: A list of citron.data.Quote objects or None.
                
        Returns:
            A list of tuples (one for each sentence). Each tuple contains:
            - sentence_features: A list of feature dicts.
            - sentence_labels: A list of binary labels.
        """
        
        doc_features_and_labels = []
        in_quotes = False
        
        if quotes is not None:
            # Training and evaluation
            cue_labels = utils.get_cue_iob_labels(doc, quotes)
            source_labels = utils.get_source_iob_labels(doc, quotes)
        
        else:
            # Prediction
            source_labels = None
            
        # Create features
        for sentence in doc.sents:
            sentence_features = []
            sentence_labels = []
            
            # Get sentence features
            #comma_isolated_spans = utils.getComma_isolated_spans(sentence)
            sentence_has_cue = False
            
            for token in sentence:
                if cue_labels[token.i] != "O":
                    sentence_has_cue = True
            
            # Create word features
            for token in sentence:
                index = token.i  # word index (in sentence)
                
                if token.pos_ == "PUNCT":
                    if token.tag_ == "``":
                        in_quotes = not in_quotes
                        
                    elif token.tag_ == "''" or token.tag_ == "\"\"":
                        in_quotes = False
                        
                # Word features
                word_features = []
                word_features.append("text=" + token.text)
                word_features.append("lemma=" + token.lemma_)
                word_features.append("pos=" + token.pos_)
                word_features.append("tag=" + token.tag_)
                word_features.append("ent_iob=" + token.ent_iob_)
                word_features.append("ent_type=" + token.ent_type_)
                word_features.append("content_label=" + content_labels[token.i])
                
                # Neighbour related features
                for n in range(0, SourceClassifier.MAX_OFFSET):
                    if index - n - 1 >= 0:
                        word_features.append("previous" + str(n) + "=" + doc[index - n - 1].text)
                        
                for n in range(0, SourceClassifier.MAX_OFFSET):
                    if index + n + 1 < len(doc):
                        word_features.append("next" + str(n) + "=" + doc[index + n + 1].text)
                        
                # Environment features
                if in_quotes:
                    word_features.append("in_quotes")
                    
                # Tree features
                parent = token
                depth = 0
                ancestor_is_cue = False
                parent_cue = None
                
                while parent != parent.head:
                    if cue_labels[parent.i] != "O":
                        ancestor_is_cue = True
                        parent_cue = parent
                        
                    depth += 1
                    parent = parent.head
                    
                if parent_cue is not None:
                    word_features.append("distance_from_cue=" + str(token.i - parent_cue.i))
                    
                word_features.append("depth=" + str(depth))
                word_features.append("dep=" + token.head.dep_)           
                word_features.append("ancestor_is_cue=" + str(ancestor_is_cue))
                
                if ancestor_is_cue and token.left_edge == token:
                    word_features.append("left_most")
                    
                if ancestor_is_cue and token.right_edge == token:
                    word_features.append("right_most")
                
                word_features.append("sentence_has_cue=" + str(sentence_has_cue))
                word_features.append("sentence_word=" + str(index))
                word_features.append("sentence_length=" + str(len(sentence)))  
                
                sentence_features.append(word_features)
                
                if source_labels is not None:
                    sentence_labels.append(source_labels[index])
            
            doc_features_and_labels.append((sentence_features, sentence_labels))
        
        return doc_features_and_labels
    
    
    @staticmethod
    def _remove_trailing_apostrophes(doc, labels):
        """
        Remove trailing apostrophes from the ends of spans defined by a list of
        IOB labels and correct the labels to match the trimmed spans.
        
        Args:
            doc: A spaCy Doc object.
            labels: A list containing an IOB label for each token in the document.
        
        Returns:
            corrected_labels: A list containing an IOB label for each token in the document.    
        """
        
        corrected_labels = ["O"] * len(labels)
        
        for i in range(0, len(labels)):
            label = labels[i]
            
            if i + 1 < len(labels):
                next_label = labels[i + 1]
            else:
                next_label = "O"
            
            if label == "I" and next_label == "O":
                token = doc[i]
                
                if token.tag_ == "POS":
                    label = "O"
            
            corrected_labels[i] = label
        
        return corrected_labels


class SourceResolver():
    """
    Class which resolves the relationship between quote sources and cues.
    
    Note that although Citron's Quote class can support multiple sources
    this source resolver always returns one source span per quote.
    """
    
    MODEL_FILENAME = "source-resolver.crfsuite"
    PROBABILITY_THRESHOLD = 0.01
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Source Resolver model: %s", filename)
        
        with open(filename, "rb") as infile:
            self._model = pickle.load(infile)
    
    
    def resolve_sources(self, sources, cues, sentence_section_labels):
        """
        Resolve which source span is associated with each quote cue.
        
        Args:
            sources: A list of spaCy Span objects.
            cues: A list of spaCy Span objects.
            sentence_section_labels: A list containing an integer label each token in the document.
        
        Returns:
            A dict mapping each cue to a source span. The cue is represented by a 
            tuple containing the start and end index.
        """
        
        quote_cue_to_sources_map = {}
        
        for cue in cues:
            predicted_source, probability = self.predict_source(sources, cue, sentence_section_labels)
            
            if predicted_source is not None:
                key = (cue.start, cue.end)  
                predicted_source._.probability = probability
                quote_cue_to_sources_map[key] = predicted_source
        
        return quote_cue_to_sources_map
    
    
    def predict_source(self, sources, cue, sentence_section_labels):
        """
        Predict the source associated with a quote queue.
        
        Args:
            sources: A list of spaCy Span objects.
            cue: A spaCy Span object.
            sentence_section_labels: A list containing an integer label each token in the document.
        
        Returns:
            A tuple containing:
                predicted_source: A spaCy Span object.
                probability: A float value between zero and one.
        """
        
        candidate_sources = utils.get_spans_within_span(sources, cue.sent)
        
        if len(candidate_sources) == 0:
            return None, 0.0
        
        features = []
        
        for candidate_source in candidate_sources:
            candidate_features = self._get_features(candidate_source, cue, sentence_section_labels)
            features.append(candidate_features)
        
        test_vectors = self._model["vectorizer"].transform(features)
        predicted_probabilities = self._model["classifier"].predict_proba(test_vectors)
        
        predicted_index, probability = utils.get_index_of_max(predicted_probabilities)
        
        if probability >= self.PROBABILITY_THRESHOLD:
            predicted_source = candidate_sources[predicted_index]
        else:
            predicted_source = None
        
        return predicted_source, probability
        
    
    def evaluate(self, nlp, test_path):
        """
        Evaluate the Source Resolver.
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        logger.info("Evaluating Source Resolver model using: %s", test_path)
        tp = 0
        fp = 0
        fn = 0
        
        for doc, quotes, _ in DataSource(nlp, test_path):
            sentence_section_labels = utils.get_sentence_section_labels(doc)
            
            # There are no annotations of potential sources so use predicted sources as candidates.
            candidate_sources = utils.get_sources(quotes)
            
            for quote in quotes:
                predicted_source = self.predict_source(candidate_sources, quote.cue, sentence_section_labels)[0]
                
                if predicted_source is None:
                    fn += 1
                
                else:
                    if utils.are_overlapping_span_lists([predicted_source], quote.sources):
                        tp += 1
                        
                    else:
                        fp += 1
                        fn += 1
        
        print("--------  Metrics  --------")
        exact_scores = metrics.get_exact_scores(tp, fp, fn)
        metrics.print_metrics(*exact_scores)
    
    
    @staticmethod
    def build_model(nlp, input_path, model_path):
        """
        Build and save a Source Resolver model.
        
        Args:
            nlp: A spaCy Language object.
            train_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively. 
            model_path: The path (string) to the Citron model directory.           
        """
        
        logger.info("Building Source Resolver model using: %s", input_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
         
        features, labels = SourceResolver._get_features_and_labels(nlp, input_path)
        
        logger.debug("Vectorising training data")
        vectorizer = DictVectorizer()
        train_vectors = vectorizer.fit_transform(features)
        logger.debug("train_vectors.shape: %s", train_vectors.shape)
        
        logger.info("Training Source Resolver model.")
        classifier = LogisticRegression(random_state=0)
        classifier.fit(train_vectors, labels)
        
        model = {}
        model["classifier"] = classifier
        model["vectorizer"] = vectorizer
        model["timestamp"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        filename = os.path.join(model_path, SourceResolver.MODEL_FILENAME)
        logger.info("Saving Source Resolver model: %s", filename)
        
        try:
            with open(filename, "wb") as outfile:
                pickle.dump(model, outfile)
        
        except IOError:
            logger.error("Unable to save Source Resolver model: %s", filename)
    
    
    @staticmethod
    def _get_features_and_labels(nlp, input_path):
        """
        Get the features and labels for the sources in each document in the corpus.
         
        Args:
            input_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        
        Returns:
            A tuple containing:
                a list of feature dicts.
                a list of boolean labels.
        """
        
        features = []
        labels = []
        
        for doc, quotes, _ in DataSource(nlp, input_path):
            sentence_section_labels =  utils.get_sentence_section_labels(doc)
                         
            for quote in quotes:
                for actual_source in quote.sources:
                    cue = quote.cue
                    candidate_sources = utils.get_spans_within_span(quote.sources, cue.sent)                                    
                    
                    # Get features and labels for each candidate source
                    for candidate_source in candidate_sources:  
                        candidate_features = SourceResolver._get_features(candidate_source, cue, sentence_section_labels)
                        
                        if candidate_features is not None:                
                            features.append(candidate_features)
                            label = SourceResolver._get_label(candidate_source, actual_source)
                            labels.append(label)
        
        return features, labels
    
    
    @staticmethod
    def _get_features(source, cue, sentence_section_labels):
        """
        Get the features for a candidate source in relation to a cue.
        
        Args:
            source: A spaCy Span object.
            cue: A spaCy Span object.
            sentence_section_labels: A list containing an integer label each token in the document.
        
        Returns:
            A features dict.
        """
        
        features = {}
        
        distance_from_start_to_cue = min(abs(source.start - cue.start), abs(source.start - cue.end))
        distance_from_end_to_cue   = min(abs(source.end - cue.start), abs(source.end - cue.end))
        features["distance_from_cue"] = min(distance_from_start_to_cue, distance_from_end_to_cue)
        features["is_same_sentence"] = str(utils.is_same_sentence(cue, source))
        features["is_same_comma_span"] = sentence_section_labels[source.start] == sentence_section_labels[cue.start]
        
        return features
    
    
    @staticmethod
    def _get_label(candidate_source, actual_source):
        """
        Get the label for a candidate source. Returns True when the candidate source
        matches the actual source. Otherwise returns False.
        
        Args:
            candidate_source: A spaCy Span object.
            actual_source: A spaCy Span object.
        
        Returns:
            A boolean value.
        
        """
        
        return utils.are_matching_spans(candidate_source, actual_source)
