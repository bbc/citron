# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides methods to identify quote contents and the associated cue.
"""

from collections import defaultdict
import datetime
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import pycrfsuite

from .data import DataSource 
from . import utils
from . import metrics
from .logger import logger


class ContentClassifier():
    """
    Classifier which identifies quote content. A document-based approach is
    adopted as quote contents may cross sentence boundaries.
    
    Based on the paper:
    "Automatically Detecting and Attributing Indirect Quotations"
    Silvia Pareti, Timothy O'Keefe, Ioannis Konstas, James R Curran, Irena Koprinska
    Proceedings of the Conference on Empirical Methods in Natural Language Processing
    (EMNLP), Seattle, U.S.
    """
    
    MODEL_FILENAME = "content-classifier.crfsuite"   
    MAX_OFFSET = 5
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Content Classifier model: %s", filename)
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(filename)
    
    
    def predict_contents_and_labels(self, doc, cue_labels):
        """
        Predict content spans and labels for a document. 
        
        Args:
            doc: A spaCy Doc object.
            cue_labels: A list containing an IOB label for each token in the document.
        
        Returns:
            A tuple containing:
                content_spans: A list of spaCy Span objects.
                content_labels: A list containing an IOB label for each token in the document.
        """
        
        features = self._get_features_and_labels(doc, cue_labels)[0]
        predicted_content_labels = self._tagger.tag(features)
        content_labels = utils.conform_labels(predicted_content_labels)
        content_spans = utils.get_spans(doc, content_labels)
        return content_spans, content_labels
    
    
    def evaluate(self, nlp, test_path):
        """
        Evaluate the Content Classifier.
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron annotation format JSON files.
                Directories will be explored recursively.
        """
        
        logger.info("Evaluating Content Classifier model using: %s", test_path)
        
        exact_tp = 0
        exact_fp = 0
        exact_fn = 0
        
        sum_of_overlaps = 0
        sum_of_predicted_lengths = 0
        sum_of_actual_lengths = 0
        
        for doc, quotes, _ in DataSource(nlp, test_path):
            source_count = 0
            content_count = 0
            
            for quote in quotes:
                source_count += len(quote.sources)
                content_count += len(quote.contents)
            
            cue_labels = utils.get_cue_iob_labels(doc, quotes)
            features, actual_content_labels = self._get_features_and_labels(doc, cue_labels, quotes)
            actual_content_labels = utils.conform_labels(actual_content_labels)
            actual_contents = utils.get_spans(doc, actual_content_labels)
            
            predicted_content_labels = self._tagger.tag(features)
            predicted_content_labels = utils.conform_labels(predicted_content_labels)
            predicted_contents = utils.get_spans(doc, predicted_content_labels)
            
            # Exact metrics
            tp, fp, fn = metrics.get_span_exact_match_metrics(actual_contents, predicted_contents)
            exact_tp += tp
            exact_fp += fp
            exact_fn += fn
            
            scores = metrics.get_span_overlap_counts_and_lengths(actual_contents, predicted_contents)
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
        Build and save a Content Classifier model.
        
        Args:
            nlp: A spaCy Language object.
            train_path: The path (string) to a file or directory of Citron format JSON data files.        
                Directories will be explored recursively.
            model_path: The path (string) to the Citron model directory.         
        """
        
        logger.info("Building Content Classifier model using: %s", train_path)
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        trainer = pycrfsuite.Trainer(verbose=False)
        
        for doc, quotes, _ in DataSource(nlp, train_path):
            cue_labels = utils.get_cue_iob_labels(doc, quotes)
            features, labels = ContentClassifier._get_features_and_labels(doc, cue_labels, quotes)
            trainer.append(features, labels)
        
        trainer.set_params({
            "c1": 1.0,
            "c2": 1e-3,
            "max_iterations": 100,
            "feature.possible_transitions": True
        })
        
        logger.info("Training Content Classifier model")
        filename = os.path.join(model_path, ContentClassifier.MODEL_FILENAME)
        trainer.train(filename)
        logger.info("Training complete - last_iteration: %s", trainer.logparser.last_iteration)
    
    
    @staticmethod
    def _get_features_and_labels(doc, cue_labels, quotes=None):
        """
        Get features and labels for each token in a document.
        
        Args:
            doc: A spaCy Doc object.
            cue_labels: A list containing an IOB label for each token in the document.
            quotes: A list of citron.data.Quote objects or None.
        
        Returns:
            A tuple containing:
                doc_features: A list of feature dicts.
                doc_labels: A list of binary labels.
        """
        
        doc_features = []
        doc_labels = []
        in_quotes = False
        
        if quotes is not None:
            # Testing and evaluation
            cue_labels = utils.get_cue_iob_labels(doc, quotes)
            content_labels = utils.get_content_iob_labels(doc, quotes)
        
        else:
            # Prediction
            content_labels = None
            
        # Create features
        for sentence in doc.sents:
            # Get sentence features
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
                
                # Neighbour related features
                for n in range(0, ContentClassifier.MAX_OFFSET):
                    if index - n - 1 >= 0:
                        word_features.append("previous" + str(n) + "=" + doc[index - n - 1].text)
                        
                for n in range(0, ContentClassifier.MAX_OFFSET):
                    if index + n + 1 < len(doc):
                        word_features.append("next" + str(n) + "=" + doc[index + n + 1].text)
                        
                # Environment features
                if in_quotes:
                    word_features.append("in_quotes")
                
                # Tree features
                parent = token
                depth = 0
                ancestor_is_cue = False
                
                while parent != parent.head:
                    if cue_labels[parent.i] != "O":
                        ancestor_is_cue = True
                        
                    depth += 1
                    parent = parent.head
                    
                word_features.append("depth=" + str(depth))
                word_features.append("dep=" + token.head.dep_)           
                word_features.append("ancestorIsCue=" + str(ancestor_is_cue))
                
                if ancestor_is_cue and token.left_edge == token:
                    word_features.append("leftMost")
                    
                if index == 0:
                    word_features.append("followsCue=False")
                
                else:
                    previous = doc[index - 1]
                    followsCue = cue_labels[previous.i] != "O"
                    word_features.append("followsCue=" + str(followsCue))
                
                precedesComma = index < len(doc) - 1 and doc[index + 1].tag_ == ","
                word_features.append("precedesComma=" + str(precedesComma))
                
                # Sentence-based features
                word_features.append("sentenceHasCue=" + str(sentence_has_cue))
                word_features.append("sentenceWord=" + str(index))
                word_features.append("sentenceLength=" + str(len(sentence)))  
                
                doc_features.append(word_features)
                
                if content_labels is not None:
                    doc_labels.append(content_labels[index])
                    
        return doc_features, doc_labels


class ContentResolver():
    """
    Class which resolves the relationship between quote content and cues.
    """
    
    MODEL_FILENAME = "content-resolver.crfsuite"
    PROBABILITY_THRESHOLD = 0.3
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Content Resolver model: %s", filename)
        
        with open(filename, "rb") as infile:
            self._model = pickle.load(infile)
    
    
    def resolve_contents(self, contents, cues):
        """
        Resolve which content spans are associated with each quote cue.
        All spans are spaCy Span objects.
        
        Args:
            contents: A list of content spans.
            cues: A list of cue spans.
        
        Returns:
            A dict mapping each cue to a list of content spans. The cue is represented by a 
            tuple containing the start and end index.
        """
        
        quote_cue_to_contents_map = defaultdict(list)
        
        for content in contents:  
            predicted_cue, probability = self.predict_cue(content, cues)
            
            if predicted_cue is not None:
                key = (predicted_cue.start, predicted_cue.end)
                content._.probability = probability
                quote_cue_to_contents_map[key].append(content)
        
        return quote_cue_to_contents_map
    
    
    def predict_cue(self, content, cues):
        """
        Predict the cue associated with quote content.
        
        Args:
            content: A spaCy Span object.
            cues: A list of spaCy Span objects.
        
        Returns:
            A tuple containing:
                predicted_cue: A spaCy Span object.
                probability: A float value between zero and one.
        """
        
        features = []
        
        for candidate_cue in cues:  
            candidate_features = self._get_features(content, candidate_cue)
            features.append(candidate_features)
        
        test_vectors = self._model["vectorizer"].transform(features)
        predicted_probabilities = self._model["classifier"].predict_proba(test_vectors)        
        predicted_index, probability = utils.get_index_of_max(predicted_probabilities)
        
        if probability >= self.PROBABILITY_THRESHOLD:
            predicted_cue = cues[predicted_index]
        
        else:
            predicted_cue = None
        
        return predicted_cue, probability
    
    
    def evaluate(self, nlp, test_path):
        """
        Evaluate the Content Resolver.
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        logger.info("Evaluating content resolver model using: %s", test_path)
        
        tp = 0
        fp = 0
        fn = 0
        
        for _, quotes, _ in DataSource(nlp, test_path):
            candidate_cues = utils.get_cues(quotes)
            
            for quote in quotes:
                for content in quote.contents:
                    predicted_cue = self.predict_cue(content, candidate_cues)[0]
                    
                    if predicted_cue is None:
                        fn += 1
                    else:
                        if utils.are_matching_spans(predicted_cue, quote.cue):
                            tp += 1
                        
                        else:
                            fp += 1
                            fn += 1
        
        print("--------  Metrics  --------")
        exact_scores = metrics.get_exact_scores(tp, fp, fn)
        metrics.print_metrics(*exact_scores)
    
    
    @staticmethod
    def build_model(nlp, train_path, model_path):
        """
        Build and save a Content Resolver model.
        
        Args:
            nlp: A spaCy Language object.
            train_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
            model_path: The path (string) to the Citron model directory.          
        """
        
        logger.info("Building Content Resolver model using: %s", train_path)
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        features, labels = ContentResolver._get_features_and_labels(nlp, train_path)
        
        logger.info("Vectorising training data")
        vectorizer = DictVectorizer()
        train_vectors = vectorizer.fit_transform(features)
        logger.debug("trainVectors.shape: %s", train_vectors.shape)
        
        logger.info("Training Content Resolver model.")
        classifier = LogisticRegression(random_state=0)
        classifier.fit(train_vectors, labels)
        
        model = {}
        model["classifier"] = classifier
        model["vectorizer"] = vectorizer
        model["timestamp"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        filename = os.path.join(model_path, ContentResolver.MODEL_FILENAME)
        logger.info("Saving Content Resolver model: %s", filename)
        
        try:
            with open(filename, "wb") as outfile:
                pickle.dump(model, outfile)
        
        except IOError:
            logger.error("Unable to save Content Resolver model: %s", filename)
    
    
    @staticmethod
    def _get_features_and_labels(nlp, input_path):
        """
        Get the features and labels for the cues in each document in the corpus.
         
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
        
        for _, quotes, _ in DataSource(nlp, input_path):                        
            for quote in quotes:
                actual_cue = quote.cue
                
                for content in quote.contents:                
                    # Get features and labels for each candidate cue        
                    for candidate_quote in quotes:
                        candidate_cue = candidate_quote.cue
                        candidate_features = ContentResolver._get_features(content, candidate_cue)                        
                        features.append(candidate_features)
                        label = ContentResolver._get_label(candidate_cue, actual_cue)
                        labels.append(label)
        
        return features, labels
    
    
    @staticmethod
    def _get_features(content, cue):
        """
        Get the features for a cue in relation to a content span.
        
        Args:
            content: A spaCy Span object.
            cue: A spaCy Span object.
        
        Returns:
            A features dict.
        """
        
        features = {}
        distance_from_start_to_cue = min(abs(content.start - cue.start), abs(content.start - cue.end))
        distance_from_end_to_cue = min(abs(content.end - cue.start), abs(content.end - cue.end))
        features["distanceFromCue"] = min(distance_from_start_to_cue, distance_from_end_to_cue)
        features["isSameSentence"] = str(utils.is_same_sentence(cue, content))
        features["CueIsAncestor"] = str(utils.is_ancestor_of(content, cue))    
        return features
    
    
    @staticmethod
    def _get_label(candidate_cue, actual_cue):
        """
        Get the label for a candidate cue. Returns True when the candidate cue
        matches the actual cue. Otherwise returns False.
        
        Args:
            candidate_cue: A spaCy Span object.
            actual_cue: A spaCy Span object.
        
        Returns:
            A boolean value.
        """
        
        return utils.are_matching_spans(candidate_cue, actual_cue)
