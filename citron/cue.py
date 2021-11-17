# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides methods to identify quote cues.
"""

import xml.etree.ElementTree as ET
import datetime
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from .data import DataSource
from .logger import logger
from . import utils
from . import metrics


class CueClassifier():
    """
    Classifier which identifies quote cues e.g. "said". Based on the paper:
    
    "Automatically Detecting and Attributing Indirect Quotations"
    Silvia Pareti, Timothy O'Keefe, Ioannis Konstas, James R Curran, Irena Koprinska
    Proceedings of the Conference on Empirical Methods in Natural Language Processing
    (EMNLP), Seattle, U.S.
    """
    
    MODEL_FILENAME = "cue-classifier.pickle"
    MAX_OFFSET = 5
    
    
    def __init__(self, model_path):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model.
        
        """
        
        filename = os.path.join(model_path, self.MODEL_FILENAME)
        logger.debug("Loading Cue Classifier model: %s", filename)
        
        with open(filename, "rb") as infile:
            self.model = pickle.load(infile)
        
        CueClassifier.verbnet = self.model["verbnet"]
    
    
    def predict_cues_and_labels(self, doc):
        """
        Predict cue spans and labels for a document. 
        
        Args:
            doc: A spaCy Doc object.
        
        Returns:
            A tuple containing:
                cue_spans: A list of spaCy Span objects.
                cue_labels: A list containing an IOB label for each token in the document.
        """
        
        inside_quotation_marks_labels = utils.get_inside_quotation_marks_labels(doc)
        cue_labels = ["O"] * len(doc)
        previous = "O"
        
        for sentence in doc.sents:
            for token in sentence:
                # Avoid cues inside quotation marks
                if inside_quotation_marks_labels[token.i] == 1:
                    continue
                
                features = self._get_features(token)
                test_vectors = self.model["vectorizer"].transform([features])
                predictions = self.model["classifier"].predict(test_vectors)
                
                if predictions[0] == 1:
                    if previous == "O":
                        iob_label = "B"
                    else:
                        iob_label = "I"
                else:
                    iob_label = "O"
                
                cue_labels[token.i] = iob_label
                previous = iob_label
        
        cue_spans = utils.get_spans(doc, cue_labels)
        return cue_spans, cue_labels
    
    
    def evaluate(self, nlp, test_path):
        """
        Evaluate the Cue Classifier.
        
        Args:
            nlp: A spaCy Language object.
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        logger.info("Evaluating Cue Classifier model using: %s", test_path)
        features, labels = self._get_features_and_labels(nlp, test_path)
        logger.debug("Test labels: %s", len(labels))
        
        test_vectors = self.model["vectorizer"].transform(features)
        predictions = self.model["classifier"].predict(test_vectors)
        
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(0, len(labels)):
            if predictions[i] == 1:
                if labels[i] == 1:
                    tp += 1
                
                else:
                    fp += 1
            
            else:
                if labels[i] == 1:
                    fn += 1
        
        print()
        print("---- Metrics ----")
        exact_scores = metrics.get_exact_scores(tp, fp, fn)
        metrics.print_metrics(*exact_scores)
    
    
    @staticmethod
    def build_model(nlp, train_path, model_path, verbnet_path):
        """
        Build and save a Cue Classifier model.
        
        Args:
            nlp: A spaCy Language object.
            train_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
            model_path: The path (string) to the Citron model directory.
            verbnet_path: The path (string) to the VerbNet directory.
        """
        
        logger.info("Building Cue Classifier model using: %s", train_path)
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        CueClassifier.verbnet = VerbNet(verbnet_path)         
        features, labels = CueClassifier._get_features_and_labels(nlp, train_path)
        
        logger.info("Vectorising training data")
        vectorizer = DictVectorizer()
        train_vectors = vectorizer.fit_transform(features)
        logger.debug("train_vectors.shape: %s", train_vectors.shape)
        
        logger.info("Training Logistic Regression")
        classifier = LogisticRegression(solver="liblinear")
        classifier.fit(train_vectors, labels)
        
        model = {}
        model["classifier"] = classifier
        model["vectorizer"] = vectorizer
        model["verbnet"] = CueClassifier.verbnet 
        model["timestamp"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        filename = os.path.join(model_path, CueClassifier.MODEL_FILENAME)
        logger.info("Saving Cue Classifier model: %s", filename)
        
        try:
            with open(filename, "wb") as outfile:
                pickle.dump(model, outfile)
        
        except IOError:
            logger.error("Unable to save model: %s", filename)
    
    
    @staticmethod
    def _get_features_and_labels(nlp, input_path):
        """
        Get features and labels for each token in a corpus of documents.
        
        Args:
            input_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        
        Returns:
            A tuple containing:
                features: A list of feature dicts.
                labels: A list of binary labels.
        """
        
        features = []
        labels = []
        
        for doc, quotes, _ in DataSource(nlp, input_path):
            # Add features for each token, avoiding those inside quotation marks.         
            inside_quotation_marks_labels = utils.get_inside_quotation_marks_labels(doc)
            actual_cue_labels = utils.get_cue_iob_labels(doc, quotes)
            
            for sentence in doc.sents:            
                for token in sentence:
                    if inside_quotation_marks_labels[token.i] == 1:
                        continue
                    
                    token_features = CueClassifier._get_features(token)
                    token_label = actual_cue_labels[token.i]
                    features.append(token_features)
                    
                    if token_label == "O":
                        labels.append(0)
                    else:
                        labels.append(1)
        
        return features, labels
    
    
    @staticmethod
    def _get_features(token):
        """
        Get the features for a token.
        
        Args:
            token: A spaCy Token object. 
        
        Returns:
            A features dict.
        """
        
        index = token.i
        doc = token.doc
        features = {}
        features["text"] = token.text
        features["lemma"] = token.lemma_
        features["tag"] = token.tag_
        features["ent_iob"] = token.ent_iob_
        features["dep"] = token.dep_
        features["head_tag"] = token.head.tag_
        
        verbnet_class = CueClassifier.verbnet.get_class(token)
        
        if verbnet_class is not None:
            features["vnclass"] = verbnet_class
        
        features["depth"] = utils.get_dependency_depth(token)
        features["children"] = len(list(token.children))
        
        # Neighbour related features
        for n in range(0, CueClassifier.MAX_OFFSET):
            if index - n - 1 >= 0:
                features["previousText" + str(n)] = doc[index - n - 1].text
        
        for n in range(0, CueClassifier.MAX_OFFSET):
            if index + n + 1 < len(doc):
                features["nextText" + str(n)] = doc[index + n + 1].text
        
        return features


class VerbNet():
    """
    Class which provides the VerbNet classes of verbs.
    """
    
    def __init__(self, path):
        """
        Constructor.
        
        Args:
            path: The path (string) to the VerbNet directory.
        """
        
        logger.info("Loading VerbNet from: %s", path)
        self._classes = {}
        
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                file_path = os.path.join(path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()
                self._process_element(root)
        
        logger.debug("Verbnet classes: %s", len(self._classes))
    
    
    def get_class(self, token):
        """
        Get the VerbNet class of a token, or None if inapplicable or unknown.
        
        Args:
            token: A spaCy Token object.
        
        Returns:
            The VerbNet class (string) or None.
        """
        
        return self._classes.get(token.lemma_)
    
        
    def _process_element(self, element):
        """
        Process an xml.etree.ElementTree.Element
        
        Args:
            element: an Element object.
        """
        
        verbnet_class_id = element.attrib["ID"]
        
        for child in element.find("MEMBERS"):
            name = child.attrib["name"]
            self._classes[name] = verbnet_class_id
        
        for subclass in element.find("SUBCLASSES"):
            self._process_element(subclass)
