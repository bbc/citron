# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

"""
This module provides the primary functions of the Citron quote extraction 
and attribution system and a web server supporting a REST API.
"""

import atexit
import json
import os

import cherrypy

from .data import Quote
from .cue import CueClassifier
from .content import ContentClassifier
from .content import ContentResolver
from .source import SourceResolver
from .source import SourceClassifier
from .coreference import CoreferenceResolver
from . import utils
from . import metrics
from .logger import logger

APPLICATION_NAME = "citron-extractor"


class Citron():
    """
    Class providing methods to extract quotes from documents.
    """
    
    def __init__(self, model_path, nlp=None):
        """
        Constructor.
        
        Args:
            model_path: The path (string) to the Citron model directory.
            nlp: A spaCy Language object, or None.
        """
        
        if nlp is None:
            self.nlp = utils.get_parser()
        else:
            self.nlp = nlp
        
        logger.info("Loading Citron model: %s", model_path)
        self.cue_classifier = CueClassifier(model_path)
        self.content_classifier = ContentClassifier(model_path)
        self.source_classifier = SourceClassifier(model_path)
        self.content_resolver = ContentResolver(model_path)
        self.source_resolver = SourceResolver(model_path)
        self.coreference_resolver = CoreferenceResolver(model_path)
        
        self.source = {
            "application": APPLICATION_NAME,
            "model:": self.cue_classifier.model["timestamp"] 
        }
    
    
    def extract(self, text, resolve_coreferences=True):
        """
        Extract quotes from the supplied text.
        
        Args:
            text: The text (string)
            resolve_coreferences: A boolean flag indicating whether to resolve coreferences.
            
        Returns:
            A JSON serialisable object containing the extracted quotes.
        """
        
        doc = self.nlp(text)
        quotes = self.get_quotes(doc, resolve_coreferences)
        quotes_json = []
        
        for quote in quotes:
            quotes_json.append(quote.to_json())
        
        return { "quotes": quotes_json, "source": self.source }
    
    
    def get_quotes(self, doc, resolve_coreferences=True):
        """
        Extract quotes from a spaCy Doc.
        
        Args:
            doc: A spaCy Doc object.
            resolve_coreferences: A boolean flag indicating whether to resolve coreferences.
        
        Returns:
            A list of citron.data.Quote objects.
        """
        
        quotes = []
        sentence_section_labels =  utils.get_sentence_section_labels(doc)
        
        # First find quote-cues.
        cue_spans, cue_labels = self.cue_classifier.predict_cues_and_labels(doc)
        
        if len(cue_spans) > 0:
            # Identify source and content spans.
            content_spans, content_labels = self.content_classifier.predict_contents_and_labels(doc, cue_labels)
            source_spans = self.source_classifier.predict_sources_and_labels(doc, cue_labels, content_labels)[0]
            
            if len(content_spans) == 0:
                return []
            
            if len(source_spans) == 0:
                return []
            
            # Identify the quote-cue associated with each source and content span.
            cue_to_contents_map = self.content_resolver.resolve_contents(content_spans, cue_spans)
            cue_to_sources_map  = self.source_resolver.resolve_sources(source_spans, cue_spans, sentence_section_labels)
            
            # Join source and content spans which share the same quote-cue.
            for cue in cue_spans:
                key = (cue.start, cue.end)
                
                if key in cue_to_contents_map and key in cue_to_sources_map:
                    contents = cue_to_contents_map[key]
                    source = cue_to_sources_map[key]
                    confidence = source._.probability
                    
                    for content in contents:
                        confidence = confidence * content._.probability
                    
                    quote = Quote(cue, [source], contents, confidence=confidence)
                    quotes.append(quote)
        
        if resolve_coreferences and len(quotes) > 0:
            self.coreference_resolver.resolve_document(doc, quotes, source_spans, content_spans, content_labels)
        
        return quotes
    
    
    def evaluate(self, test_path):
        """
        Evaluate Citron using the supplied test data.
        
        Args:
            test_path: The path (string) to a file or directory of Citron format JSON data files.
                Directories will be explored recursively.
        """
        
        metrics.evaluate(self, test_path)


class CitronWeb():
    """
    Class providing a web server which supports a REST API and a demonstration.
    """
    
    def __init__(self, citron):
        """
        Constructor.
        
        Args:
            citron: a citron.citron.Citron object.
        """
        
        logger.info("Initialising Citron API...")
        self._citron = citron
    
    
    def start(self, port, logfile=None):
        """
        Start the web server.
        
        Args:
            port: the server port number (integer)
            logfile: The path (string) to the log file, or None.
        """
        
        cherrypy.engine.autoreload.unsubscribe()
        
        options = {
          "server.socket_host": "0.0.0.0",
          "server.socket_port": port
        }
        
        if not logfile:
            log_options = { "log.screen": True }
        else:
            log_options = { "log.screen": False, "log.access_file": logfile, "log.error_file": logfile }
        
        options.update(log_options)
        cherrypy.config.update(options)
        
        if cherrypy.__version__.startswith("3.0") and cherrypy.engine.state == 0:
            cherrypy.engine.start(blocking=False)
            atexit.register(cherrypy.engine.stop)
        
        config = {
            "/": {
                "request.query_string_encoding": "utf-8",
                "tools.staticdir.on": True,
                "tools.staticdir.dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "public")),
                "tools.staticdir.index": "index.html"
            },
        }
        cherrypy.quickstart(self, "/", config = config)
        
    
    @cherrypy.expose
    def quotes(self, text=None):
        """
        The quote method end point.
        
        Args:
            text: the text (string) from which to extract quotes.
        """
        
        if text is None:
            cherrypy.response.headers["Content-Type"] = "application/json"
            cherrypy.response.status = 400
            return json.dumps({"error": "A text parameter must be provided."}).encode("utf8")
        
        try:
            results = self._citron.extract(text)
        
        except ValueError as err:
            cherrypy.response.headers["Content-Type"] = "application/json"
            cherrypy.response.status = 500
            return json.dumps({ "error": err.message }).encode("utf8")
        
        cherrypy.response.headers["Content-Type"] = "application/json; charset=utf-8"
        return json.dumps(results, ensure_ascii=False).encode("utf8")
