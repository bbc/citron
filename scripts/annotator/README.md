<img src="../../citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# Citron Annotator #

Citron Annotator is a GUI programme that can be used to create Citron format annotation data for text files.

## Running ##
        
    $ export PYTHONPATH=$PYTHONPATH:/path/to/citron
    
    $ python3 citron_annotator.py <input directory> <output directory> [<citron model directory>]

## User Guide ##

Click **Open File** and select a text file.
- If a matching annotation file exists in the output directory then the existing annotations will be loaded.
- Otherwise, if a Citron model has been specified there will be an option to predict quote-cues. This automatically annotates cues in the text.

The following functions are then available:

- To annotate a quote, highlight the cue span and click **Select or Add Cue**. You can now select, add, amend or remove the other spans (source, content and coreference).
- To add a span, highlight the range in the text and click the appropriate **Add** button.
- To amend a span, highlight the new range in the text and click **Amend Span**.
- To remove a span, highlight any part of the span and click **Remove Span**. 
- To save the annotations click **Save File**.

Note that annotations can be reloaded and amended by re-running the programme.

Copyright 2021 British Broadcasting Corporation.
