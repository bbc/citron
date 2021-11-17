# Citron Annotation Format #

Citron uses a JSON data format for training and evaluation purposes. Each file represents one document and contains an object with the following properties:

    text                  The text of the document
    quotes                Quote annotations
    
    coreference_groups    Coreference annotations    (optional)

### Quote Annotations ###

The quotes property is an array where each quote is represented by an object with the following properties:

    cue             Cue span
    sources         An array of source spans
    contents        An array of contents spans
    
    coreferences    An array of source coreference spans     (optional)
    confidence      Confidence score for predicted quotes    (optional)
    id              Unique quote identifier                  (optional)

Each span is represented by an object with the following properties:

    start    Start character index
    end      End character index
    text     Span text

Example quote object:

        {
            "cue": {
                "start": 364,
                "end": 368,
                "text": "said"
            },
            "sources": [
                {
                    "start": 350,
                    "end": 363,
                    "text": "Humpty Dumpty"
                }
            ],
            "contents": [
                {
                    "start": 302,
                    "end": 322,
                    "text": "\"When I use a word,\""
                },
                {
                    "start": 369,
                    "end": 423,
                    "text": "\"it means just what I choose it to mean — neither more nor less.\""
                }
            ]
        }

### Coreference Group Annotations ###

The optional coreference_groups property is an array of coreference groups. Each coreference group is an array of spans, defining the members of the group.

Example:

    "coreference_groups": [
        [
            {
                "start": 350,
                "end": 363,
                "text": "Humpty Dumpty"
            },
            {
                "start": 432,
                "end": 434,
                "text": "he"
            }
        ]
    ]

Copyright 2021 British Broadcasting Corporation.
