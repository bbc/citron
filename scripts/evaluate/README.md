<img src="../../citron/public/img/citron_logo.png" alt="Citron logo" align="right">

# Citron Evaluate #

**citron_evaluate.py** can be used to evaluate the Citron quote extraction system, using data in [Citron's Annotation Format](../../docs/data_format.md).

The components are evaluated collectively, rather than individually (as done in the [model building scripts](../train)). The performance of each component is therefore affected by the behaviour of components earlier in the pipeline.

The *exact match metrics* require the predicted and correct spans to be an exact match. The *overlap metrics* are based on the number of predicted tokens that overlap with correct tokens, compared to the total number of predicted and correct tokens.

The *All Quote Spans* results are a collective evaluation of the quote cue, source and contents spans.

## Usage ##
        
    $ export PYTHONPATH=$PYTHONPATH:/path/to/citron
    
    $ python3 citron_evaluate.py
        --model-path    Path to model directory
        --test-path     Path to test data
        -v              Optional: Verbose mode

## Example Output ##

	---- Cue Span exact match metrics ----
	Precision: 0.9334
	Recall:    0.7363
	F1:        0.8232
	
	---- Cue Span overlap metrics ----
	Precision: 0.9646
	Recall:    0.6386
	F1:        0.7684
	
	---- Source Spans exact match metrics ----
	Precision: 0.9293
	Recall:    0.7331
	F1:        0.8197
	
	--- Source Spans overlap metrics ----
	Precision: 0.9739
	Recall:    0.7584
	F1:        0.8527
	
	---- Content Spans exact match metrics ----
	Precision: 0.6726
	Recall:    0.5305
	F1:        0.5932
	
	---- Content Spans overlap metrics ----
	Precision: 0.9322
	Recall:    0.7528
	F1:        0.8329
	
	---- All Quote Spans exact match metrics ----
	Precision: 0.6427
	Recall:    0.5070
	F1:        0.5668
	
	---- All Quote Spans overlap metrics ----
	Precision: 0.7474
	Recall:    0.9377
	F1:        0.8318

Copyright 2021 British Broadcasting Corporation.
