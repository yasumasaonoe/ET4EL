#!/bin/zsh
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/cudacache' python -u main.py conll_60k_eval_test -enhanced_mention -model_type ETModel -mode test -reload_model_name et_conll_60k_best -eval_data entity_linking_data/test/test_et4el_conll_60k_1stsent.json -load -eval_batch_size 32 -goal conll_60k
