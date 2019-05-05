#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${loading_housing[raw_data]} -d ${processing[script]} \
        -o ${processing[training]} -o ${processing[test]} \
        -f run/02-processing.dvc \
        -M ${processing[metric]} \
        --overwrite-dvcfile \
        python ${processing[entry_point]} --test_size 0.4 --random_state 42
