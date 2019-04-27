#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${data[raw]} -d ${script[processing]} \
        -o ${data[training]} -o ${data[test]} \
        -f run/02-processing.dvc \
        --overwrite-dvcfile \
        python ${script[processing]} --test_size 0.4 --random_state 42
