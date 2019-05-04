#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${load[raw]} -d ${script[processing]} \
        -o ${process[training]} -o ${process[test]} \
        -f run/02-processing.dvc \
        -M ${process[metric]} \
        --overwrite-dvcfile \
        python ${script[processing]} --test_size 0.4 --random_state 42
