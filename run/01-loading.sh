#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${script[loading]} \
        -o ${load[raw]} \
        -f run/01-loading.dvc \
        -M ${load[metric]} \
        --overwrite-dvcfile \
        python ${script[loading]} --round ${general[round]}
