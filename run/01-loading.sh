#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -o ${data[raw]} \
        --overwrite-dvcfile \
        python ${script[loading]}
