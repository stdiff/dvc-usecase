#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${data[training]} -d ${data[test]} -d ${script[processing]} \
        -o ${model[model]} \
        -f ${model[metric]} \
        --overwrite-dvcfile \
        python ${script[modeling]}
