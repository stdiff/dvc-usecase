#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${process[training]} -d ${process[test]} -d ${script[modeling]} \
        -o ${model[model]} \
        -f run/03-modeling.dvc \
        -M ${model[metric]} \
        --overwrite-dvcfile \
        python ${script[modeling]}
