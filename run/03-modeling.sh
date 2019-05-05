#!/usr/bin/env bash

source <(run/configuration-parser.py --ini config.ini)

dvc run -d ${processing[training]} -d ${processing[test]} -d ${modeling[script]} \
        -o ${modeling[model]} \
        -f run/03-modeling.dvc \
        -M ${modeling[metric]} \
        --overwrite-dvcfile \
        python ${modeling[entry_point]}
