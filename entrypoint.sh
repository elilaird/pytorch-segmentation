#!/bin/bash

# if $1 == "baseline", else config_res.json
if [ "$1" = "baseline" ]; then
    config_file="config.json"
else
    config_file="config_res.json"
fi

python -m pip install -r requirements.txt
python train.py --config $config_file