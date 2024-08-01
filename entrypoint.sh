#!/bin/bash

python -m pip install -r requirements.txt

# if $1 == "baseline", else config_res.json
if [ "$1" = "baseline" ]; then
    config_file="config.json"
else
    config_file="config_res.json"
fi

# if $2 is a path then pass as resume
if [ -f "$2" ]; then
    echo "Resuming training from $2"
    python train.py --config $config_file --resume $2
else
    python train.py --config $config_file
fi
