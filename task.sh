#!/usr/bin/env bash
# Other variables can also be passed as input now. No need to change file
python3 /u/abhinav/Projects/BoostingSpeech/train.py  --cell LSTM -log logs/lstm.log -train test -dev test
