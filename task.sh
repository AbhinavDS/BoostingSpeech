#!/usr/bin/env bash
# Other variables can also be passed as input now. No need to change file
python3 /u/abhinav/Projects/BoostingSpeech/train.py  --cell LSTM -log logs/lstm.log -train test -dev test

#Boosting - PLEASE READ
# REMEMEBER TO CHANGE BN, also note that not run multiple process or keep verifying your META file is correct before running new boost
# first time copy ckpt files and run so that weight files 
# python3 boosting_train.py  --cell GRU -log logs/gru_boost.log -train train -dev dev -ss 11229 -s boost -bn 1 -bd boosting_models/1/
# OPTIONAL -ckpt boosting_models/1/model_boost_LSTM.ckpt 
