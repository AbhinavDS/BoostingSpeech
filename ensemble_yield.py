from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram

import os
import operator
import argparse
import librosa
import numpy as np
import pandas as pd



def ensemble_data_generator(logits_dirs, batch_size=1):
	assert len(logits_dirs) > 0
	logits_dirs.sort()
	logit_filenames = []
	for i in range(len(os.listdir(logits_dirs[0]))):
		logit_filenames.append("logits_%i.npy"%i)
	for logits_dir in logits_dirs:
		for logit_file in os.listdir(logits_dir):
			assert logit_file in logit_filenames
	
	for logit_file in logit_filenames:
		logits = None
		for logits_dir in logits_dirs:
			logit_path = os.path.join(logits_dir,logit_file)
			n_array = np.load(logit_path)
			n_array = np.expand_dims(n_array,0)
			if logits is None:
				logits = n_array
			else:
				logits = np.concatenate((logits,n_array), axis=0)
		yield logits
