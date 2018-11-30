from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram

import os
import operator
import argparse
import librosa
import numpy as np
import pandas as pd
import pickle

#preprocessing speech to mfcc and spectrogram 
#Note that the resulting features are not of the same size

#borrowed from github for text: https://github.com/holm-aune-bachelor2018/ctc/blob/master/utils/text_utils.py
char_map_str = """
<SPACE> 0
a 1
b 2
c 3
d 4
e 5
f 6
g 7
h 8
i 9
j 10
k 11
l 12
m 13
n 14
o 15
p 16
q 17
r 18
s 19
t 20
u 21
v 22
w 23
x 24
y 25
z 26
' 27
_ 28
<EOS> 29
"""

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
	ch, index = line.split()
	char_map[ch] = int(index)
	index_map[int(index)] = ch
index_map[0] = ' '

# character_map = np.array([char_map.items()], dtype=[('num', 'U10'), ('char', 'U10')])

def text_to_int_sequence(text):
	""" Use a character map and convert text to an integer sequence """
	int_sequence = []
	for sent in text:
		for c in sent:
			if c == ' ':
				ch = char_map['<SPACE>']
			else:
				ch = char_map.get(c, char_map['<SPACE>'])
			int_sequence.append(ch)
	return int_sequence


def data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir="", batch_size=1, feature='spec', num_features=50, overfit=False, maxlen_mfcc=500, maxlen_spec=500, maxlen_seq=300):
	pickle_file = text_dir+'/../data.pickle'
	data = pickle.load( open( pickle_file, "rb" ))
	(spec, seq_padded, seq_len, epoch, input_length_spec, unpadded_data_y) = data
	total_size = len(spec)
	# print (spec.shape, len(seq_padded), len(seq_len), epoch, len(input_length_spec), len(unpadded_data_y))
	assert feature in ['spec', 'mfcc']
	epoch = 0
	while True:
		for i in range(0,total_size,batch_size):
			batch_data = (spec[i:i+batch_size], seq_padded[i:i+batch_size], seq_len[i:i+batch_size], epoch, input_length_spec[i:i+batch_size], unpadded_data_y[i:i+batch_size])
			# (batch_spec, batch_seq_padded, batch_seq_len, batch_epoch, batch_input_length_spec, batch_unpadded_data_y) = batch_data
			# print (seq_len)
			# print (batch_seq_len)
			# print (batch_spec.shape, len(batch_seq_padded), len(batch_seq_len), batch_epoch)#, len(batch_input_length_spec), len(batch_unpadded_data_y))
			yield batch_data
		epoch += 1