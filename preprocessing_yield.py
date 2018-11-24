from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram

import os
import operator
import argparse
import librosa
import numpy as np
import pandas as pd


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
				ch = char_map[c]
			int_sequence.append(ch)
	return int_sequence


def data_generator(text_dir='TEDLIUM_release1/test/stm', speech_dir='TEDLIUM_release1/test/sph', batch_size=1, feature='spec', num_features=50, overfit=False, maxlen_mfcc=1000, maxlen_spec=1000, maxlen_seq=800):
	assert feature in ['spec', 'mfcc']
	current_batch = 0
	mfcc=[]
	spec=[]
	cur_sequence=[]
	count = 0
	epoch = -1
	max_files = 2
	all_files =  os.listdir(text_dir)
	all_files.sort()
	while True:
		epoch += 1
		file_counter = 0
		for filename in all_files:
			# if file_counter >= max_files:
			# 	break
			# print (filename)
			file_counter += 1
			if filename.endswith(".stm"): 
				text_path_name =(os.path.join(text_dir, filename))
				speech_path_name=(os.path.join(speech_dir, filename[:-4]+".wav"))
				
				if not os.path.exists(speech_path_name):
					continue		
				# print ("READING", filename[:-4]+".wav", filename)


				# LOAD WHOLE SPEECH
				y, sr = librosa.load(speech_path_name,sr=16000)  
				
				# TEXT
				data = pd.read_csv(text_path_name, header = None)
				test_text=[]
				time_seq = []
				row=data.shape[0]
				for x in range(0,row):
					time=data[0][x].strip().split()[3]
					time_seq.append(float(time))
					text=data[2][x][8:]
					if text ==' ignore_time_segment_in_scoring':
						text = '_'
					text = text_to_int_sequence(text)
					test_text.append(text)
				
				time_seq = list(map(operator.mul, time_seq, [sr]*len(time_seq)))
				time_seq = list(map(int, time_seq))
				time_seq.append(y.shape[0])
				for i in range(1,len(time_seq)-2):
					time1 = time_seq[i]
					time2 = time_seq[i+1]-1
					speech_features_mfcc = librosa.feature.mfcc(y=y[time1:time2], sr=sr, n_mfcc=num_features)
					speech_features_spec = librosa.feature.melspectrogram(y=y[time1:time2],sr=sr)
					mfcc.append(speech_features_mfcc)
					spec.append(speech_features_spec)
					cur_sequence.append(test_text[i])
					current_batch += 1
					count += 1
					if (current_batch >= batch_size):
						current_batch = 0
						data = pad_stuff(mfcc, spec, cur_sequence, maxlen_mfcc, maxlen_spec, maxlen_seq, feature, epoch)
						mfcc=[]
						spec=[]
						cur_sequence=[]
						yield data

					if overfit and count >= 1:
						break
			if overfit and count >= 1:
				count = 0
				break

		# To handle left over files in the batch (e.g. last batch in epoch can have less datapoints than actual batch_size)
		if (not overfit) and (len(mfcc) > 0):
			current_batch = 0
			data = pad_stuff(mfcc, spec, cur_sequence, maxlen_mfcc, maxlen_spec, maxlen_seq, feature, epoch)
			mfcc=[]
			spec=[]
			cur_sequence=[]
			yield data

def pad_stuff(mfcc, spec, seq, maxlen_mfcc, maxlen_spec, maxlen_seq, feature, epoch):	
	input_length_mfcc = []
	input_length_spec = []
	for i in range(len(mfcc)):
		mfcc[i] = pad_sequences(mfcc[i], maxlen=maxlen_mfcc, dtype='float', padding='post', truncating='post')
		spec[i] = pad_sequences(spec[i], maxlen=maxlen_spec, dtype='float', padding='post', truncating='post')
		input_length_mfcc.append(len(mfcc[i]))
		input_length_spec.append(len(spec[i]))
	seq_padded = pad_sequences(seq, maxlen=maxlen_seq, dtype='int32', padding='post', truncating='post', value=29)
	#seq_padded = seq
	
	# Get sequence length
	seq_len = np.zeros((len(seq_padded)), dtype=int)

	for i in range(len(seq_padded)):
		seq_len[i] = len(seq_padded[i])

	mfcc = np.array(mfcc)
	mfcc = np.transpose(mfcc, axes=[0, 2, 1])
	spec = np.array(spec)
	spec = np.transpose(spec, axes=[0, 2, 1])
	# print ("mfcc::", mfcc.shape)
	# print ("spec::", spec.shape)
	# print ("labels::", seq_padded.shape)
	SOS = len(char_map)
	unpadded_data_y = seq.copy()
	# for i in range(len(seq_padded)):
	# 	unpadded_data_y[i].insert(0, SOS)

	if feature =='spec':
		return (spec, seq_padded, seq_len, epoch, input_length_spec, unpadded_data_y)
	else:
		return (mfcc, seq_padded, seq_len, epoch, input_length_mfcc, unpadded_data_y)
