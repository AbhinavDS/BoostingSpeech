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
	
#use text_to_int_sequence() to generate numbers:

text_dir='TEDLIUM_release1/test/stm'
speech_dir='TEDLIUM_release1/test/sph'
mfcc=[]
spec=[]
sequences_final=[]
mfcc_feat_len = 50

for filename in os.listdir(text_dir):
	if filename.endswith(".stm"): 
		text_path_name =(os.path.join(text_dir, filename))
		speech_path_name=(os.path.join(speech_dir, filename[:-4]+".wav"))
		
		if not os.path.exists(speech_path_name):
			continue
		
		print (filename[:-4]+".wav", filename)


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
			if text!=' ignore_time_segment_in_scoring':
				text = text_to_int_sequence(text)
				test_text.append(text)
			else:
				test_text.append([])
		sequences_final.extend(test_text)

		time_seq = list(map(operator.mul, time_seq, [sr]*len(time_seq)))
		time_seq = list(map(int, time_seq))
		time_seq.append(y.shape[0])
		for i in range(len(time_seq)-1):
			time1 = time_seq[i]
			time2 = time_seq[i+1]-1
			speech_features_mfcc = librosa.feature.mfcc(y=y[time1:time2], sr=sr, n_mfcc=mfcc_feat_len)
			speech_features_spec = librosa.feature.melspectrogram(y=y[time1:time2],sr=sr)
			mfcc.append(speech_features_mfcc)
			spec.append(speech_features_spec)

maxlen_mfcc = 15000
maxlen_spec = 15000
maxlen_seq = 800

for i in range(len(mfcc)):
	if len(mfcc[i][0]) > maxlen_mfcc:
		maxlen_mfcc = len(mfcc[i][0])
	if len(spec[i][0]) > maxlen_spec:
		maxlen_spec = len(spec[i][0])
	if len(sequences_final[i]) > maxlen_seq:
		maxlen_seq = len(sequences_final[i])

print (maxlen_mfcc, maxlen_spec, maxlen_seq)
for i in range(len(mfcc)):
	mfcc[i] = pad_sequences(mfcc[i], maxlen=maxlen_mfcc, dtype='float', padding='post', truncating='post')
	spec[i] = pad_sequences(spec[i], maxlen=maxlen_spec, dtype='float', padding='post', truncating='post')
seq_padded = pad_sequences(sequences_final, maxlen=maxlen_seq, dtype='int32', padding='post', truncating='post', value=29)

mfcc = mfcc[:10]
spec = spec[:10]
seq_padded = seq_padded[:10]

mfcc = np.array(mfcc)
spec = np.array(spec)
		
np.save(speech_dir+"/mfcc.npy", mfcc)
np.save(speech_dir+"/spec.npy", spec)
np.save(text_dir+"/label.npy", seq_padded)

print ("mfcc::", mfcc.shape)
print ("spec::", spec.shape)
print ("labels::", seq_padded.shape)
