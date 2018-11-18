from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
from librosa.feature import mfcc, melspectrogram

import os
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
text_final=[]
sequences_final=[]

for filename in os.listdir(text_dir):
	count=0
	if filename.endswith(".stm"): 
		text_path_name =(os.path.join(text_dir, filename))
		speech_path_name=(os.path.join(speech_dir, filename[:-4]+".wav"))
		
		if not os.path.exists(speech_path_name):
			continue
		
		print (filename[:-4]+".wav", filename)


		# SPEECH
		y, sr = librosa.load(speech_path_name,sr=16000)  
		speech_features_mfcc = librosa.feature.mfcc(y=y, sr=16000,n_mfcc=20)
		#max length determined from the max length in test dataset
		mfcc_padded = pad_sequences(speech_features_mfcc, maxlen=55376, dtype='float', padding='post',
								truncating='post')
		speech_features_spec = librosa.feature.melspectrogram(y=y,sr=sr)
		spec_padded = pad_sequences(speech_features_spec, maxlen=55376, dtype='float', padding='post',
								truncating='post')
		mfcc.append(mfcc_padded)
		spec.append(spec_padded)

		# TEXT
		data = pd.read_csv(text_path_name, header = None)
		test_text=[]
		row=data.shape[0]
		for x in range(0,row):
			text=data[2][x][8:]
			if text!=' ignore_time_segment_in_scoring':
				test_text.append(text)
		text_final.append(test_text)
		count=count+1
		int_sequence = text_to_int_sequence(test_text)
		sequences_final.append(int_sequence)


mfcc = np.array(mfcc)
spec = np.array(spec)

maxlen = 0
for item in sequences_final:
	if len(item) > maxlen:
		maxlen = len(item)

seq_padded = pad_sequences(sequences_final, maxlen=maxlen, dtype='int32', padding='post',
								truncating='post', value=29)
		

np.save(speech_dir+"/mfcc.npy", mfcc)
np.save(speech_dir+"/spec.npy", spec)
np.save(text_dir+"/label.npy", seq_padded)

print ("mfcc::", mfcc.shape)
print ("spec::", spec.shape)
print ("labels::", seq_padded.shape)
