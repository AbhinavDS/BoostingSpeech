import os
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
#borrowed from github for text: https://github.com/holm-aune-bachelor2018/ctc/blob/master/utils/text_utils.py
char_map_str = """
<EOS> 0
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
<SPACE> 29
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
text_final=[]
sequences_final=[]

for filename in os.listdir(text_dir):
    count=0
    #print filename
    if filename.endswith(".stm"): 
        path_name =(os.path.join(text_dir, filename))
        data = pd.read_csv(path_name, header = None)
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

maxlen = 0
for item in sequences_final:
    if len(item) > maxlen:
        maxlen = len(item)

seq_padded = pad_sequences(sequences_final, maxlen=maxlen, dtype='int32', padding='post',
                                truncating='post', value=0)

np.save(text_dir+"/label.npy", seq_padded)
