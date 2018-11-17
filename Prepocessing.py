import argparse
from datetime import datetime
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import librosa
from librosa.feature import mfcc, melspectrogram
import os
#preprocessing speech to mfcc and spectrogram 
#Note that the resulting features are not of the same size

#change the directory
speech_dir='TEDLIUM_release1/test/sph'
mfcc=[]
spec=[]
for filename in os.listdir(speech_dir):
    print (filename)
    if filename.endswith(".wav"): 
        path_name =(os.path.join(speech_dir, filename))
        y, sr = librosa.load(path_name,sr=16000)  
        speech_features_mfcc = librosa.feature.mfcc(y=y, sr=16000,n_mfcc=20)
        #max length determined from the max length in test dataset
        mfcc_padded = pad_sequences(speech_features_mfcc, maxlen=55376, dtype='float', padding='post',
                                truncating='post')
        speech_features_spec = librosa.feature.melspectrogram(y=y,sr=sr)
        spec_padded = pad_sequences(speech_features_spec, maxlen=55376, dtype='float', padding='post',
                                truncating='post')
        mfcc.append(mfcc_padded)
        spec.append(spec_padded)
        
mfcc = np.array(mfcc)
spec = np.array(spec)

np.save(speech_dir+"/mfcc.npy", mfcc)
np.save(speech_dir+"/spec.npy", spec)

print ("mfcc::", mfcc.shape)
print ("spec::", spec.shape)

