import argparse
from datetime import datetime
import keras.backend as K
import tensorflow as tf
from keras import models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import librosa
from librosa.feature import mfcc, melspectrogram

#preprocessing speech to mfcc and spectrogram 
#Note that the resulting features are not of the same size

#change the directory
speech_dir='/home/madhumitha/Desktop/Fall2018/DeepLearningSeminar/Project2/TEDLIUM_release1/test/sph'
mfcc=[]
spec=[]
for filename in os.listdir(speech_dir):
    #print filename
    if filename.endswith(".wav"): 
        path_name =(os.path.join(speech_dir, filename))
        y, sr = librosa.load(path_name,sr=16000)  
        speech_features_mfcc = librosa.feature.mfcc(y=y, sr=16000,n_mfcc=20)
        speech_features_spec = librosa.feature.melspectrogram(y=y,sr=sr)
        mfcc.append(speech_features_mfcc)
        spec.append(speech_features_spec)
