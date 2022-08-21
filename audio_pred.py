#!/usr/bin/env python
# coding: utf-8

# In[1]:

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

def audio_classification(file, weights):
    model = keras.models.load_model(weights)
    
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    
    # Utility functions for loading audio files and making sure the sample rate is correct.

    @tf.function
    def load_wav_16k_mono(filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
              file_contents,
              desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav
    
    testing_wav_data = load_wav_16k_mono(file)
    scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    result = model(embeddings).numpy()
    m_classes = ['unnatural', 'natural']
    inferred_class = m_classes[result.mean(axis=0).argmax()]
    #print(f'The main sound is: {inferred_class}')
    return inferred_class


# In[ ]:




