#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import base64
from audio_pred import audio_classification
from IPython import display
from IPython.display import Audio
import wave
from pydub import AudioSegment
import tensorflow as tf
#import audio_feature
#from audio_feature.audio_featurizer import audio_process

header = st.container()
dataset = st.container()
background = st.container()
var = st.columns
# with background:
#     @st.cache(allow_output_mutation=True)
#     def get_base64_of_bin_file(bin_file):
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()

#     def set_png_as_page_bg(image_file):
#         bin_str = get_base64_of_bin_file(image_file)
#         page_bg_img = '''
#         <style>
#         body {
#         background-image: url("data:image/png;base64,%s");
#         background-size: cover;
#         }
#         <style/>
#         ''' %bin_str

#         st.markdown(page_bg_img, unsafe_allow_html=True)
#         return

#     set_png_as_page_bg('WildAI.png') 

with header:
    st.title('WildAI : AI-based Early Warning System for Wildlife')
    file = st.file_uploader("Choose a file", type=['wav'])
    #file_var = AudioSegment.from_ogg(file) 
    #st.audio(file)
    #audio = wave.open(file,"r")
    #Audio(file)
    #st.button("Classify")
    #color = st.color_picker('Pick A Color', '#00f900')
    #st.write('The current color is', color)
    #audio = Audio.import(file)
    #with st.form:    
    #    submit = st.form_submit_button(label='Classify')
    #with dataset: 
        #if st.button("Classify"):
        #st.write("Clicked")
   
                  
    if file is not None:
        st.audio(file, format='wav')
        #audio = file.read()
        #audio_file=open(file)
        #audio_bytes= audio_file.read(file)
        st.write("Classifying..")
        #audio = audio_process(file)
        #label='natural'
        file1 = tf.keras.utils.get_file('explosion.wav',
                                                'https://storage.googleapis.com/audioset/yamalyzer/audio/explosion.wav',
                                                cache_dir='./',
                                                cache_subdir='test_data')
        
        label = audio_classification(file1, 'my_model.h5')
        if label == 'natural':
            #st.write("The audio is natural")
            #st.set_page_config(layout="wide") 
            st.markdown(""" <style> .big-font { font-size:50px !important;color:Green;} </style> """, unsafe_allow_html=True) 
            st.markdown('<p class="big-font">The audio is natural, no threats found.</p>', unsafe_allow_html=True)


        else:
            #st.write("The audio has potential threat sounds")
            st.markdown(""" <style> .big-font { font-size:50px !important;color:Red; } </style> """, unsafe_allow_html=True) 
            st.markdown('<p class="big-font">The audio contains unnatural sounds, potential threats found!!</p>', unsafe_allow_html=True)

    


