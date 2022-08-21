# Audio-classification
This project aims at classifying audio files to detect unnatural sounds from the natural sounds found in forests to classify/identify illegal deforestation or poaching

DATA COLLECTION
To implement our deforestation prevention model, we needed a dataset with ample data points in the categories “natural sounds” and “non-natural sounds”. The dataset we chose, ESC-50, is a collection of audio files, stored in .wav format, grouped into 50 different classes. 
Link: https://github.com/karolpiczak/ESC-50#esc-50-dataset-for-environmental-sound-classification
Each of these classes consists of 40 examples. An overview of the dataset is provided in the table below:
![image](https://user-images.githubusercontent.com/22393419/147980369-2415ee91-22f1-431f-b662-e7da77a134ba.png)
  
WildAI uses deep learning model to perform audio classification on real time audio collected by the hardware. To classify the captured audio as a threat or a natural sound, we have made use of a Convolutional Neural Network (CNN) model. Currently, we pass the audio waveform to a pre-trained transfer learning model, YAMNet, which uses the MobileNet v1 architecture and was trained using the AudioSet corpus (includes environmental sounds and several unnatural sounds). The YAMNet model outputs the audio in the form of embeddings which is then passed to a simple sequential model allowing it to make the classification. The simple sequential model has an input layer with 512 neurons is implemented. The network has been trained on 2000 audio clips from the ESC-50 dataset. This model is able to make the classification with 91% accuracy. TensorFlow has been used to support this implementation. 
This functionality has been showcased in this prototype through a web based UI whereas the MVP will have a mobile application for consumer engagement.   
The interface has been created using Streamlit, which allows the user to browse and drop an audio file. Streamlit communicates with the deep learning model which performs audio classification on it, and sends out results if the audio has a potential threat sound in it or not. 
![image](https://user-images.githubusercontent.com/22393419/185800272-66e05a9a-0e99-4f97-ada6-d40ff0fa04cc.png)
