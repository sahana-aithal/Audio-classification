# Audio-classification
This project aims at classifying audio files to detect unnatural sounds from the natural sounds found in forests to classify/identify illegal deforestation or poaching

DATA COLLECTION
To implement our deforestation prevention model, we needed a dataset with ample data points in the categories “natural sounds” and “non-natural sounds”. The dataset we chose, ESC-50, is a collection of audio files, stored in .wav format, grouped into 50 different classes. Each of these classes consists of 40 examples. An overview of the dataset is provided in the table below:
![image](https://user-images.githubusercontent.com/22393419/147980369-2415ee91-22f1-431f-b662-e7da77a134ba.png)
  
 
To suit our project's specific goals, we decided to use only 2 columns of this dataset, for a total of 20 classes. The 2 major categories we used data from were “Natural Soundscapes & Water Sounds'' and “Exterior/Urban Noises”. Each category consists of 10 labels with 40 examples each, for a total of 400 data points per category. Our thoughts were as follows: if we could detect exterior/urban noises in a rainforest environment, such as chainsaws, engines, or hand saws, we could predict more accurately when illegal deforestation activity was taking place.
To further simplify the dataset, we decided to relabel the two categories of the dataset either “natural” or “unnatural”. This changed the problem from a classification with 20 separate labels to a binary classification problem with 2 classes. Each example from the category “Natural Soundscapes & Water Sounds” was assigned a label value of “1”, representing a natural sound, while the examples from the “Exterior/Urban Noises” category were assigned a label value of “0”, representing an unnatural sound.
 
We had to manually move the data from the .csv file that was provided with the ESC-50 dataset to a new file, where we could sort and relabel the data for use in our project. During this process, we were able to simplify the data in the .csv to suit our needs. The .csv file we created had only the relevant information we required for the project: a column consisting of filenames corresponding to the audio files we were hoping to analyze, and the labels associated with those filenames, tying them to the “natural” or “unnatural” categories. These filenames and labels were imported into the python notebook at random to use for training and testing the sequential model later on. 
 
4	DATA PREPROCESSING

The preprocessing of the data turned out to be crucial for improving the performance of the model. The key challenge for this project was finding a way to convert the data from .wav format into a format we could use to train a neural network. We achieved this through the librosa library, a python package specifically designed for music and audio analysis. Using a loop, we iterate through all the files in the folder containing the .wav files, using the filenames from the .csv file we imported. Each .wav file is imported using the librosa function load, which processes an audio file and converts it to a floating-point time series, then temporarily stores it in a variable called ‘signal’. 

Immediately after the audio is converted to a floating-point time series, represented as an array of data type float32, it is passed into the librosa function mfcc. Mfcc converts the floating-point time series into an array of 216 Mel-Frequency Cepstral Coefficients, which is essentially a representation of an audio file’s short-term power spectrum. The Mel-Frequency scale is unique in that its frequency scale is equally spaced in a way that best represents what a human auditory system can process. When the system is finished processing each .wav file, the output is an array representation of the dataset, an array of 216 coefficients for each of the 800 examples. The dataset is then normalized to provide better classification results. Below are examples of the MFCC function’s output for an audio file in the natural category, as well as the unnatural category. 
![image](https://user-images.githubusercontent.com/22393419/147980413-d27c55fb-ed77-49c0-bf92-67a247ff6062.png)
![image](https://user-images.githubusercontent.com/22393419/147980425-8ae7f3fb-6352-4e30-b980-a154ba84476b.png)
 
These Mfcc arrays are the data that we will pass into the neural network. The dataset is split into train and test data, the first 600 examples used for the training data, while the final 200 are used for evaluation. 
     
METHODOLOGY
As our project turned into a binary classification problem, we decided that a sequential neural network would be the best approach. Sequential models perform best when the classification is straight-forward, the system has a plain stack of layers, and there is only 1 input and 1 output tensor. The first layer of the sequential model is a dense input layer with a shape of 1x216 to accommodate the shape of the input tensors. The associated activation function to this input layer is Relu.
The following layer is a dropout layer with a 0.5 dropout rate. The dropout layer in the model is to combat overfitting of the training dataset. Initially, we were seeing a very high model accuracy on the training dataset of approximately 85%. However, when the model was evaluated with the testing dataset, the model only performed with around ~50% accuracy. This was proof that the model was overfitting to the training dataset and could not perform accurately when new data was introduced. The dropout layer is responsible for randomly deactivating 50% of the nodes in the neural network, reducing the amount of unnecessary data that is being used to fit the model. This brought the training data accuracy down slightly (from 85% to roughly 80%) but improved the testing accuracy significantly (from 50% to just over 70%).
The final layer of the sequential model is another dense layer with 1 node and the sigmoid activation function. This layer is responsible for assigning the input data to one of the two classes, natural or unnatural.
When we compile the model, we use the Adam optimizer with a learning rate of 0.00005. After testing multiple learning rates and optimizer architectures, we determined that this combination of optimizer and learning rate gave us the best results. The loss function we used was binary cross entropy, a very standard loss function that aids in binary classification applications. Binary cross entropy is essentially a representation of the logarithmic probability that the example belongs to the positive (in our case, “natural”) class. The binary accuracy metric helps us keep track of how well our model is classifying each example into the “natural” and “unnatural” classes.


RESULTS AND INTERPRETATION 
Overall, our model is performing poorly, and we believe this is a result of a lack of good data. The portion of the dataset we’ve used was originally split into 20 classes based on the contents of the .wav files in the ESC-50 dataset. We combined the data into 2 classes based on the categories the sounds were tied to, hoping that the outcome would be a large dataset in 2 categories. However, the differences in the .wav files were significant enough to reduce the effectiveness of the model. Alongside this, the dataset only has 800 examples, 600 of which are being used to train the model. This is a small amount of data to train such a complex model.
We used two metrics to evaluate the performance of our model, loss calculated with binary cross-entropy and accuracy of the model as a percentage.  Binary cross-entropy is a common function used to calculate a loss metric for binary classification applications. Binary accuracy is a measure of how many of the labels are successfully predicted when the input data is passed through the model.
You can see below a typical output of the performance of the model during training. As the training progresses across 100 epochs, the loss decreases, quickly at first before steadying out around .51. The binary accuracy of the model increases throughout the training but begins to level off around 80% for the training stage.

![image](https://user-images.githubusercontent.com/22393419/147980498-5556ea07-a711-42f6-9578-81b14419c4ac.png)

During evaluation of the model’s performance with a test dataset, we are seeing a binary accuracy of roughly 70-72%, and a loss around 0.57. With multiple iterations tweaking the activation functions, model optimizers, dataset preprocessing and model architecture, we were unable to improve the performance of the model beyond this 70-72% threshold. 
