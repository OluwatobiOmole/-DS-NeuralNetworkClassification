# Neural Network Comparison Classification

**PART 1 [View the Notebook](Part1.ipynb)**

The aim of this part is to use a variety of model architectures on data
from a Guardian dataset to complete a multi-class prediction problem and
evaluate the different models used. The dataset itself has seven columns
but only three columns are of interest which are 'bodyContent',
'webTitle' and 'sectionNames'. 'bodyContent' will serve as the main
descriptive feature and 'webTitle' will be used as a descriptive feature
at one point while 'sectionNames' is the target class.

**Pre-Processing Text into suitable format for Neural and Non-neural
Models:**

The necessary columns 'bodyContent', 'sectionNames' and 'webTitle' were
cleaned and transformed into a structure that is better suited for
analysis. Using a defined function that takes a string of text and
preprocesses it by converting to lowercase, removing unwanted characters
using regular expressions, tokenizing the text into words, removing stop
words, lemmatizing the words, and joining the words back into a single
string. Tokenize text data in the \'body Content\' column and create a
tokenizer object with a maximum number of features of 5000 and fits it
on the text data. The text is then converted to a sequence of integers
using and the resulting sequences are padded to ensure equal length. The
padded sequences are stored in the variable X which holds the
descriptive features. Create a dictionary to map the unique labels in
the \'sectionNames\' column called to numerical values as this is the
target variable, Y. From the descriptive and target features create
obtain training, validation, and test sets for the models. Pad the
sequences of the training, validation, and testing sets to the same
length which is necessary for feeding the data into the models.

**Some notes:**

In explaining the model architecture in this part, all the models will
share some properties highlighted here. All models will use the softmax
activation function in the final layer as it is a multiclass
classification problem. All models will use the Adam optimizer as it
helps models converge faster and more efficiently than traditional SGD.
The evaluation metric will be accuracy for all models. The batch size
will be 32 and sparse categorical cross-entropy loss function will be
used as target variables were label encoded with numerical integers.
There will be 16 neurons in the RNN, LSTM and CNN layers. This is
explained here to avoid redundancy when explaining the models.

**RNN Variants**

**Basic RNN with single layer (No Embeddings Layer):**

Create an RNN model architecture. The model includes a Reshape layer
that reshapes the input to have the same shape as the input length,
followed by a Dropout layer to prevent overfitting. Then a SimpleRNN
layer is added. Another Dropout layer is added before the final Dense
layer with a softmax activation function. The model is then compiled
using the Adam optimizer and sparse categorical cross-entropy loss
function, with accuracy as the evaluation metric. It is trained on the
training data with the validation data used to monitor the model\'s
performance during training. The training is run for 5 epochs with a
batch size of 32.

![image](https://github.com/user-attachments/assets/fd477973-8243-4817-92c0-eede697da704)
![image](https://github.com/user-attachments/assets/abc0a518-8087-408f-a093-cd54ee561bd7)



Basic RNN with single layer (No Embeddings Layer) performance

**Basic LSTM with single layer (No Embeddings Layer):**

Create an LSTM model architecture. The input data is reshaped to have
dimensions using the Reshape layer, and a dropout layer is added to help
prevent overfitting. Then, an LSTM layer is added. Another dropout layer
is added, and then a Dense layer with a softmax activation function is
added to output the predicted probabilities for each class. The model is
trained the training data. The same process is used to define and train
the RNN model, but with a SimpleRNN layer instead of an LSTM layer.

![image](https://github.com/user-attachments/assets/8830c9de-10cc-4978-adae-c152f2805483)
![image](https://github.com/user-attachments/assets/1fe77007-27dc-4a27-83d8-97c1a492ea7b)



Basic LSTM with single layer (No Embeddings Layer) performance

**LSTM with multiple layers (No Embeddings Layer):**

Create an LSTM model architecture. The model architecture includes a
Reshape layer that takes the input shape of the training data, two LSTM
layers with the same number of neurons, and two Dropout layers to
prevent overfitting. The final layer is a Dense layer with a softmax
activation function. The model is trained and the history of accuracy
and loss during training is saved.

![image](https://github.com/user-attachments/assets/f323ea55-3ec6-4df6-9e86-0b84ac90de9c)
![image](https://github.com/user-attachments/assets/edd01115-a24e-410d-ac7d-bee9e5838eaf)



LSTM with multiple layers (No Embeddings Layer) performance

**Model Comparison of RNN Variants**

**Comparison of Basic RNN model and LSTM with single layer**


![image](https://github.com/user-attachments/assets/c7ecebf5-ae73-40fc-9215-820592270fba)

Performance of Basic RNN model and LSTM with single layer compared.

**Comparison of Single Layer LSTM with Multiple layer LSTM**

![image](https://github.com/user-attachments/assets/0ff80619-8a51-4f07-bfdf-fc3464183f91)

Performance of Single Layer LSTM with Multiple layer LSTM compared.

**Embeddings**

**On the fly Embeddings**

Create an LSTM model architecture with an embeddings layer. The
Embedding layer is added as the first layer. The model has 2 LSTM layers
with a dropout added after each LSTM layer. Next, the model has a Dense
layer a softmax activation. The training history is saved.

![image](https://github.com/user-attachments/assets/1f79c22b-d6a0-4a8e-8e7a-fc1ec9e4cf6c)
![image](https://github.com/user-attachments/assets/01b2803a-9b93-460a-ab64-9d4a24b18f3d)



On the fly Embeddings performance

**Pre-trained Embeddings (Model didn't run but this was the idea)**

Use pre-trained embeddings from TensorFlow Hub. Load the pre-trained
embeddings, and then two dense layers with ReLU and softmax activation
functions are added.

**Using Bag of Words instead of Embeddings**

Create a bag-of-words model architecture. The model takes input of
maximum number of and passes it through a dense layer with 64 units and
ReLU activation. Then, a dropout layer added to prevent overfitting.
Finally, an output layer with softmax activation is added. The history
of accuracy and loss during training is saved.

**Model Comparison of Embeddings**

**Comparison of On the Fly Embeddings Model with Bag of Words model**

![image](https://github.com/user-attachments/assets/76d25df0-369a-40a2-9f77-6047bcc2a614)

Performance of On-the-Fly Embeddings Model with Bag of Words model

**CNN for Text Classification**

**CNNs as an Alternative to an LSTM Solution**

Create a CNN model architecture. The model architecture includes an
embedding layer, followed by two convolutional layers with different
filter sizes and activation functions, a dropout layer to reduce
overfitting, and a global max pooling layer to select the most important
features. Finally, a dense layer with a softmax activation function is
used to output the predicted class probabilities.

![image](https://github.com/user-attachments/assets/087c40ff-1011-4271-b6ce-424078add10c)
![image](https://github.com/user-attachments/assets/d2035664-820b-4511-bbf3-335d0697b4aa)



CNNs as an Alternative to an LSTM Solution performance

**CNNs as an Additional Layer Before an LSTM Solution**

Create a CNN and LSTM model architecture. The model architecture
includes an embedding layer to convert text data into numerical vectors,
followed by a convolutional layer and an activation function of ReLU. A
dropout layer is used to reduce overfitting and a global max pooling
layer to select the most important features. The resulting features are
then reshaped to prepare for the LSTM layer. An LSTM layer is used
before the dense layer. Finally, a dense layer with a softmax activation
function is used.

![image](https://github.com/user-attachments/assets/a5718e47-5faa-44f9-b8c8-971e914a6e17)
![image](https://github.com/user-attachments/assets/abd3cd96-8baf-4992-8f36-2d48b6fb6e43)



**Model Comparison of CNNs for Text Classification**

**Comparison of CNN as alternative to LSTM and CNN with LSTM models**

![image](https://github.com/user-attachments/assets/08bfcbd3-bbcd-4ab4-b7e1-ca0b7140c943)

Performance of CNN as alternative to LSTM and CNN with LSTM models

**Comparison to Non-Neural Methods**

A Multinomial Naive Bayes classifier was created as the non-neural
model. The model is trained five times on the training set and evaluated
on both the training and validation sets. After all iterations, the
average training accuracy and average validation accuracy are
calculated.

![image](https://github.com/user-attachments/assets/7372a75d-41f0-43f7-b3c6-6da46c5e2f61)


Non-Neural Method performance

**Model Comparison of non-neural method to best performing neural
method**

**Comparison of non-neural method to best performing neural method**

![image](https://github.com/user-attachments/assets/871a695f-d4ae-419d-8d11-44e5b3a5c6ec)


**Performance of non-neural method to best performing neural method**

**Additional Data**

This section requires the addition of the webTitle column in combination
with the bodyContent column as separate inputs into a neural model. The
WebTitle has already been preprocessed partly, but the text data must
still be tokenized. Create a tokenizer object with a maximum number of
features of 5000 and fits it on the text data. The text is then
converted to a sequence of integers using and the resulting sequences
are padded to ensure equal length. The padded sequences are stored in
the variable X which holds the descriptive features. The target features
have already been label encoded so new separate training, validation and
test sets containing either webTitle or bodyContent values.

Create a model with two input layers. The inputs are web titles and body
content. Each input goes through an embedding layer and a LSTM layer,
followed by a dropout layer for regularization. The outputs of both LSTM
layers are concatenated and passed through a dense layer with softmax
activation.

![image](https://github.com/user-attachments/assets/2739f6f5-9e69-4507-b2f9-d77c1202279c)


**Model Comparison of Additional Data model and best performing neural
model**

**Comparison of Two Input Model with best performing neural model**

![image](https://github.com/user-attachments/assets/af3bc27c-6129-4200-ac3d-9a6efe61ac4c)


Performance of Two Input Model with best performing neural model

**PART 2 [View the Notebook](Part2.ipynb)**

The aim of this part is to use a variety of model architectures on data
from a BBC dataset to complete a classification prediction problem and
evaluate the different models used. Some models will use models created
in Part1 as inputs into the model demonstrating transfer learning. The
dataset itself has two columns and both columns are of interest which
are 'category' and 'text'. 'text' will serve as the descriptive feature
while 'category' is the target class.

**Pre-Processing Text into suitable format for Neural Networks:**

The necessary columns in the dataset for creating the models, 'category'
and 'text', were preprocessed the same way as text in part 1. All
preprocessing is the same as part 1 and new training, validation, and
tests.

**Model Variant from Scratch (CNN model) - No transfer Learning:**

Create a CNN model architecture. The model has an embedding layer
followed by two convolutional layers with a set filter size and
heterogenous kernel sizes of 3 and 4, respectively. A dropout layer with
a rate of is added to prevent overfitting, followed by a global max
pooling layer and a dense output layer with a softmax activation
function.

![image](https://github.com/user-attachments/assets/27c1f00c-8063-4065-8487-7bb2e23506df)


Performance of Model Variant from Scratch (CNN model) - No transfer
Learning

**Transfer Learning (I) - New CNN model with input from previous CNN
model created in Part 1:**

Create an LSTM model architecture. The model consists of a pretrained
CNN model from Part 1 which was the alternative to LSTM CNN variant
followed by a reshape layer, an LSTM layer with dropout, and a dense
layer with a softmax activation function. The weights of the pretrained
CNN layers are frozen to prevent them from being trained again.

![image](https://github.com/user-attachments/assets/7fd9a6c0-12a6-4c06-8d83-f7e40430909f)

Performance of Transfer Learning (I) - New CNN model with input from
previous CNN model created in Part 1.

**Transfer Learning (II) - New CNN model with input from previous CNN
model created in Part 1:**

LSTM model architecture with a pretrained LSTM-CNN model which was the
CNN model variant (Additional Layer Before an LSTM Solution) as the
base. The model has the architecture of three layers - a pretrained
model, a Reshape layer to ensure the input shape compatibility with the
LSTM layer, and an LSTM layer. The weights of the pretrained model are
frozen, and a dense output layer is added for softmax.

![image](https://github.com/user-attachments/assets/283efa70-32f6-4046-b7c5-f6e36cbed0f2)


Performance of Transfer Learning (II) - New CNN model with input from
previous CNN model created in Part 1.

**Model Comparison of Model from Scratch and Transfer Learning Models**

**Comparison of Model Variant with no transfer learning to New CNN model
with transfer learning I**

![image](https://github.com/user-attachments/assets/e11b7c01-cf4c-48a0-9eb8-db41ca097bb4)


Performance of Model Variant with no transfer learning to New CNN model
with transfer learning I

**Comparison of Model Variant with no transfer learning to New CNN model
with transfer learning II**

![image](https://github.com/user-attachments/assets/453e683f-b0b2-4b44-8189-3ddafa773b73)


Performance of Model Variant with no transfer learning to New CNN model
with transfer learning II

**Comparison of Transfer Learning Models**

![image](https://github.com/user-attachments/assets/64ff5efd-b738-43a6-8fad-fc2de828f3eb)


Performance of Transfer Learning Models

**PART 3 [View the Notebook](Part1.ipynb)**


The aim of this part is to create a text generation model based on a
model trained on the BBC dataset, to generate random text with some sort
of coherence. The approach would be to train a model on the BBC dataset
so that the model learns the patterns and relationships between words
and phrases. Then, a seed text will be provided as input to the model in
a function and the model will generate new text based on the patterns
learned from the training dataset.

**Pre-Processing Text into suitable format for Neural Networks:**

The necessary columns in the dataset for creating the models, 'category'
and 'text', were preprocessed the same way as text in part 1. However,
as it is language modeling task, it is important to make sure that the
sequence structure of the data is preserved. Therefore, the data was
split by selecting a portion of the data as the training set, and the
remaining part as the validation set without any shuffling of the data
to ensure that the sequence structure of the data is preserved, and the
model can learn to generate coherent text.

**LSTM with multiple layers for creating the generative model:**

Create a language model with an LSTM architecture. The architecture
includes an Embedding layer, two LSTM layers, and a Dense layer with a
softmax activation function.

**Generating Text**:

Generate new text using the model created. A seed text is randomly
selected from the dataset with a specified category, and the sequence is
tokenized and padded to a fixed length. Then, a function takes in the
model, tokenizer, seed text, and the number of words to generate. Within
the function, for each word to generate, the model predicts the
probability distribution for the next word, maps the predicted index to
the word using the tokenizer, and appends the word to the input text.
Finally, the function prints the generated text. Code from this was
influenced from this page
<https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/>

**Evaluate Model**:

Evaluate the language on a test set. Calculates the loss and accuracy of
the model using the evaluate method with the test data. calculates the
perplexity of the model and the test loss, test accuracy, and
perplexity.

![image](https://github.com/user-attachments/assets/f955f6ce-0563-4398-b840-ce1ad9596576)
![image](https://github.com/user-attachments/assets/b5cc34e0-f6e5-4839-ad8f-e8f706204fd2)


Performance of LSTM with multiple layers for creating the generative
model.

**References**

Brownlee, J. (2020). How to Develop a Word-Level Neural Language Model
and Use it to Generate Text. MachineLearningMastery.com.
https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
