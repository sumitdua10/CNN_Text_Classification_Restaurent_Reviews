
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score


# Download the data:
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip


# some configuration
MAX_SEQUENCE_LENGTH = 80 # Maximum sequence length of text in one example/row of input data
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

#1. REad the data - Input Reviews
FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Udemy\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Restaurant_Reviews.tsv"
#TestFile = FILENAME = "C:\\Users\\IBM_ADMIN\\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Udemy\\Machine Learning A-Z Template Folder\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\test.tsv"
dataset = pd.read_csv(FILENAME, delimiter = '\t', quoting = 3)
m_Num_Rows = dataset.shape[0]
print(dataset.head())
print(dataset.shape)
targets = dataset['Liked']

y = dataset.iloc[:, 1].values # y is the output 1 indicates favorable and 0 indicates not good.
print(y[0:5])

print("Max Length of sequence is ", max(len(dataset.loc[i,'Review']) for i in range(m_Num_Rows)))
print("Min Length of sequence is ", min(len(dataset.loc[i,'Review']) for i in range(m_Num_Rows)))
print("Avg Length of sequence is ", sum(len(dataset.loc[i,'Review']) for i in range(m_Num_Rows)) / m_Num_Rows)


#2. Tokenize the text iby converting them into integers
# convert the sentences (strings) into integers. Sequen
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(dataset['Review'])

sequences = tokenizer.texts_to_sequences(dataset['Review'])
print(len(sequences))
print(sequences[0])
print(max(sequences[10]))

# 3. get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
print('Sample is .', list(word2idx.values())[0:10])
print('Sample is .', list(word2idx.keys())[0:10])


# 4. pad sequences so that we get a m x T matrix where m is the no. of sentences (training examples) and T is the seq length
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding = 'post', truncating='post')
print('Shape of data tensor:', data.shape)
print(data[0])


# 5. Load word embeddings from standordglove. which is of dimension 400K X 50 (i.e. 400K words and 50 dimensions)
print('Loading word vectors...')
word2vec = {}
with open('C:\\Users\\IBM_ADMIN\Desktop\\Personal\\Trainings\\Machine Learning\\Data\\Standfordglove\\glove.6B.50d.txt', encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))
print(len(word2vec['the']))


# 6. Create embedding dimension of size 2072 (unique corpus words found in our all text rows) X 50 (no. of cols in embeeding dimension)
# Creating Bag of words model instead of this would have given shape of m X 2072. but embedding would give m X 2072 X 50
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
print('Num_words. ', num_words)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word) #get(word) is used instead of [word] as it won't give exception in case word is not found
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)


# 7. load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False)


#8. Buildin the model
print('Building model...')

LAYER1_NUM_FILTERS = 32
LAYER1_KERNAL_SIZE = 8

LAYER2_NUM_FILTERS = 24
LAYER2_KERNAL_SIZE = 8

LAYER3_NUM_FILTERS = 16
LAYER3_KERNAL_SIZE = 8


# train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
print(input)
x = embedding_layer(input_)
print(x)
x = Conv1D( filters = LAYER1_NUM_FILTERS, kernel_size=LAYER1_KERNAL_SIZE, activation='relu', padding='same')(x) # 128 is the no. of filtesr and 3 is the kernal size
x = MaxPooling1D(pool_size=3, strides=2)(x)
x = Conv1D(LAYER2_NUM_FILTERS, LAYER2_KERNAL_SIZE, activation='relu', padding='same')(x)
x = MaxPooling1D(pool_size=3, strides=2)(x)
x = Conv1D(LAYER3_NUM_FILTERS, LAYER3_KERNAL_SIZE, activation='relu', padding='same')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(16, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(input_, output)

print(model.summary())
print("Modelling Done. Setting the compile parameter")

model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size = 0.20, random_state=0)

r = model.fit(
  X_train,
  y_train,
  batch_size=12, #BATCH_SIZE,
  epochs=7, #EPOCHS,
  validation_split= VALIDATION_SPLIT
)
print("Training Done", r)


score = model.evaluate(X_test, y_test) #, batch_size=128)
print("Accuracy Score = ", score)
response = model.predict(X_test)
print(y_test[0:10])
print(response[0:10])
print(dataset.iloc[230,0])


