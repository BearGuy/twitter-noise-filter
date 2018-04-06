'''
model was created in a separate and saved using the following:

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
'''


import string
import os
import numpy as np
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

import pickle

tweet_data = pd.read_csv("text_quality.csv", encoding = "ISO-8859-1")
tweet_data = tweet_data.drop_duplicates(subset="text")

tweet_data['class'] = tweet_data['class'].map({'reg': 1, 'cont': 0})

corpus = tweet_data["text"].values.tolist()
y_labels = tweet_data['class'].values.tolist()



# Quick example with sklearn

vectorizer = CountVectorizer(stop_words='english')
vecMatrix = vectorizer.fit_transform(corpus)#.todense()

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def clean_tweet(tweet):
    tokens = tweet.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def tweet_to_line(tweet, vocab):
    # clean
    tokens = clean_tweet(tweet)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

# load doc and add to vocab

def process_tweets(corpus, vocab):
    lines = list()
    for tweet in corpus:
        line = tweet_to_line(tweet, vocab)
        lines.append(line)
    return lines


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# load training data
tweet_data = process_tweets(corpus, vocab)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweet_data)


train_tweets, test_tweets, ytrain, ytest = train_test_split(tweet_data, y_labels, test_size=0.33)

Xtrain = tokenizer.texts_to_matrix(train_tweets, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_tweets, mode='freq')


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(Xtest, ytest, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def predict_sentiment(review, vocab, tokenizer, model):
    # clean
    tokens = clean_tweet(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0,0])

prediction_data = pd.read_csv("text_quality.csv", encoding = "ISO-8859-1")
prediction_data = prediction_data.drop_duplicates(subset="text")
prediction_data['class'] = prediction_data['class'].map({'reg': 1, 'cont': 0})
prediction_data = prediction_data.values

i = 0
predicted_class = []

for i in range(len(prediction_data)):
    predicted_class.append(predict_sentiment(prediction_data[i,0],vocab,tokenizer,loaded_model))

conf = confusion_matrix(prediction_data[:,17].tolist(), predicted_class)
conf_normalized = conf / conf.astype(np.float).sum(axis=1)

tp = conf[0,0]
fn = conf[0,1]
fp = conf[1,0]
tn = conf[1,1]

recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print("Confusion Matrix")
print(conf)

print("Confusion Matrix Normalized")
print(conf_normalized)
print("Recall")
print(recall)
print("Specificity")
print(specificity)
print("Precision")
print(precision)
