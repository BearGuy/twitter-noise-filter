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

import pickle 
# nltk.download('stopwords')

tweet_data = pd.read_csv("text_quality.csv", encoding = "ISO-8859-1")
tweet_data = tweet_data.drop_duplicates(subset="text")

tweet_data['class'] = tweet_data['class'].map({'reg': 1, 'cont': 0})

corpus = tweet_data["text"].values.tolist()
y_labels = tweet_data['class'].values.tolist()


# Quick example with sklearn

vectorizer = CountVectorizer(stop_words='english')
vecMatrix = vectorizer.fit_transform(corpus)#.todense()

print( len(vectorizer.vocabulary_) )

#final_tweet_data = pd.concat([tweet_data, pd.DataFrame(vecMatrix)], axis=1)
#final_tweet_data.to_csv('tweet_quality_with_text.csv', encoding='utf-8')


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
'''
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)
    
def vocab_min_occurence():
    min_occurane = 2
    more_tokens = [k for k,c in vocab.items() if c >= min_occurane]
    return more_tokens
    '''  
def process_tweets(corpus, vocab):
    lines = list()
    for tweet in corpus:
        line = tweet_to_line(tweet, vocab)
        lines.append(line)
    return lines
    
#all_tokens_flat = [item for sublist in all_tokens for item in sublist]
    
# vocab = Counter()
# vocab.update(all_tokens_flat)

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data
tweet_data = process_tweets(corpus, vocab)
tokenizer = Tokenizer()

# Train Tweets
# fit the tokenizer on the data
tokenizer.fit_on_texts(tweet_data)
#encode training data set

train_tweets, test_tweets, ytrain, ytest = train_test_split(tweet_data, y_labels, test_size=0.33)

Xtrain = tokenizer.texts_to_matrix(train_tweets, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_tweets, mode='freq')

# # Test Tweets
# # fit the tokenizer on the data
# tokenizer.fit_on_texts(test_tweets)
# #encode training data set
# Xtest = tokenizer.texts_to_matrix(test_tweets, mode='freq')



print(Xtrain.shape)
print(Xtest.shape)


n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(20, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
