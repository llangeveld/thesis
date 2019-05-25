#!/usr/bin/python3
"""This program runs on the feminist file"""

import csv
import random
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lemmatizer import Lemmatizer
from nltk.corpus import stopwords


def two_lists(f):
    tweets = []
    stances = []
    for tweet, stance in f:
        tweets.append(tweet)
        stances.append(stance)

    return tweets, stances


def pre_process(tweets):
    nlp = spacy.load('en_core_web_sm')
    lemmatizer = Lemmatizer()
    tokenized = [nlp(tweet.lower()) for tweet in tweets]
    strTweets = []
    for tokens in tokenized:
        lemmas = []
        for token in tokens:
            if token.text not in stopwords.words('english') and token.text.isalpha():
                lemma = lemmatizer(token.text, token.pos_)
                lemmas.append(lemma)
            elif token.text not in stopwords.words('english'):
                lemmas.append(token)
        strTweets.append(str(lemmas))
    return strTweets


def get_data(strTweets, stances):
    le = preprocessing.LabelEncoder()
    tf = TfidfVectorizer(max_features=5000)

    X_train, X_test, y_train, y_test = train_test_split(strTweets, stances, test_size=0.3, random_state=0)
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    tf.fit(strTweets)
    X_train = tf.transform(X_train)
    X_test = tf.transform(X_test)

    return X_train, X_test, y_train, y_test


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)

    print("Accuracy score: {0}".format(accuracy_score(predictions, y_test)*100))


def main():
    with open("resources/feminismStance.csv") as f:
        feminismFile = [line for line in csv.reader(f, delimiter=",")]

    random.shuffle(feminismFile)

    tweets, stances = two_lists(feminismFile)
    f = 0
    a = 0
    n = 0
    for stance in stances:
        if stance == "F":
            f += 1
        elif stance == "A":
            a += 1
        elif stance == "N":
            n += 1
    print("Favour: {}, Against: {}, Neutral: {}".format(f, a, n))
    #strTweets = pre_process(tweets)

    #X_train, X_test, y_train, y_test = get_data(strTweets, stances)

    #run_tests(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


# NOTE : Favour = 186, Against = 513, Neutral = 250