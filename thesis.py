#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import itertools
import random
import re
import emoji
import pandas as pd

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import spacy
from spacy.lemmatizer import Lemmatizer
from nltk.corpus import stopwords


def pre_process(dataMT):
    nlp = spacy.load('en')
    tokenized = []
    genders = []
    stances = []
    strTweets = []
    newFile = []

    for tweetID, tweet, gender, stance, in dataMT:
        noURLS = re.sub(r"http\S+", "", tweet)
        #demoji = emoji.demojize(noURLS)
        tweetText = nlp(noURLS.lower())
        genders.append(gender)
        stances.append(stance)
        lemmas = []
        for token in tweetText:
            if token.text not in stopwords.words('english') and token.text.isalpha():
                lemmas.append(token.lemma_)
        strTweets.append(str(lemmas))

    return strTweets, genders, stances


def import_data():
    with open("resources/finalData.csv") as f:
        f = [line for line in csv.reader(f, delimiter=",")]
    return f
    

def get_data(strTweets, genders, stances):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(strTweets)

    finalList = [[tweet, gender] for tweet, gender in zip(strTweets, genders)]

    X_train, X_test, y_train, y_test = train_test_split(finalList, stances, test_size=0.3, random_state=0)
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')*100
    print("Accuracy_score: {0}".format(f1))


def do_analysis(dataMT):
    strTweets, genders, stances = pre_process(dataMT)
    X_train, X_test, y_train, y_test = get_data(strTweets, genders, stances)
    run_tests(X_train, X_test, y_train, y_test)

def main():
    dataMT = import_data()  # [tweetID, tweetText, gender, stance]
    do_analysis(dataMT)


if __name__ == "__main__":
    main()


"""
NOTES:
Favour = 406, Neutral = 348, Against = 122
Female = 497, Male = 379

fF = 250, fA = 59, fN = 188
mF = 156, mA = 63, mN = 160
h_stack 
"""