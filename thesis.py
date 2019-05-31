#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import re
import spacy
import random
import numpy as np

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack


def import_data():
    with open("resources/train.csv") as thing:
        f1 = [line for line in csv.reader(thing, delimiter=",")]
    with open("resources/feminismStance.csv") as thing2:
        f3 = [line for line in csv.reader(thing2, delimiter=",")]
    with open("resources/test.csv") as thing3:
        f2 = [line for line in csv.reader(thing3, delimiter=",")]
    return f1, f2, f3


def get_data(data):
    tweets = []
    genders = []
    stances = []
    for _, tweet, gender, stance in data:
        tweets.append(tweet)
        genders.append(gender)
        stances.append(stance)

    return tweets, genders, stances


def pre_process(tweets):
    nlp = spacy.load('en')
    strTweets = []

    for tweet in tweets:
        noURLS = re.sub(r"http\S+", "", tweet)
        #noHashtags = re.sub(r"#", "", noURLS)
        tweetText = nlp(tweet.lower())
        lemmas = []
        for token in tweetText:
            if not token.is_stop and token.text.isalpha():
                lemmas.append(token.lemma_)
        strTweets.append(''.join(lemmas))

    return strTweets


def data_run(trainTweets, trainStances, testTweets, testStances, option, trainGenders=None, testGenders=None):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    print("## Running vectorizer...")
    vectorizer.fit(trainTweets+testTweets)

    if option == "AG":
        le2 = preprocessing.LabelEncoder()
        le2.fit(trainGenders+testGenders)
        trainGenders = le2.transform(trainGenders)
        testGenders = le2.transform(testGenders)
        trainTweets = vectorizer.transform(trainTweets)
        testTweets = vectorizer.transform(testTweets)
        trainList = hstack([trainTweets, newTG])
        testList = [str([tweet, gender]) for tweet, gender in zip(testTweets, testGenders)]
        trainTweets = trainList.copy()
        testTweets = testList.copy()

    print("## Fitting stances...")
    le.fit(trainStances + testStances)
    trainStances = le.transform(trainStances)
    testStances = le.transform(testStances)

    #trainTweets = vectorizer.transform(trainTweets)
    #testTweets = vectorizer.transform(testTweets)

    f1 = run_tests(trainTweets, testTweets, trainStances, testStances)
    return f1


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')*100
    return f1


def main():
    print("# Importing data...")
    dataTrain, dataTest, dataF = import_data()
    
    trainTweets, trainGenders, trainStances = get_data(dataTrain)
    testTweets, testGenders, testStances = get_data(dataTest)

    tweetsF = []
    stancesF = []
    for tweet, stance in dataF:
        tweetsF.append(tweet)
        stancesF.append(stance)

    print("# Preprocessing data...")
    trainTweets = pre_process(trainTweets)
    testTweets = pre_process(testTweets)
    tweetsF = pre_process(tweetsF)

    print("# Running task AG...")
    taskAG = data_run(trainTweets, trainStances, testTweets, testStances, "AG", trainGenders=trainGenders, testGenders=testGenders)
    print("# Running task A...")
    taskA = data_run(trainTweets, trainStances, testTweets, testStances,"A")
    print("# Running task B...")
    taskB = data_run(tweetsF, stancesF, testTweets, testStances,"B")

    print("\nResults:")
    print("Task A w/ gender: {}".format(taskAG))
    print("Task A w/o gender: {}". format(taskA))
    print("Task B: {}".format(taskB))



if __name__ == "__main__":
    main()