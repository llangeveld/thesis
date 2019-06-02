#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import re
import spacy
import numpy as np

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, coo_matrix


def import_data():
    """Imports the data from the training-, testing- and feminism-files.
    :return: The files, in list-form (2D)."""
    with open("resources/train.csv") as thing:
        f1 = [line for line in csv.reader(thing, delimiter=",")]
    with open("resources/feminismStance.csv") as thing2:
        f3 = [line for line in csv.reader(thing2, delimiter=",")]
    with open("resources/test.csv") as thing3:
        f2 = [line for line in csv.reader(thing3, delimiter=",")]
    return f1, f2, f3


def get_data(data):
    """Takes a list and returns the differents elements (tweets, genders, stances)
    :param data: Matrix-list, with columns tweetID, tweet, gender, and stance
    :return: Seperate lists with tweets, genders, and stances"""
    tweets = []
    genders = []
    stances = []
    for _, tweet, gender, stance in data:
        tweets.append(tweet)
        genders.append(gender)
        stances.append(stance)

    return tweets, genders, stances


def pre_process(tweets):
    """Pre-processes the tweets by removing URLS,
    lower-casing the tweets and transforming the
    words to lemmas.
    :param tweets: list of strings
    :return: a list of pre-processed strings"""
    nlp = spacy.load('en')
    strTweets = []
    for tweet in tweets:
        noURLS = re.sub(r"http\S+", "", tweet)
        tweetText = nlp(noURLS.lower())
        lemmas = []
        for token in tweetText:
            lemmas.append(token.lemma_)
        strTweets.append(' '.join(lemmas))

    return strTweets


def transform_tg(tweets, genders, le):
    """Transforms the genders and makes one
    features of genders + tweets using hstack
    :param tweets: A list of vectors
    :param genders: A list of strings
    :param le: the gender's LabelEncoder
    :return: The combined gender + tweet-features"""
    genders = le.transform(genders)
    genders = np.reshape(genders, (-1, 1))
    tweets = coo_matrix(tweets)
    stacked = hstack((tweets, genders))
    return stacked


def data_run(trainTweets, trainStances, testTweets, testStances,
             option, trainGenders=None, testGenders=None):
    """Prepares the data for actual running, and sends the data through to
    the linear SVC.
    :param trainTweets: A list of strings
    :param trainStances: A list of strings
    (all items being eiter 'F', 'N' or 'A')
    :param testTweets: see trainTweets
    :param testStances: see trainStances
    :param option: A string to see what test is being run
    :param trainGenders: An optional list of strings
    (all items being either 'F' or 'M'),
    only if option == 'AG'
    :param testGenders: see trainGenders
    :return: the F1-score for the run test"""

    # Vectorizer
    count_word = TfidfVectorizer(ngram_range=(3, 3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(trainTweets)
    trainTweets = vectorizer.transform(trainTweets)
    testTweets = vectorizer.transform(testTweets)

    # For Task A w/ gender
    if option == "AG":
        le2 = preprocessing.LabelEncoder()
        le2.fit(trainGenders)
        # Stack the features together
        trainTweets = transform_tg(trainTweets, trainGenders, le2)
        testTweets = transform_tg(testTweets, testGenders, le2)

    # Transform stances
    le = preprocessing.LabelEncoder()
    le.fit(trainStances)
    trainStances = le.transform(trainStances)
    testStances = le.transform(testStances)

    f1 = run_tests(trainTweets, testTweets, trainStances, testStances)
    return f1


def run_tests(X_train, X_test, y_train, y_test):
    """Runs the linear SVC.
    :param X_train: List of features to train on
    :param X_test: List of features to test on
    :param y_train: List of labels for training
    :param y_test: List of labels for testing
    :return: F1-score"""
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
    taskAG = data_run(trainTweets, trainStances, testTweets, testStances,
                      "AG", trainGenders=trainGenders, testGenders=testGenders)
    print("# Running task A...")
    taskA = data_run(trainTweets, trainStances, testTweets, testStances, "A")
    print("# Running task B...")
    taskB = data_run(tweetsF, stancesF, testTweets, testStances, "B")

    print("\nResults:")
    print("Task A w/ gender: {}".format(taskAG))
    print("Task A w/o gender: {}". format(taskA))
    print("Task B: {}".format(taskB))


if __name__ == "__main__":
    main()
