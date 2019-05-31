#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import re
import spacy
import random

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


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
        genders.append(genders)
        stances.append(stances)

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
        print(tweet)
        print(lemmas, "\n")
        strTweets.append(str(lemmas))

    return strTweets


def data_run(trainTweets, trainStances, testTweets, testStances, option, trainGenders=None, testGenders=None):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(trainTweets+testTweets)
    cutOff = int(0.7*len(tweets))

    if option == "AG":
        vectorizer.fit(tweets+genders)
        genders = le.fit_transform(genders)
        fullList = [str([tweet, gender]) for tweet, gender in zip(tweets, genders)]
        X_train, X_test, y_train, y_test = fullList[cutOff:], fullList[:cutOff], stances[cutOff:], stances[:cutOff]


    le.fit(trainStances + testStances)
    trainStances = le.transform(trainStances)
    trainStances = le.transform(trainSTances)

    f1 = run_tests(trainTweets, testTweets, trainStances, testStances)
    return f1


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')*100
    return f1


def main():
    dataTrain, dataTest, dataF = import_data()
    
    trainTweets, trainGenders, trainStances = get_data(dataTrain)
    testTweets, testGenders, testStances = get_data(dataTest)

    tweetsF = []
    stancesF = []
    for tweet, stance in dataF:
        tweetsF.append(tweet)
        stancesF.append(stance)

    trainTweets = pre_process(trainTweets)
    testTweets = pre_process(testTweets)
    tweetsF = pre_process(tweetsF)
    taskAG = data_run(trainTweets, trainStances, testTweets, testStances, "AG", trainGenders=trainGenders, testGenders=testGenders)
    taskA = data_run(trainTweets, trainStances, testTweets, testStances,"A")
    taskB = data_run(tweetsF, stancesF, testTweets, testStances,"B")

    print("Task A w/ gender: {}".format(agAC/10))
    print("Task A w/o gender: {}". format(aAC/10))
    print("Task B: {}".format(bAC/10))



if __name__ == "__main__":
    main()