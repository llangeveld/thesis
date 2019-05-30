#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import re
import spacy
import random
import pandas as pd

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


def import_data():
    with open("resources/finalData.csv") as f:
        f1 = [line for line in csv.reader(f, delimiter=",")]
    with open("resources/feminismStance.csv") as f2:
        f3 = [line for line in csv.reader(f2, delimiter=",")]
    return f1, f3


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
        strTweets.append(str(lemmas))

    return strTweets


def data_run(tweets, stances, option, genders=None, tweets2=None, stances2=None):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    cutOff = int(0.7*len(tweets))

    if option == "AG":
        d = {'tweet':tweets, 'gender':genders}
        df = pd.DataFrame(d)
        ct = ColumnTransformer(
            [('vec', vectorizer, 'tweet'),
            ('lab', le, 'gender')])
        X_features = ct.fit_transform(df)
        X_train, X_test, y_train, y_test = X_features[cutOff:], X_features[:cutOff], stances[cutOff:], stances[:cutOff]

    elif option == "A":
        vectorizer.fit(tweets)
        X_train, X_test, y_train, y_test = tweets[cutOff:], tweets[:cutOff], stances[cutOff:], stances[:cutOff]


    elif option == "B":
        vectorizer.fit(tweets + tweets2)
        X_train, X_test, y_train, y_test = tweets2, tweets[:cutOff], stances2, stances[:cutOff]

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    le.fit(stances)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    f1 = run_tests(X_train, X_test, y_train, y_test)
    return f1


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')*100
    return f1


def main():
    dataMT, dataF = import_data()
    agAC = 0
    aAC = 0
    bAC = 0
    for i in range(10):
        random.shuffle(dataMT)

        tweetsMT = []
        genders = []
        stancesMT = []
        for _, tweet, gender, stance in dataMT:
            tweetsMT.append(tweet)
            genders.append(gender)
            stancesMT.append(stance)

        tweetsF = []
        stancesF = []
        for tweet, stance in dataF:
            tweetsF.append(tweet)
            stancesF.append(stance)

        tweetsMT = pre_process(tweetsMT)
        tweetsF = pre_process(tweetsF)
        taskAG = data_run(tweetsMT, stancesMT, "AG", genders=genders)
        taskA = data_run(tweetsMT, stancesMT, "A")
        taskB = data_run(tweetsMT, stancesMT, "B", tweets2=tweetsF, stances2=stancesF)

        agAC += taskAG
        aAC += taskA
        bAC += taskB

    print("Task A w/ gender: {}".format(agAC/10))
    print("Task A w/o gender: {}". format(aAC/10))
    print("Task B: {}".format(bAC/10))



if __name__ == "__main__":
    main()