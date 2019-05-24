#!/usr/bin/python3
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def get_files():
    with open("resources/feminismStance.csv") as f1:
        trainFile = [line for line in csv.reader(f1, delimiter=",")]
    with open("resources/annotate300.csv", 'rt') as f2:
        testFile = [line for line in csv.reader(f2, delimiter=",")]
    return trainFile, testFile

def get_data(trainFile, testFile):
    le = preprocessing.LabelEncoder()
    tf = TfidfVectorizer(analyzer='char', ngram_range=(2,4))

    X_train = [el[0] for el in trainFile]
    y_train = [el[1] for el in trainFile]
    X_test = [el[1] for el in testFile]
    y_test = [el[2] for el in testFile]

    fitThing = X_train + X_test
    tf.fit(fitThing)
    X_train = tf.transform(X_train)
    y_train = le.fit_transform(y_train)


    X_test = tf.transform(X_test)
    y_test = le.fit_transform(y_test)

    return X_train, X_test, y_train, y_test


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("F1: {0}".format(f1_score(y_test, predictions, average='micro')*100))

def main():
    trainFile, testFile = get_files()
    X_train, X_test, y_train, y_test = get_data(trainFile, testFile)
    run_tests(X_train, X_test, y_train, y_test)

main()