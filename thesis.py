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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
from spacy.lemmatizer import Lemmatizer
from nltk.corpus import stopwords


def get_gender():
    """
    :return: two lists, one of female names and one of male names
    Function reads out annotated names-file, and makes two lists
    (one with female, and one with male names).
    """
    with open("resources/names_annotated.csv") as f:
        genderFile = [line for line in csv.reader(f, delimiter=",")]
    femaleNames = []
    maleNames = []
    for name, gender in genderFile:
        if gender == "F":
            femaleNames.append(name)
        elif gender == "M":
            maleNames.append(name)
    return femaleNames, maleNames


def classify_gender(femaleNames, maleNames, tweetFile):
    """
    :param: femaleNames: A list of female names
    :param: maleNames: A list of male names
    :param: tweetFile: A file with the raw tweets [tweetID, author, text]
    :return: genderedFile: A file with the raw tweets, with gender
    included [tweetID, author, text, gender]
    """
    genderedFile = []
    tweetIDs = []
    for tweetID, tweetAuthor, tweetText in tweetFile:
        if tweetID not in tweetIDs:
            tweetAuthor = tweetAuthor.lower()
            namesList = []
            for item in itertools.zip_longest(femaleNames, maleNames):
                femaleName = item[0]
                maleName = item[1]
                if maleName is not None and maleName.lower() in tweetAuthor:
                    namesList.append(maleName)
                elif femaleName.lower() in tweetAuthor:
                    namesList.append(femaleName)
            if namesList != []:
                name = max(namesList, key=len)
                if name in femaleNames:
                    genderedFile.append([tweetID, tweetAuthor, tweetText, "F"])
                elif name in maleNames:
                    genderedFile.append([tweetID, tweetAuthor, tweetText, "M"])
            tweetIDs.append(tweetID)
    return genderedFile


def pre_process(dataMT):
    nlp = spacy.load('en')
    tokenized = []
    genders = []
    stances = []
    strTweets = []
    newFile = []

    for tweetID, tweet, gender, stance, in dataMT:
        noURLS = re.sub(r"http\S+", "", tweet)
        noHashtags = re.sub(r"#", "", noURLS)
        demoji = emoji.demojize(noHashtags)
        tweetText = nlp(demoji.lower())
        genders.append(gender)
        stances.append(stance)
        lemmas = []
        for token in tweetText:
            if token.text not in stopwords.words('english') and token.text.isalpha():
                lemmas.append(token.lemma_)
        strTweets.append(str(lemmas))

    return strTweets, genders, stances


def import_data():
    femaleNames, maleNames = get_gender()
    with open("resources/annotate1000.csv") as f:
        annotationFile = [line for line in csv.reader(f, delimiter=",")]
    with open("resources/tweets.csv") as f2:
        tweetFile = [line for line in csv.reader(f2, delimiter=",")]
    genderedFile = classify_gender(femaleNames, maleNames, tweetFile)
    completeFile = []
    IDS = []
    for line in annotationFile:
        for otherLine in genderedFile:
            if line[0] == otherLine[0] and line[0] not in IDS:
                completeFile.append([line[0], otherLine[2], otherLine[3], line[2]])
    strTweets, genders, stances = pre_process(completeFile)
    return strTweets, genders, stances



def get_data(strTweets, genders, stances):
    le = preprocessing.LabelEncoder()
    tf = TfidfVectorizer()
    #union = FeatureUnion([('tfidf', TfidfVectorizer()), 'encoder', preprocessing.LabelEncoder()])


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
    print("Accuracy_score: {0}".format(accuracy_score(predictions,y_test)*100))
    


def try_something(strTweets, genders, stances):
    le = preprocessing.LabelEncoder()
    tf = TfidfVectorizer()

    cutOff = int(len(genders) * 0.7)
    genders = le.fit_transform(genders)
    stances = le.fit_transform(stances)
    tfTweets = tf.fit_transform(strTweets)
    do_tests = []
    for gender, tweet in zip(genders, tfTweets):
        do_tests.append([gender, tweet])
    X_train, X_test, y_train, y_test = train_test_split(do_tests, stances, test_size=0.3)
    
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(predictions, y_test)*100))



def do_analysis(dataMT):
    strTweets, genders, stances = pre_process(dataMT)
    X_train, X_test, y_train, y_test = get_data(strTweets, genders, stances)
    run_tests(X_train, X_test, y_train, y_test)


def main():
    strTweets, genders, stances = import_data()  # [tweetID, tweetText, gender, stance]
    #random.shuffle(dataMT)
    #do_analysis(dataMT)
    #try_pipeline(dataMT)
    try_something(strTweets, genders, stances)


if __name__ == "__main__":
    main()


"""
NOTES:
Favour = 405, Neutral = 348, Against = 122

PIPELINE:
main()
import_data() -> get_gender() -> classify_gender() -> pre_process() -> import_data()
main()
try_something()
"""

"""def try_pipeline(dataMT):
    #X_train, y_train, X_test, y_test = make_data(dataMT)
    cutOff = int(len(dataMT) * 0.7)
    train = dataMT[:cutOff]
    test = dataMT[cutOff:]
    union = FeatureUnion([('tfidf', TfidfVectorizer()), ('encoder', preprocessing.OneHotEncoder())])
    X_train = [[tweetText, gender] for tweetID, tweetText, gender, stance in train]
    y_train = [stance for tweetID, tweetText, gender, stance in train] 
    X_train1 = [tweetText for tweetText, gender in X_train]
    X_train2 = [gender for tweetText, gender in X_train]

    X_test = [[tweetText, gender] for tweetID, tweetText, gender, stance in test]
    X_test1 = [tweetText for tweetText, gender in X_test]
    X_test2 = [gender for tweetText, gender in X_test]
    #print(X_train)
    X_train = union.fit_transform(X_train1, X_train2)
    X_train = X_train.reshape(-1, 1)

    X_test = union.fit_transform(X_test1, X_test2)
    y_test = [stance for tweetID, tweetText, gender, stance in train]

    y_test = y_test.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    #pipeline = create_pipeline()
    #print("train")
    #pipeline.fit(X_train, y_train)
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print("accurarcy: {}".format(accuracy_score(predictions, y_test)*100))
    #print("predict")
    #predcictions = pipeline.predict(X_test)
    #print("accuracy: {}".format(accuracy_score(predictions, y_test)*100))"""

"""def make_data(dataMT):
    le = preprocessing.LabelEncoder()
    cutOff = int(len(dataMT) * 0.7)
    train = dataMT[:cutOff]
    test = dataMT[cutOff:]
    train_df = pd.DataFrame(train, columns=['tweetID', 'tweetText', 'gender', 'stance'])
    test_df = pd.DataFrame(test, columns=['tweetID', 'tweetText', 'gender', 'stance'])
    X_train = train_df[['tweetText', 'gender']]
    y_train = train_df['stance']
    X_test = test_df[['tweetText', 'gender']]
    y_test = train_df['stance']

    return X_train, y_train, X_test, y_test"""