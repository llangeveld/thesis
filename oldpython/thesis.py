#!/usr/bin/python3

# ----- ALL IMPORTS ------
import csv
import re
import spacy

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


def pre_processMT(dataMT):
    nlp = spacy.load('en')
    genders = []
    stances = []
    strTweets = []

    for _, tweet, gender, stance, in dataMT:
        noURLS = re.sub(r"http\S+", "", tweet)
        noHashtags = re.sub(r"#", "", noURLS)
        tweetText = nlp(noHashtags.lower())
        genders.append(gender)
        stances.append(stance)
        lemmas = []
        for token in tweetText:
            if token.text not in stopwords.words('english') and token.text.isalpha():
                lemmas.append(token.lemma_)
        strTweets.append(str(lemmas))

    return strTweets, genders, stances


def pre_processF(dataF):
    nlp = spacy.load('en')
    stances = []
    strTweets = []

    for line in dataF:
        noURLS = re.sub(r"http\S+", "", line[0])
        noHashtags = re.sub(r"#", "", noURLS)
        tweetText = nlp(noHashtags.lower())
        stances.append(line[1])
        lemmas = []
        for token in tweetText:
            if token.text not in stopwords.words('english') and token.text.isalpha():
                lemmas.append(token.lemma_)
        strTweets.append(str(lemmas))

    return strTweets, stances


def import_data():
    with open("resources/finalData.csv") as f:
        f1 = [line for line in csv.reader(f, delimiter=",")]
    with open("resources/feminismStance.csv") as f2:
        f3 = [line for line in csv.reader(f2, delimiter=",")]
    return f1, f3


def taskA_gender(strTweets, genders, stances):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(strTweets)
    genders2 = le.fit_transform(genders)

    finalList = [str([tweet, gender]) for tweet, gender in zip(strTweets, genders2)]
    
    X_train, X_test, y_train, y_test = train_test_split(finalList, stances, test_size=0.3, random_state=0)
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("Task A w/ gender: {}".format(run_tests(X_train, X_test, y_train, y_test)))


def taskA(strTweets, stances, X_train, X_test, y_train, y_test):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(strTweets)

    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    X_test = vectorizer.transform(X_test)
    X_train = vectorizer.transform(X_train)

    print("Task A w/o gender: {}".format(run_tests(X_train, X_test, y_train, y_test)))


def taskB(strTweetsMT, strTweetsF, stancesF, X_test, y_test):
    le = preprocessing.LabelEncoder()
    count_word = TfidfVectorizer(ngram_range=(1,3))
    count_char = TfidfVectorizer(analyzer='char', ngram_range=(4,4))
    vectorizer = FeatureUnion([('word', count_word), ('char', count_char)])
    vectorizer.fit(strTweetsMT + strTweetsF)

    X_train = vectorizer.transform(strTweetsF)
    y_train = le.fit_transform(stancesF)
    X_test = vectorizer.transform(X_test)
    y_test = le.fit_transform(y_test)

    print("Task B: {}".format(run_tests(X_train, X_test, y_train, y_test)))


def get_data(strTweetsMT, genders, stancesMT, strTweetsF, stancesF):
    taskA_gender(strTweetsMT, genders, stancesMT)
    X_train, X_test, y_train, y_test = train_test_split(strTweetsMT, stancesMT, test_size=0.3, random_state=0)
    taskA(strTweetsMT, stancesMT, X_train, X_test, y_train, y_test)
    taskB(strTweetsMT, strTweetsF, stancesF, X_test, y_test)


def run_tests(X_train, X_test, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')*100
    return f1


def do_analysis(dataMT, dataF):
    strTweetsMT, genders, stancesMT = pre_processMT(dataMT)
    strTweetsF, stancesF = pre_processF(dataF)
    get_data(strTweetsMT, genders, stancesMT, strTweetsF, stancesF)


def main():
    dataMT, dataF = import_data()  # [tweetID, tweetText, gender, stance]
    do_analysis(dataMT, dataF)


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
