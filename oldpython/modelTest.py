#!/usr/bin/python3

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

tweets = ["Hello world, today is a good day",
          "Bye, bye, world, I am sleeping",
          "Hello bikey, it is bleh",
          "Good bye popa, window",
          "Maybe now I will say hello",
          "Tomorrow I will do bye",
          "It is a good night for be hello",
          "Perhaps bye will be okay"]

tokTweets = [nltk.word_tokenize(tweet) for tweet in tweets]
stances = ['yes','no','yes','no','yes','no','yes','no']

stringTweets = [str(tweet) for tweet in tokTweets]


X_train, X_test, y_train, y_test = train_test_split(stringTweets, stances, test_size=0.33, random_state=2)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

tf = TfidfVectorizer(max_features=5000)
tf.fit(stringTweets)
X_train_tf = tf.transform(X_train)
X_test_tf = tf.transform(X_test)

svm = LinearSVC()
svm.fit(X_train_tf, y_train)
predictions = svm.predict(X_test_tf)

print("SVM Accuracy score: {0}".format(accuracy_score(predictions, y_test)*100))