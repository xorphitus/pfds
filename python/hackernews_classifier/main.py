import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

datalen = 200

X_train = []
y_train = []
X_test = []
y_test = []

with open('./dataset.tsv') as f:
    tsv = csv.reader(f, delimiter = '\t')

    i = 0
    for row in tsv:
      if i < datalen:
          y_train.append(row[0])
          X_train.append(row[1])
      else:
          y_test.append(row[0])
          X_test.append(row[1])
      i = i + 1

stop_words = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']

vectorizer = CountVectorizer(stop_words=stop_words)
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
print(X_train.shape)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))

# test
X_test = vectorizer.transform(X_test)
print(clf.score(X_test, y_test))
