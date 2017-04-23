import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

X_train = []
y_train = []

with open('./dataset.tsv') as f:
    tsv = csv.reader(f, delimiter = '\t')

    for row in tsv:
      y_train.append(row[0])
      X_train.append(row[1])

stop_words = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')', '*', '--', '\\']

vectorizer = CountVectorizer(stop_words=stop_words)
vectorizer.fit(X_train)

X_train = vectorizer.transform(X_train)
print(X_train.shape)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))
