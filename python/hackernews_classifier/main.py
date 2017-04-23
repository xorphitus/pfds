# pip install numpy scipy scikit-learn

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

X_train_vec = vectorizer.transform(X_train)
print(X_train_vec.shape)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
print(clf.score(X_train_vec ,y_train))

# test
X_test_vec = vectorizer.transform(X_test)
print(clf.score(X_test_vec, y_test))

print('-----')
for i in range(0, 100):
    print(X_test[i])
    print(clf.predict(X_test_vec[i:i+1]))
