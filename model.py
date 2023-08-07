import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib

df = pd.read_csv("Names_Dataset.csv")

vec = CountVectorizer()
X= vec.fit_transform(df['names'])
ylabel = df['nationality']

# feature extraction train test split
x_train, x_test, y_train, y_test = train_test_split(X,ylabel,test_size=0.30)

nb = MultinomialNB()
nb.fit(x_train, y_train)
score = nb.score(x_test, y_test)

print('Accuracy of Naive Bayes classifier on training set: {:.2f}', score)


sample1 = ["Yin","Bathsheba","Brittany","Vladmir"]
vec1 = vec.transform(sample1).toarray()
print(nb.predict(vec1))
nationality_predictor = open('nationality_predictor.pkl','wb')
joblib.dump(nb, nationality_predictor)
nationality_predictor.close()
