# CS 412 Project
# Adam Beigel

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# download the dataset from here:
# https://www.kaggle.com/datasets/rmisra/news-category-dataset
# name it 'news_category_dataset.json' and move it to the working directory

# also download this dataset:
# https://www.kaggle.com/datasets/rtatman/english-word-frequency 
# name it 'unigram_freq.csv' and move it to the working directory

# import to pandas dataframe
df = pd.read_json("news_category_dataset.json", lines=True)
df = df.sample(frac=0.8, random_state=1)
df.head()



# combine headline and short_description into one column, drop everything else
df['text'] = df['headline'] + ' ' + df['short_description']
df = df.drop(['link', 'date', 'authors', 'headline', 'short_description'], axis=1)

print("Before:", df['text'].head(1))

# to lowercase
df['text'] = df['text'].str.lower()

# remove punctuation and numbers
df['text'] = df['text'].str.replace('[{}]'.format(string.punctuation), '')
df['text'] = df['text'].str.replace('[\d]', '')

print("After:", df['text'].head(1))



# process the labels:
df = df.rename(columns={"category": "label"})

# combine ARTS, ARTS & CULTURE, and CULTURE & ARTS
df.loc[df["label"] == "ARTS", "label"] = "ARTS & CULTURE"
df.loc[df["label"] == "CULTURE & ARTS", "label"] = "ARTS & CULTURE"

# combine PARENTS and PARENTING
df.loc[df["label"] == "PARENTS", "label"] = "PARENTING"

# combine STYLE and STYLE & BEAUTY
df.loc[df["label"] == "STYLE", "label"] = "STYLE & BEAUTY"

# lets keep only more distinct categories
keep = ('ARTS & CULTURE', 'BUSINESS', 'CRIME', 'ENTERTAINMENT', 'FOOD & DRINK',
        'HOME & LIVING', 'MEDIA', 'MONEY', 'PARENTING', 'POLITICS', 'RELIGION',
        'SCIENCE', 'SPORTS', 'STYLE & BEAUTY', 'TECH', 'TRAVEL',  'WELLNESS',
        'WOMEN', 'WORLD NEWS')
df = df[df['label'].isin(keep)]

c = pd.Categorical(df['label'])

# change labels from text to numbers 0 - 18
df['label'] = df.label.map({category:i for (i, category) in enumerate(c.categories)})
print(enumerate(c.categories))



# creating a fixed bag of words
# import 10000 most common english words
english_words = pd.read_csv("unigram_freq.csv")
english_words = english_words[:10000].drop(['count'], axis=1)
english_words['word'] = english_words['word'].str.replace('[{}]'.format(string.punctuation), '')
english_words = np.asarray(english_words['word'].to_numpy(np.str_))

# create stop words array
stop_words = list(text.ENGLISH_STOP_WORDS)
stop_words = np.array(stop_words)



# creating a CountVectorizer object using the bag of words
def fitVectorizer(featureModel):
    vect = CountVectorizer()
    vect.fit(featureModel)
    return vect



# splitting the data and transforming features
def splitTransform(X, y, t, vect):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t, random_state=1)
    X_train = vect.transform(X_train).toarray()
    X_test = vect.transform(X_test).toarray()
    return X_train, X_test, y_train, y_test


# prep the experiment:
# can change num features or whether or not to include stop words
num_features = 800
# fitSet = np.setdiff1d(english_words[0:f], stop_words)  # without stop words
fitSet = english_words[:num_features]  # with stop words
vectorizer = fitVectorizer(english_words[:num_features])
print(f'number of features: {len(vectorizer.get_feature_names_out())}')

# Create training and test split
X_train, X_test, y_train, y_test = splitTransform(df['text'], df['label'], 0.8, vectorizer)



# Multinomial Naive Bayes model
mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)

# compute accuracy metrics
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('\nAccuracy Metrics for Naive Bayes:')
print(f'Accuracy Score: {acc:.3f}')
print(f'Balanced Accuracy Score: {bal_acc:.3f}')
print(f'Weighted F1 Score: {f1:.3f}\n')



# Logistic Regression model:
lr = LogisticRegression(random_state=1, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# compute accuracy metrics
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('\nAccuracy Metrics for Logistic Regression:')
print(f'Accuracy Score: {acc:.3f}')
print(f'Balanced Accuracy Score: {bal_acc:.3f}')
print(f'Weighted F1 Score: {f1:.3f}\n')



# Stochastic Gradient Descent with log-loss model
sgd = SGDClassifier(loss='log', random_state=1, max_iter=100)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

# compute accuracy metrics
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('\nAccuracy Metrics for Stochastic Gradient Descent with log-loss:')
print(f'Accuracy Score: {acc:.3f}')
print(f'Balanced Accuracy Score: {bal_acc:.3f}')
print(f'Weighted F1 Score: {f1:.3f}\n')
