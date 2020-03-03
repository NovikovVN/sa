import pickle
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as soup
from glob import glob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def parse_price_ru_html(f):
    parser = soup(f, 'html.parser')

    reviews = parser.find_all('div', {'class': 'grid_21 push_1 reviews__text'})
    data_text = []
    for review in reviews:
        data_text.append(review.text.strip())

    ratings = parser.find_all('meta', {'itemprop': 'ratingValue'})
    data_target = []
    for rating in ratings:
        rating_value = rating.attrs["content"]
        data_target.append(1 if int(rating_value) >= 5 else 0)

    df = pd.DataFrame({'text': data_text, 'target': data_target}, columns=['text', 'target'])

    return df


train_data = pd.DataFrame(columns=['text', 'target'])
for file_name in glob(os.path.join('html', '*.html')):
    with open(file_name, 'r', encoding='utf-8') as f:
        train_data = train_data.append(parse_price_ru_html(f), ignore_index=True)

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), stop_words=stopwords.words('russian'))
classifier = LogisticRegression(multi_class='ovr', n_jobs=-1, random_state=45, class_weight='balanced')

train_data_vectorized = vectorizer.fit_transform(train_data.text)
classifier.fit(train_data_vectorized, train_data.target.astype(int))

with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
