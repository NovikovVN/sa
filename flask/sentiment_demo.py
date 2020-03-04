import sqlite3
from datetime import datetime
from collections import deque, namedtuple
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, redirect, url_for, request
from time import time

from sentiment_classifier import SentimentClassifier

maxlen = 5

with sqlite3.connect('sa.db') as conn:
    cursor = conn.cursor()
    cursor.execute('create table if not exists sa(date text, positive text, negative text, comment text, tag text)')
    cursor.execute('SELECT * FROM sa ORDER BY date DESC LIMIT {}'.format(maxlen))
    rows = cursor.fetchall()
    conn.commit()

app = Flask(__name__)
run_with_ngrok(app)

print('Preparing classifier...')
start_time = time()
classifier = SentimentClassifier()
print('Classifier is ready')
print(round(time() - start_time, 6), 'seconds\n')

Review = namedtuple('Review', 'positive negative comment tag')
reviews = deque(maxlen=maxlen)
for row in rows:
    reviews.append(Review(*row[1:]))


@app.route('/', methods=['GET'])
def main():
    return render_template('add_review.html', reviews=reviews)


@app.route('/add_review', methods=['POST'])
def add_review():
    positive = request.form['positive']
    negative = request.form['negative']
    comment = request.form['comment']
    text = ' '.join([positive, negative, comment])
    tag = classifier.get_prediction_message(text)
    reviews.appendleft(Review(positive, negative, comment, tag))

    date = datetime.now()

    with sqlite3.connect('sa.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sa VALUES('{}', '{}', '{}', '{}', '{}')".format( \
                        date, positive, negative, comment, tag))
        conn.commit()

    with open('sa_log.log', 'a+', encoding='utf-8') as f:
        f.write('{}: TAG: {}\n'.format(date, tag))

    return redirect(url_for('main'))


if __name__ == '__main__':
    app.run()