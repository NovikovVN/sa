from collections import deque, namedtuple
from time import time

from flask import Flask, render_template, redirect, url_for, request
from sentiment_classifier import SentimentClassifier

app = Flask(__name__)

print('Preparing classifier...')
start_time = time()
classifier = SentimentClassifier()
print('Classifier is ready')
print(round(time() - start_time, 6), 'seconds\n')

Review = namedtuple('Review', 'positive negative comment tag')
reviews = deque()


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

    return redirect(url_for('main'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)