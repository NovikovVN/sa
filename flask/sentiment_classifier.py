__author__ = 'xead'
from pickle import load


class SentimentClassifier(object):
    def __init__(self):
        with open('classifier.pickle', 'rb') as f:
            self.classifier = load(f)
        with open('vectorizer.pickle', 'rb') as f:
            self.vectorizer = load(f)
        self.classes_dict = {0: 'отрицательный', 1: 'положительный', -1: 'неопределенный'}
        self.features_set = set(self.vectorizer.get_feature_names())

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return 'нейтрально или неопределенно'
        if probability < 0.7:
            return 'вероятно'
        if probability > 0.95:
            return 'определенно'
        else:
            return ''

    def predict_text(self, text):
        try:
            text_set = set(text.lower().split())
            if len(text_set) < 5 or len(text_set.intersection(self.features_set)) < 5:
                print('ошибка предсказания')
                return -1, 0.8
            vectorized = self.vectorizer.transform([text])
            return self.classifier.predict(vectorized)[0], \
                   self.classifier.predict_proba(vectorized)[0].max()
        except:
            print('ошибка предсказания')
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.classifier.predict(vectorized),\
                   self.classifier.predict_proba(vectorized)
        except:
            print('ошибка предсказания')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + ' ' + self.classes_dict[class_prediction]