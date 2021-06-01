from flask import Flask, request, render_template
import tensorflow as tf
from keras.models import load_model

import string
import re

from nltk.corpus import stopwords
# nltk.download('stopwords')
stop = set(stopwords.words("english"))

def remove_stopwords(text):
    text = [word for word in text.split() if word not in stop]

    return " ".join(text)

app = Flask(__name__)

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')  

model = tf.keras.models.load_model('dl_model/model_3epc_64bs')
# model.predict(['the book is amazing', 'the book is terrible'])

# text preprocessing
stop = set(stopwords.words("english"))

def remove_stopwords(text):
    text = [word for word in text.split() if word not in stop]

    return " ".join(text)

@app.route('/')
def home_page():
    return render_template('index.html')
    # return 'Hello World'

@app.route('/prediction',  methods=["POST"])
def display_prediction():

    text = request.form['userReview']

    prediction, confidence = get_prediction(text)

    return render_template('display.html', sentiment=prediction, confidence=confidence)


def get_prediction(text):

    text = remove_stopwords(text)

    prediction = model.predict([text])[0]
    print(prediction[0], text)

    pred = prediction[0]

    if ( pred > 0.05):
        sentiment = 'positive'
        confidence = round((pred - 0.05)/0.95, 10)
    else:
        sentiment = 'negative'
        confidence = round((0.05-pred)/0.05, 10)

    return sentiment, confidence


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)