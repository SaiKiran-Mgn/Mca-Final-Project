from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import re
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle5 as pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
import sqlite3

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def init():
    global model,graph
    graph = tf.Graph()
    

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")


@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/sentiment_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():

    if request.method=='POST':
        text = request.form['text']

        tw = tokenizer.texts_to_sequences([text])
        tw = sequence.pad_sequences(tw,maxlen=200)

        with graph.as_default():
            # load the pre-trained Keras model
            model = load_model('sentiment.h5')

            probability = model.predict(tw)[0][0]
            print(probability)
            prediction = int(model.predict(tw).round().item())
            
        if prediction == 0:
            sentiment = 'Negative'
            stop_words = stopwords.words('english')
            text1 = request.form['text'].lower()
            text_final = ''.join(c for c in text1 if not c.isdigit())   
            processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
            sa = SentimentIntensityAnalyzer()
            dd = sa.polarity_scores(text=processed_doc1)
            dd = sa.polarity_scores(text=processed_doc1)
            #compound = round((1 + dd['compound'])/2, 2)
            text5=dd['neg']*100
            if text5 >= 10 and text5 <= 30:
                result = 'Anxiety Emotional State'
            elif text5 >= 31 and text5 <= 50:
                result = 'Frustration Emotional State'
            elif text5 >= 51 and text5 <= 70:
                result = 'Resentment Emotional State'
            elif text5 >= 71:
                result = 'Anger Emotional State'
            
        else:
            sentiment = 'Positive'
            stop_words = stopwords.words('english')
            text1 = request.form['text'].lower()
            text_final = ''.join(c for c in text1 if not c.isdigit())   
            processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])
            sa = SentimentIntensityAnalyzer()
            dd = sa.polarity_scores(text=processed_doc1)
            dd = sa.polarity_scores(text=processed_doc1)
            #compound = round((1 + dd['compound'])/2, 2)
            text5=dd['pos']*100
            if text5 >= 10 and text5 <= 30:
                result = 'Interest Emotional State'
            elif text5 >= 31 and text5 <= 50:
                result = 'Gratitude Emotional State'
            elif text5 >= 51 and text5 <= 70:
                result = 'Inspiration and awe Emotional State'
            elif text5 >= 71:
                result = 'Love Emotional State'
            
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability, emotional = result)


if __name__ == '__main__':
    init()
    app.run(debug=True)


