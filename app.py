from flask import Flask, jsonify, request
from Lexicon import LexiconAnalysis
from ML import MLAnalysis
from config import *

app = Flask(__name__)

bucket = custombucket
region = customregion

#Define API access secret key for authorized access
app.config['SECRET_PASSCODE'] = 'SentimentAPIfyp2023'

@app.route("/")
def index():
    return "API Success"

@app.route("/lexiconMT/<auth>")
def lexiconMT(auth):
    #Obtain input via GET
    text = request.args.get('text')

    #Perform authentication
    if auth == app.config['SECRET_PASSCODE']:
        #Perform SA using SentiLexM + Machine Translation
        analyzer = LexiconAnalysis()
        translated, sentiment = analyzer.analyzeMT(text)
        
        #Return success user data
        user_data = {
            'Accessed Module' : 'Lexicon-based Sentiment Analysis with Machine Translation',
            'Response Code': 200,
            'Description' : 'Successful',
            'Input Text' : text,
            'Translated Text' : translated,
            'Sentiment' : sentiment
        }
        return jsonify(user_data), 200, {'Content-Type': 'application/json'}
    else:
        #Return access failure
        user_data = {
            'Accessed Module' : 'Lexicon-based Sentiment Analysis with Machine Translation',
            'Response Code': 203,
            'Description' : 'Authentication failure due to invalid passcode.',
            'Input Text' : text
        }
        return jsonify(user_data), 203, {'Content-Type': 'application/json'}

@app.route("/machineLearningMT/<auth>")
def machineLearningMT(auth):
    #Obtain input via GET
    text = request.args.get('text')

    #Perform authentication
    if auth == app.config['SECRET_PASSCODE']:
        #Perform SA using SentiLexM + Machine Translation
        analyzer = MLAnalysis()
        translated, sentiment = analyzer.analyzeMT(text)
        
        #Return success user data
        user_data = {
            'Accessed Module' : 'ML-based Sentiment Analysis with Machine Translation',
            'Response Code': 200,
            'Description' : 'Successful',
            'Input Text' : text,
            'Translated Text' : translated,
            'Sentiment' : sentiment
        }
        return jsonify(user_data), 200, {'Content-Type': 'application/json'}
    else:
        #Return access failure
        user_data = {
            'Accessed Module' : 'Lexicon-based Sentiment Analysis with Machine Translation',
            'Response Code': 203,
            'Description' : 'Authentication failure due to invalid passcode.',
            'Input Text' : text
        }
        return jsonify(user_data), 203, {'Content-Type': 'application/json'}
