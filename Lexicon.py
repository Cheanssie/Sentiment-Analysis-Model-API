import nltk
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import contractions
import csv
import re
import pandas as pd
from googletrans import Translator

class LexiconAnalysis:
    def lower_casing(self, review):
        return review.lower()

    def expand_contraction(self, review):
        expanded_words = []
        for word in review.split():
            expanded_words.append(contractions.fix(word))
        return ' '.join(expanded_words)

    def pos_tagger(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def token_pos(self, tokenized_data):
        tokenized_data = word_tokenize(tokenized_data)
        tokenized_data = pos_tag(tokenized_data)

        tokenized_data = list(map(lambda x: (x[0], self.pos_tagger(x[1])), tokenized_data))
        return tokenized_data

    def lemmatize(self, data):
        wn_lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = ""
        for word, tag in data:
            if tag is None:
                lemmatized_sentence = lemmatized_sentence + " " + word
            else:
                lemma = wn_lemmatizer.lemmatize(word, tag)
                lemmatized_sentence = lemmatized_sentence + " " + lemma

        return lemmatized_sentence

    def remove_punctuation(self, review):
        PUNCT_TO_REMOVE = string.punctuation
        return review.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    def remove_stopwords(self, review):
        STOPWORDS = set(stopwords.words('english'))
        return " ".join([word for word in str(review).split() if word not in STOPWORDS])

    def analyze(self, text_review):
        SentiLexM = pd.read_csv('static/res/SentiLexM.txt', sep='\t', names=['Word', 'SentimentScore'])
        
        abbr = Abbreviation()
        abbr.load_abbreviations()
        abbreviated = abbr.replace_abbreviations(text_review)

        lowercased = self.lower_casing(review=abbreviated)
        expanded = self.expand_contraction(review=lowercased)
        rmvPunc = self.remove_punctuation(review=expanded)
        rmvStop = self.remove_stopwords(review=rmvPunc)
        tokenized = self.token_pos(tokenized_data=rmvStop)
        lemmatized = self.lemmatize(data=tokenized)
        
        tokens_analysis = word_tokenize(lemmatized)
        sentiment_score = 0.0
        matching_entries = SentiLexM.loc[SentiLexM['Word'].isin(tokens_analysis)]

        sentiment_score = matching_entries['SentimentScore'].sum()
        
        # Classify sentiment
        if sentiment_score > 0:
            return "Positive"
        elif sentiment_score < 0:
            return "Negative"
        else:
            return "Neutral"
        
    def analyzeMT(self, text_review):
        SentiLexM = pd.read_csv('static/res/SentiLexM.txt', sep='\t', names=['Word', 'SentimentScore'])
        
        abbr = Abbreviation()
        abbr.load_abbreviations()
        abbreviated = abbr.replace_abbreviations(text_review)

        translator = Translator()
        translated = translator.translate(abbreviated, dest='en').text

        lowercased = self.lower_casing(review=translated)
        expanded = self.expand_contraction(review=lowercased)
        rmvPunc = self.remove_punctuation(review=expanded)
        rmvStop = self.remove_stopwords(review=rmvPunc)
        tokenized = self.token_pos(tokenized_data=rmvStop)
        lemmatized = self.lemmatize(data=tokenized)
        
        tokens_analysis = word_tokenize(lemmatized)
        sentiment_score = 0.0
        matching_entries = SentiLexM.loc[SentiLexM['Word'].isin(tokens_analysis)]

        sentiment_score = matching_entries['SentimentScore'].sum()
        
        # Classify sentiment
        if sentiment_score > 0:
            return translated, "Positive"
        elif sentiment_score < 0:
            return translated, "Negative"
        else:
            return translated, "Neutral"

class Abbreviation:

    # Define the abbreviations as a global variable
    abbreviations = {}

    def load_abbreviations(self):
        with open("static/res/abbreviations.csv", mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                abbreviation, original = row
                self.abbreviations[abbreviation] = original

    def replace_abbreviations(self, text):
        # Regular expression to match whole words
        word_pattern = r'\b\w+\b'

        def replace(match):
            word = match.group(0)
            return self.abbreviations.get(word, word)

        # Use regex to find words and replace them
        processed_text = re.sub(word_pattern, replace, text)
        return processed_text


