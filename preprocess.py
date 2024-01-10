import re
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocessing(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), '' , text)
    text = re.sub('[^A-Za-z0-9]+' ,' ', text)
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text) # remove extra whitespaces
    text = remove_stopwords(text)
    text = text.lower()
    text = lemmatize_words(text)

    return text
