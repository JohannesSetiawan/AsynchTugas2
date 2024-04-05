import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from googletrans import Translator
import os

def translate_en(df):
    df['text'] = df['text'].apply(transalate)
    return df

def transalate(text):
    translator = Translator()
    return translator.translate(text, des='en').text
     
def remove_html_tags(text):
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text

def remove_urls(text):
    clean_text = re.sub(r'http\S+', '', text)
    return clean_text

def convert_to_lowercase(text):
    return text.lower()

def replace_chat_words(text):
    chat_words = {
        "brb": "Be right back",
        "btw": "By the way",
        "omg": "Oh my God/goodness",
        "ttyl": "Talk to you later",
        "omw": "On my way",
        "smh/smdh": "Shaking my head/shaking my darn head",
        "lol": "Laugh out loud",
        "tbd": "To be determined", 
        "imho/imo": "In my humble opinion",
        "hmu": "Hit me up",
        "iirc": "If I remember correctly",
        "lmk": "Let me know", 
        "og": "Original gangsters (used for old friends)",
        "ftw": "For the win", 
        "nvm": "Nevermind",
        "ootd": "Outfit of the day", 
        "ngl": "Not gonna lie",
        "rq": "real quick", 
        "iykyk": "If you know, you know",
        "ong": "On god (I swear)", 
        "yaas": "Yes!", 
        "brt": "Be right there",
        "sm": "So much",
        "ig": "I guess",
        "wya": "Where you at",
        "istg": "I swear to god",
        "hbu": "How about you",
        "atm": "At the moment",
        "asap": "As soon as possible",
        "fyi": "For your information"
    }
    for word, expanded_form in chat_words.items():
        text = text.replace(word, expanded_form)
    return text

def remove_punctuation(text):
    clean_text = ''.join(ch for ch in text if ch not in string.punctuation)
    return clean_text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_whitespace(text):
    return text.strip()

def remove_special_characters(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = convert_to_lowercase(text)
    text = replace_chat_words(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_whitespace(text)
    text = remove_special_characters(text)
    return text

def sentiment_analysis(input_txt):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(current_directory, 'sentiment.csv')
    df=pd.read_csv(csv_file_path)

    # Preprocessing
    df['text'] = df['text'].apply(preprocess_text)
    df.head()

    # Training
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Prediction
    # (0 = negative, 2 = neutral, 4 = positive)
    df_input = pd.DataFrame({'text': input_txt})
    df_input = translate_en(df_input)
    df_input['text'] = df_input['text'].apply(preprocess_text)
    X_input = vectorizer.transform(df_input['text'])
    y_pred = logreg.predict(X_input)
    res = []
    for i in range(len(input_txt)):
        if y_pred[i] == 4:
            res_txt = "Positive"
        else:
            res_txt = "Negative"
        res.append({"text": input_txt[i], "result": res_txt})
    return res

def spam_analysis(input_txt):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(current_directory, 'spam.csv')
    df=pd.read_csv(csv_file_path, encoding='latin1')

    df.columns=['target','text','null1','null2','null3']
    df = df.drop(columns=['null1','null2','null3'])
    df['target'] = df['target'].replace({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(preprocess_text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(classification_report(y_test, y_pred))

    df_input = pd.DataFrame({'text': input_txt})
    df_input = translate_en(df_input)
    df_input['text'] = df_input['text'].apply(preprocess_text)
    X_input = vectorizer.transform(df_input['text'])
    y_pred = logreg.predict(X_input)
    res = []
    for i in range(len(input_txt)):
        if y_pred[i] == 1:
            res_txt = "Spam"
        else:
            res_txt = "Not Spam"
        res.append({"text": input_txt[i], "result": res_txt})
    return res

def hate_speech_offensive_languange_analysis(input_txt):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(current_directory, 'hate_speech.csv')
    df=pd.read_csv(csv_file_path)

    df.columns=['id','count','hate_speech','offensive_language','neither', 'target','text']
    df = df.drop(columns=['id','count','hate_speech','offensive_language','neither'])
    df['text'] = df['text'].apply(preprocess_text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(classification_report(y_test, y_pred))

    df_input = pd.DataFrame({'text': input_txt})
    df_input = translate_en(df_input)
    df_input['text'] = df_input['text'].apply(preprocess_text)
    X_input = vectorizer.transform(df_input['text'])
    y_pred = logreg.predict(X_input)
    res = []
    for i in range(len(input_txt)):
        if y_pred[i] == 0:
            res_txt = "Hate Speech"
        elif y_pred[i] == 1:
            res_txt = "Offensive Languange"
        else:
            res_txt = "Neutral"
        res.append({"text": input_txt[i], "result": res_txt})
    return res

def perform_analysis(input_text, analyze_sent, analyze_spam, analyze_speech):
    res_sentiment = ["Not analyzed"]
    res_spam = ["Not analyzed"]
    res_hate = ["Not analyzed"]

    if analyze_sent:
        res_sentiment = sentiment_analysis(input_text)

    if analyze_spam:
        res_spam = spam_analysis(input_text)

    if analyze_speech:
        res_hate = hate_speech_offensive_languange_analysis(input_text)

    result = {
        "sentiment" : res_sentiment,
        "spam" : res_spam,
        "hate_speech" : res_hate,
    }
    return result
