import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def analyze_sentiment(text):
    df = pd.read_csv("twitter_sentiment.csv", on_bad_lines="skip")

    X = df["SentimentText"]
    y = df["Sentiment"]

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    new_text = [text]
    new_text_vectorized = vectorizer.transform(new_text)

    prediction = model.predict(new_text_vectorized)

    if prediction[0] == 1:
        return True
    else:
        return False
