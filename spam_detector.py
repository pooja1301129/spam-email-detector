import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("spam.csv")
X = data["text"]
y = data["label"]
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vectorized, y)
email = input("Enter Email Text: ")
email_vectorized = vectorizer.transform([email])
prediction = model.predict(email_vectorized)
if prediction[0] == "spam":
    print("\n🚨 Spam Email")
else:
    print("\n✅ Not Spam")