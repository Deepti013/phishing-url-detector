import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import gradio as gr

df = pd.read_csv(r'C:\Users\Anshuman\Desktop\project\phishing_site_urls.csv')

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
lemmatizer = WordNetLemmatizer()

df['TextTokenize'] = df.URL.map(lambda t: tokenizer.tokenize(str(t)))
df['TextStemmed'] = df['TextTokenize'].map(
    lambda tokens: [lemmatizer.lemmatize(word.lower()) for word in tokens]
)
df['Text'] = df['TextStemmed'].map(lambda tokens: ' '.join(tokens))

cv = Word2Vec(df['TextTokenize'].tolist(), vector_size=100, window=5, min_count=1, workers=4)

features = np.array([
    np.mean([cv.wv[word] for word in tokens if word in cv.wv] or [np.zeros(100)], axis=0)
    for tokens in df['TextTokenize']
])

x_train, x_test, y_train, y_test = train_test_split(
    features, df.Label, test_size=0.2, random_state=42
)

l_model = LogisticRegression(max_iter=1000)
l_model.fit(x_train, y_train)

print("\nClassification Report\n")
print(classification_report(l_model.predict(x_test), y_test))

cm = confusion_matrix(l_model.predict(x_test), y_test)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['Text'])

x_train_nb, x_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_tfidf, df.Label, test_size=0.2, random_state=42
)

mnb = MultinomialNB()
mnb.fit(x_train_nb, y_train_nb)

pickle.dump(l_model, open("logistic_model.pkl", "wb"))
pickle.dump(cv, open("cv.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Models saved successfully")

model = pickle.load(open("logistic_model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

def to_vec(url):
    tokens = tokenizer.tokenize(url)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    vectors = [cv.wv[w] for w in tokens if w in cv.wv]
    if not vectors:
        return np.zeros((1, cv.vector_size))
    return np.mean(vectors, axis=0).reshape(1, -1)

def predict_url(url):
    result = model.predict(to_vec(url))[0]
    return "⚠️ Phishing Website" if result == "bad" else "✅ Safe Website"

gr.Interface(
    fn=predict_url,
    inputs=gr.Textbox(label="Enter URL"),
    outputs=gr.Textbox(label="Result"),
    title="Phishing Website Detection",
    description="Check whether a URL is Safe or Phishing"
).launch()

