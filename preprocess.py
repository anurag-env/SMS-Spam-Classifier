import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        word for word in tokens
        if word.isalnum()
    ]
    tokens = [
        word for word in tokens
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]
    tokens = [
        ps.stem(word)
        for word in tokens
    ]
    return " ".join(tokens)