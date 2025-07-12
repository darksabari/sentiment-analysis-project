import re

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+","",text)
    text = re.sub(r"[^a-z\s]" , "",text)
    tokens = text.split()

    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

