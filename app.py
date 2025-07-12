from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from utils import clean_text

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = load_model('models/sentiment_lstm.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Config
MAX_LEN = 100

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(pad)[0][0]
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    return render_template('result.html', review=review, sentiment=sentiment)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
