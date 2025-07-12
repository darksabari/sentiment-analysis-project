import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import clean_text

df = pd.read_csv('data/reviews.csv')
df['review'] = df['review'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['class'],
    test_size=0.2,
    stratify=df['class'],
    random_state=42
)


MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS , oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

train_seq = tokenizer.texts_to_sequences(X_train)
train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')

test_seq = tokenizer.texts_to_sequences(X_test)
test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post')


model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
    LSTM(64, dropout=0.2 , recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Train
history = model.fit(
    train_pad,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(test_pad, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save('models/sentiment_lstm.h5')

# Save tokenizer
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved.")


def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(pad)[0][0]
    return "Positive" if prob >= 0.5 else "Negative"

# Example
sample = "I love this product, it works great!"
print("Sample prediction:", predict_sentiment(sample))

