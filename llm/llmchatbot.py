import re
import string
import requests
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import PyPDF2

nltk.download('stopwords')
nltk.download('punkt')

def fetch_text_from_pdf(pdf_path):

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            return text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]

    return filtered_tokens

corpus = []
books = [
    {"title" : "chaaNakyaNiti" , "pdf_link" : r"C:\Users\Eshwar\Downloads\llm\chaaNakyaNiti.pdf"},
    {"title" : "LeGuin" , "pdf_link" : r"C:\Users\Eshwar\Downloads\llm\LeGuin.pdf"  },
    
    {"title" : "wisdombook" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\book-of-wisdom-obooko.pdf"},
    {"title" : "humanelement" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\human-element-of-the-human-obooko.pdf"},
    {"title" : "The Art of War" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\The Art of War.pdf"},
   
    {"title" : "chaaNakyaNiti" , "pdf_link" : r"C:\Users\Eshwar\Downloads\llm\chaaNakyaNiti.pdf"},
    {"title" : "LeGuin" , "pdf_link" : r"C:\Users\Eshwar\Downloads\llm\LeGuin.pdf"  },
    
    {"title" : "wisdombook" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\book-of-wisdom-obooko.pdf"},
    {"title" : "humanelement" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\human-element-of-the-human-obooko.pdf"},
    {"title" : "The Art of War" , "pdf_link": r"C:\Users\Eshwar\Downloads\llm\The Art of War.pdf"}
]

for book in books :
  pdf_link = book["pdf_link"]
  text = fetch_text_from_pdf(pdf_link)
  preprocessed_text = preprocess_text(text)
  corpus.extend(preprocessed_text)


import tensorflow as tf
from keras.layers import Embedding , LSTM , Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

vocab_size = 1000
embedding_dim = 128
max_seq_length = 50
lstm_units = 256
output_units = vocab_size

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)


max_seq_length = 50
X = []
y = []


for i in range(len(sequences) - 1):
    if len(sequences[i + 1]) > 0:
        X.append(sequences[i])
        y.append(sequences[i + 1][0])

max_seq_length = 50
X = pad_sequences(X, maxlen=max_seq_length, padding='post', truncating='post')
y = tf.keras.utils.to_categorical(y, num_classes=1000)


split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]


model = Sequential([
    Embedding(1000, 128, input_length=max_seq_length),
    LSTM(256),
    Dense(1000, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
model.save('custom_llm_model.h5')

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# Initialize model and set last 6 layers trainable
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
for layer in model.layers[-6:]:
    layer.trainable = True

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Dummy data for example purposes (replace with actual data)
X_train = np.random.randint(0, 50257, size=(100, 50))  # Replace with your actual data
y_train = np.random.randint(0, 50257, size=(100, 50))  # Replace with your actual data
X_val = np.random.randint(0, 50257, size=(20, 50))  # Replace with your actual data
y_val = np.random.randint(0, 50257, size=(20, 50))  # Replace with your actual data

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llm")

# Load the fine-tuned model and tokenizer
model = TFGPT2LMHeadModel.from_pretrained("./fine_tuned_llm")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate text
input_text = "What is the meaning of life?"
input_ids = tokenizer.encode(input_text, return_tensors="tf")
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=5)

generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

# Calculate BLEU scores
reference_text = ["The meaning of life is to be happy."]
bleu_scores = [sentence_bleu([reference_text.split()], gen.split()) for gen in generated_text]

# Calculate perplexity
input_ids = tokenizer.encode(reference_text[0], return_tensors="tf")
logits = model(input_ids)[0]
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = cross_entropy(input_ids[:, :-1], logits[:, :-1, :])
perplexity = np.exp(loss.numpy())

# Output results
print("Generated Text:")
for text in generated_text:
    print(text)
print("BLEU Scores:", bleu_scores)
print("Perplexity:", perplexity)
