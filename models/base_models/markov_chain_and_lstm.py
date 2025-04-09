# Import necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.text import Tokenizer

# -------------------------------
# Training Data Preparation
# -------------------------------

# Define a small dataset of sequences for training
# Each sequence is a list of words representing a simple sentence.
# This dataset is used to train the LSTM model to predict the next word in a sequence.
sequences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "cat", "jumped", "over", "the", "mat"],
    ["the", "dog", "barked", "at", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "tree"],
    ["the", "fish", "swam", "in", "the", "pond"],
    ["the", "cat", "played", "with", "the", "dog"],
    ["the", "dog", "chased", "the", "cat", "away"],
    ["the", "bird", "sang", "a", "beautiful", "song"],
    ["the", "fish", "jumped", "out", "of", "the", "water"],
    ["the", "tree", "grew", "tall", "and", "strong"]
]

# Tokenize the words in the dataset
# Create mappings between words and unique integer indices.
# word_to_index: Maps each unique word to a unique integer.
# index_to_word: Reverse mapping from indices to words.
word_to_index = {word: i for i, word in enumerate(set(sum(sequences, [])))}
index_to_word = {i: word for word, i in word_to_index.items()}

# Encode the sequences into integer representations using word_to_index.
# Each word in a sequence is replaced with its corresponding integer index.
encoded_sequences = [[word_to_index[word] for word in seq] for seq in sequences]

# Prepare input-output pairs for training using a sliding window approach
# X: Input sequences (up to the current word).
# y: Output sequences (the next word).
X = []
y = []
for seq in encoded_sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])  # Input is the sequence up to the current word
        y.append(seq[i])   # Output is the next word

# Pad the input sequences to ensure consistent input length
# Pads input sequences (X) with zeros at the beginning to make them the same length.
# Converts the output list (y) into a NumPy array for compatibility with TensorFlow.
X = pad_sequences(X, padding='pre')
y = np.array(y)

# -------------------------------
# Model Definition
# -------------------------------

# Define the LSTM-based model for text generation
# The model consists of the following layers:
# 1. Embedding: Converts word indices into dense vector representations.
# 2. LSTM: A recurrent layer with 100 units for learning sequential patterns.
# 3. Dense: Fully connected output layer with softmax activation for multi-class classification.
model = Sequential([
    Embedding(input_dim=len(word_to_index), output_dim=50),  # Embedding layer with 50-dimensional vectors
    LSTM(100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer with dropout
    Dense(len(word_to_index), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
# Loss: sparse_categorical_crossentropy for multi-class classification.
# Optimizer: Adam for adaptive learning rate optimization.
# Metrics: Accuracy to monitor training performance.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------------
# Learning Rate Scheduler
# -------------------------------

# Define a learning rate scheduler to dynamically adjust the learning rate during training
# The learning rate remains constant for the first 100 epochs and decreases by 1% after each subsequent epoch.
def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * 0.99

# Create a learning rate scheduler callback
lr_scheduler = LearningRateScheduler(scheduler)

# -------------------------------
# Model Training
# -------------------------------

# Reshape the output array (y) to match the LSTM output shape
# Adds an extra dimension to y to make it compatible with the model's output.
y = np.expand_dims(y, axis=-1)

# Train the model
# Trains the model for 500 epochs with the learning rate scheduler.
# The verbose parameter is set to 1 to display training progress.
model.fit(X, y, epochs=500, verbose=1, callbacks=[lr_scheduler])

# -------------------------------
# Prediction Function
# -------------------------------

# Define a function to predict the next word given a seed phrase
# The function performs the following steps:
# 1. Converts the input phrase into indices using word_to_index.
# 2. Pads the input to match the model's input shape.
# 3. Predicts the next word's probabilities using the trained model.
# 4. Finds the word with the highest probability (argmax) and returns it.
def predict_next_word(input_text, model, tokenizer, index_to_word, max_length):
    """
    Predicts the next word for a given input text using the LSTM model.

    Args:
        input_text (str): The input text for which the next word is predicted.
        model: The trained LSTM model.
        tokenizer: The tokenizer used for text preprocessing.
        index_to_word (dict): Mapping from indices to words.
        max_length (int): The maximum length of the input sequence.

    Returns:
        str: The predicted next word.
    """

    # Convert each subarray of words into a sentence
    sentences = [" ".join(subarray) for subarray in sequences]

    # Initialize the tokenizer and fit it on some sample text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([input_text])[0]

    # Pad the sequence to the required length
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding="pre")

    # Perform inference with the model
    prediction = model.predict(padded_sequence)

    # Get the index of the word with the highest probability
    predicted_index = np.argmax(prediction)

    # Map the index to the corresponding word
    return index_to_word.get(predicted_index, None)

# -------------------------------
# Example Predictions
# -------------------------------

# Demonstrate the model's ability to predict the next word for given seed phrases
# The expected output depends on the training data and the learned probabilities.
print("Predicted next word for 'the cat sat on':", predict_next_word("the cat sat on", model, word_to_index, index_to_word, max_length=5))
print("Predicted next word for 'the dog chased':", predict_next_word("the dog chased", model, word_to_index, index_to_word, max_length=5))
print("Predicted next word for 'the bird sang':", predict_next_word("the bird sang", model, word_to_index, index_to_word, max_length=5))
print("Predicted next word for 'the fish jumped':", predict_next_word("the fish jumped", model, word_to_index, index_to_word, max_length=5))

