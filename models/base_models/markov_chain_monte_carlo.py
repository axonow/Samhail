# Import necessary libraries
import numpy as np  # For numerical operations
import tensorflow as tf  # For deep learning operations
from tensorflow.keras.preprocessing.text import Tokenizer  # For text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
from tensorflow.keras.models import Sequential  # For building the LSTM model
from tensorflow.keras.layers import LSTM, Embedding, Dense  # For defining layers of the model

# -------------------------------
# Sample Text Dataset
# -------------------------------

# Define a small dataset of sentences for training
# Each sentence is a simple English sentence used to train the LSTM model to predict the next word.
sentences = [
    "The cat sat on the mat",
    "The dog barked at the stranger",
    "A bird is flying in the sky",
    "The sun is shining brightly",
    "A cat is playing with a ball"
]

# -------------------------------
# Tokenization
# -------------------------------

# Initialize the Tokenizer
# The Tokenizer converts words into numerical indices for processing by the model.
tokenizer = Tokenizer()

# Fit the Tokenizer on the dataset
# This step creates a vocabulary of unique words and assigns each word a unique integer index.
tokenizer.fit_on_texts(sentences)

# -------------------------------
# Convert Sentences to Numerical Sequences
# -------------------------------

# Initialize an empty list to store sequences
sequences = []

# Convert each sentence into a sequence of numerical indices
# For each sentence, generate input-output pairs using a sliding window approach.
for sentence in sentences:
    tokens = tokenizer.texts_to_sequences([sentence])[0]  # Convert words to indices
    for i in range(1, len(tokens)):
        sequences.append(tokens[:i+1])  # Append sequences up to the current word

# -------------------------------
# Padding Sequences
# -------------------------------

# Determine the maximum sequence length
# This ensures all sequences are padded to the same length for uniformity.
max_length = max(len(seq) for seq in sequences)

# Pad sequences with zeros at the beginning to make them uniform in length
# Padding ensures that all input sequences have the same length, which is required by the LSTM model.
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# -------------------------------
# Split Data into Input (X) and Output (y)
# -------------------------------

# Split the sequences into input (X) and output (y)
# X: All words except the last word in each sequence
# y: The last word in each sequence (the word to be predicted)
X, y = sequences[:, :-1], sequences[:, -1]

# Convert the output (y) into one-hot encoded format
# This step converts the output into a categorical format for multi-class classification.
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# -------------------------------
# Define the LSTM Model
# -------------------------------

# Create a Sequential model with the following layers:
# 1. Embedding Layer: Converts word indices into dense vector representations.
# 2. LSTM Layer 1: A recurrent layer with 100 units, returning sequences for the next LSTM layer.
# 3. LSTM Layer 2: A recurrent layer with 100 units, processing the sequences from the first LSTM layer.
# 4. Dense Layer 1: A fully connected layer with 100 units and ReLU activation.
# 5. Dense Layer 2: The output layer with softmax activation for multi-class classification.
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_length-1),  # Embedding layer
    LSTM(100, return_sequences=True),  # First LSTM layer
    LSTM(100),  # Second LSTM layer
    Dense(100, activation='relu'),  # Fully connected layer with ReLU activation
    Dense(len(tokenizer.word_index) + 1, activation='softmax')  # Output layer with softmax activation
])

# -------------------------------
# Compile the Model
# -------------------------------

# Compile the model with the following configurations:
# - Loss Function: Categorical crossentropy for multi-class classification.
# - Optimizer: Adam optimizer for adaptive learning rate optimization.
# - Metrics: Accuracy to monitor the training performance.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -------------------------------
# Train the Model
# -------------------------------

# Train the model on the dataset
# - X: Input sequences
# - y: One-hot encoded output labels
# - Epochs: Number of iterations over the entire dataset (500 epochs in this case)
# - Verbose: Displays training progress
model.fit(X, y, epochs=500, verbose=1)

# -------------------------------
# Prediction Function
# -------------------------------

# Define a function to predict the next word given a seed phrase
# The function performs the following steps:
# 1. Converts the input text into a sequence of numerical indices using the tokenizer.
# 2. Pads the sequence to match the input length of the model.
# 3. Predicts the probabilities of the next word using the trained model.
# 4. Finds the word with the highest probability and returns it.
def predict_next_word(model, tokenizer, text, max_length):
    """
    Predicts the next word for a given input text using the model.

    Args:
        model: The trained model.
        tokenizer: The tokenizer used for text preprocessing.
        text (str): The input text for which the next word is predicted.
        max_length (int): The maximum length of the input sequence.

    Returns:
        str: The predicted next word, or None if no word is found.
    """
    # Convert the input text into a sequence of indices
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Return None if the sequence is empty (unknown input)
    if not sequence:
        return None

    # Pad the sequence to match the model's input length
    sequence = pad_sequences([sequence], maxlen=max_length - 1, padding="pre")

    # Predict the probabilities of the next word
    predicted_index = np.argmax(model.predict(sequence), axis=-1)[0] + 1  # Adjust for 1-based indexing

    # Find the word corresponding to the predicted index
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word  # Return the predicted word
    return None  # Return None if no word is found

# -------------------------------
# Example Prediction
# -------------------------------

# Define an input text for prediction
input_text = "The cat sat on"

# Predict the next word for the input text
predicted_word = predict_next_word(model, tokenizer, input_text, max_length)

# Print the predicted word
print(f"Next word prediction: {predicted_word}")
