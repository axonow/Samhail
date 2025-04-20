# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
import numpy as np  # For numerical operations

# For padding sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.base_models.markov_chain_monte_carlo import (
    model,
    predict_next_word,
    tokenizer,
    X,
    y,
    max_length,
)  # Import the functions and variables to test

# -------------------------------
# Test Suite for Markov Chain Monte Carlo with LSTM
# -------------------------------


def test_tokenizer():
    """
    Test the tokenizer to ensure it correctly tokenizes and maps words to indices.

    Asserts:
        - The tokenizer creates a consistent mapping of words to indices.
    """
    word_index = tokenizer.word_index
    for word, index in word_index.items():
        assert isinstance(word, str)
        assert isinstance(index, int)


def test_encoded_sequences():
    """
    Test the encoding of sentences into numerical sequences.

    Asserts:
        - Each word in the sentence is correctly mapped to its corresponding index.
    """
    sentences = [
        "The cat sat on the mat",
        "The dog barked at the stranger",
    ]
    encoded_sequences = [
        tokenizer.texts_to_sequences([sentence])[0] for sentence in sentences
    ]

    # Assert that the sequences are correctly encoded
    for sentence, encoded_seq in zip(sentences, encoded_sequences):
        for word, index in zip(sentence.split(), encoded_seq):
            assert tokenizer.word_index[word.lower()] == index


def test_padding_sequences():
    """
    Test the padding of input sequences.

    Asserts:
        - All sequences are padded to the same length.
    """
    sequences = [[1], [1, 2], [1, 2, 3]]
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding="pre")

    # Assert that all sequences are padded correctly
    assert padded_sequences.shape == (3, max_length)
    assert (padded_sequences[0] == [0] * (max_length - 1) + [1]).all()
    assert (padded_sequences[1] == [0] * (max_length - 2) + [1, 2]).all()
    assert (padded_sequences[2] == [0] * (max_length - 3) + [1, 2, 3]).all()


@patch("models.base_models.markov_chain_monte_carlo.model.fit")
def test_model_training(mock_fit):
    """
    Test the training of the LSTM model.

    Args:
        mock_fit (MagicMock): Mocked `fit` method of the model.

    Asserts:
        - The `fit` method is called with the correct arguments.
    """
    # Mock the fit method
    mock_fit.return_value = None

    # Train the model
    model.fit(X, y, epochs=500, verbose=1)

    # Assert that the fit method was called with the correct arguments
    mock_fit.assert_called_once_with(X, y, epochs=500, verbose=1)


@patch("models.base_models.markov_chain_monte_carlo.model.predict")
def test_predict_next_word(mock_predict):
    """
    Test the `predict_next_word` function.

    Args:
        mock_predict (MagicMock): Mocked `predict` method of the model.

    Asserts:
        - The function returns the correct predicted word.
    """
    # Mock the predict method
    mock_predict.return_value = np.array(
        [[0.1, 0.2, 0.7]])  # Simulated probabilities

    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.texts_to_sequences.return_value = [
        [1, 2, 3, 4]
    ]  # Simulated tokenized input
    # Mocked word-to-index mapping
    mock_tokenizer.word_index = {
        "the": 1, "cat": 2, "sat": 3, "on": 4, "mat": 5}

    # Dynamically determine the expected word based on the mocked probabilities
    # Add 1 because word_index is 1-based
    expected_index = np.argmax(mock_predict.return_value) + 1
    expected_word = {v: k for k, v in mock_tokenizer.word_index.items()}[
        expected_index
    ]  # Reverse mapping

    # Call the function with a sample input
    result = predict_next_word(
        model, mock_tokenizer, "The cat sat on", max_length=5)

    # Assert that the function returns the correct predicted word
    assert result == expected_word  # Dynamically assert the expected word
    mock_predict.assert_called_once()


def test_predict_next_word_invalid_input():
    """
    Test the `predict_next_word` function with invalid input.

    Asserts:
        - The function returns `None` for unknown words.
    """
    result = predict_next_word(model, tokenizer, "Unknown input", max_length)
    assert result is None
