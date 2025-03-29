# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
import numpy as np  # For numerical operations
from tensorflow.keras.utils import to_categorical  # For one-hot encoding
from models.base_models.lstm import predict_next_word  # Import the function to test

# -------------------------------
# Test Suite for LSTM Model
# -------------------------------

@patch("models.base_models.lstm.tokenizer")
@patch("models.base_models.lstm.model")
def test_predict_next_word(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word` function to ensure it correctly predicts the next word.

    Args:
        mock_model (MagicMock): Mocked LSTM model.
        mock_tokenizer (MagicMock): Mocked tokenizer.

    Steps:
    1. Mock the tokenizer to simulate tokenization and decoding behavior.
    2. Mock the model to simulate inference and return predictions.
    3. Call the `predict_next_word` function with a sample input.
    4. Assert that the function returns the expected predicted word.
    """

    # Mock the tokenizer's behavior
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]  # Simulated tokenized input
    mock_tokenizer.word_index = {"word1": 1, "word2": 2, "word3": 3, "word4": 4}  # Simulated vocabulary

    # Mock the model's behavior
    mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])  # Simulated prediction probabilities

    # Call the function with a sample input
    result = predict_next_word(mock_model, mock_tokenizer, "The cat sat on", max_length=5)

    # Assert that the function returns the expected predicted word
    assert result == "word4"  # Expected predicted word
    mock_tokenizer.texts_to_sequences.assert_called_once_with(["The cat sat on"])
    mock_model.predict.assert_called_once()

def test_predict_next_word_no_prediction():
    """
    Test the `predict_next_word` function when no prediction is found.

    Steps:
    1. Mock the tokenizer to simulate tokenization behavior.
    2. Call the `predict_next_word` function with a sample input.
    3. Assert that the function returns `None` when no word is found.
    """

    # Mock the tokenizer's behavior
    tokenizer = MagicMock()
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]  # Simulated tokenized input
    tokenizer.word_index = {"word1": 1, "word2": 2, "word3": 3}  # Simulated vocabulary (no word4)

    # Mock the model's behavior
    model = MagicMock()
    model.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])  # Simulated prediction probabilities

    # Call the function with a sample input
    result = predict_next_word(model, tokenizer, "The cat sat on", max_length=5)

    # Assert that the function returns `None`
    assert result is None

def test_padding_sequences():
    """
    Test the padding of sequences to ensure uniform length.

    Steps:
    1. Create a list of sequences with varying lengths.
    2. Pad the sequences to a fixed length.
    3. Assert that all sequences are padded correctly.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = [[1], [1, 2], [1, 2, 3]]
    max_length = 5
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="pre")

    # Assert that all sequences are padded correctly
    assert padded_sequences.shape == (3, max_length)
    assert (padded_sequences[0] == [0, 0, 0, 0, 1]).all()
    assert (padded_sequences[1] == [0, 0, 0, 1, 2]).all()
    assert (padded_sequences[2] == [0, 0, 1, 2, 3]).all()

def test_one_hot_encoding():
    """
    Test the one-hot encoding of output labels.

    Steps:
    1. Create a list of output labels.
    2. Convert the labels to one-hot encoded format.
    3. Assert that the one-hot encoding is correct.
    """
    labels = [1, 2, 3]
    num_classes = 4
    one_hot_labels = to_categorical(labels, num_classes=num_classes)

    # Assert that the one-hot encoding is correct
    assert one_hot_labels.shape == (3, num_classes)
    assert (one_hot_labels[0] == [0, 1, 0, 0]).all()
    assert (one_hot_labels[1] == [0, 0, 1, 0]).all()
    assert (one_hot_labels[2] == [0, 0, 0, 1]).all()
