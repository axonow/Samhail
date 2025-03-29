# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
import numpy as np  # For numerical operations
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
from models.base_models.markov_chain_and_lstm import (
    model,
    predict_next_word,
    word_to_index,
    index_to_word,
    X,
    y,
)  # Import the functions and variables to test

# -------------------------------
# Test Suite for Markov Chain and LSTM Model
# -------------------------------

def test_word_to_index_and_index_to_word():
    """
    Test the `word_to_index` and `index_to_word` mappings.

    Asserts:
        - The mappings are consistent and reversible.
    """
    # Ensure that word_to_index and index_to_word are consistent
    for word, index in word_to_index.items():
        assert index_to_word[index] == word

def test_encoded_sequences():
    """
    Test the encoding of sequences into numerical representations.

    Asserts:
        - Each word in the sequence is correctly mapped to its corresponding index.
    """
    sequences = [
        ["the", "cat", "sat"],
        ["the", "dog", "barked"],
    ]
    encoded_sequences = [[word_to_index[word] for word in seq] for seq in sequences]

    # Assert that the sequences are correctly encoded
    for seq, encoded_seq in zip(sequences, encoded_sequences):
        for word, index in zip(seq, encoded_seq):
            assert word_to_index[word] == index

def test_padding_sequences():
    """
    Test the padding of input sequences.

    Asserts:
        - All sequences are padded to the same length.
    """
    sequences = [[1], [1, 2], [1, 2, 3]]
    max_length = 5
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="pre")

    # Assert that all sequences are padded correctly
    assert padded_sequences.shape == (3, max_length)
    assert (padded_sequences[0] == [0, 0, 0, 0, 1]).all()
    assert (padded_sequences[1] == [0, 0, 0, 1, 2]).all()
    assert (padded_sequences[2] == [0, 0, 1, 2, 3]).all()

@patch("models.base_models.markov_chain_and_lstm.model.fit")
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

    # Reshape y to match the LSTM output shape
    reshaped_y = np.expand_dims(y, axis=-1)

    # Train the model
    model.fit(X, reshaped_y, epochs=500, verbose=1)

    # Assert that the fit method was called with the correct arguments
    mock_fit.assert_called_once_with(X, reshaped_y, epochs=500, verbose=1, callbacks=pytest.ANY)

@patch("models.base_models.markov_chain_and_lstm.model.predict")
def test_predict_next_word(mock_predict):
    """
    Test the `predict_next_word` function.

    Args:
        mock_predict (MagicMock): Mocked `predict` method of the model.

    Asserts:
        - The function returns the correct predicted word.
    """
    # Mock the predict method
    mock_predict.return_value = np.array([[0.1, 0.2, 0.7]])  # Simulated probabilities

    # Call the function with a sample input
    result = predict_next_word("the cat")

    # Assert that the function returns the correct predicted word
    assert result == "sat"  # Expected predicted word based on mock probabilities
    mock_predict.assert_called_once()

def test_predict_next_word_invalid_input():
    """
    Test the `predict_next_word` function with invalid input.

    Asserts:
        - The function raises a KeyError for unknown words.
    """
    with pytest.raises(KeyError):
        predict_next_word("unknown word")
