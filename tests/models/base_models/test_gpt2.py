# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
from models.base_models.gpt2 import predict_next_word  # Import the function to test
import torch  # For tensor operations
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # For GPT-2 model and tokenizer

# -------------------------------
# Test Suite for GPT-2 Prediction Function
# -------------------------------

@patch("models.base_models.gpt2.tokenizer")
@patch("models.base_models.gpt2.model")
def test_predict_next_word(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word` function to ensure it correctly predicts the next words.

    Args:
        mock_model (MagicMock): Mocked GPT-2 model.
        mock_tokenizer (MagicMock): Mocked GPT-2 tokenizer.

    Steps:
    1. Mock the tokenizer to simulate tokenization and decoding behavior.
    2. Mock the model to simulate inference and return logits.
    3. Call the `predict_next_word` function with a sample input.
    4. Assert that the function returns the expected top-k predictions.
    """

    # Mock the tokenizer's behavior
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}  # Simulated tokenized input
    mock_tokenizer.decode.side_effect = lambda token_id: f"word{token_id}"  # Simulated decoding

    # Mock the model's behavior
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])  # Simulated logits
    mock_model.return_value = mock_outputs

    # Call the function with a sample input
    result = predict_next_word("The cat sat on the", top_k=3)

    # Assert that the function returns the expected top-k predictions
    assert result == ["word4", "word3", "word2"]  # Expected decoded words
    mock_tokenizer.assert_called_once_with("The cat sat on the", return_tensors="pt")
    mock_model.assert_called_once()

def test_predict_next_word_invalid_input():
    """
    Test the `predict_next_word` function with invalid input to ensure it handles errors gracefully.

    Steps:
    1. Call the `predict_next_word` function with an empty string.
    2. Assert that the function raises a ValueError.
    """
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_next_word("")

@patch("models.base_models.gpt2.tokenizer")
@patch("models.base_models.gpt2.model")
def test_predict_next_word_top_k(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word` function with different `top_k` values.

    Args:
        mock_model (MagicMock): Mocked GPT-2 model.
        mock_tokenizer (MagicMock): Mocked GPT-2 tokenizer.

    Steps:
    1. Mock the tokenizer and model behavior.
    2. Call the `predict_next_word` function with a specific `top_k` value.
    3. Assert that the function returns the correct number of predictions.
    """

    # Mock the tokenizer's behavior
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}  # Simulated tokenized input
    mock_tokenizer.decode.side_effect = lambda token_id: f"word{token_id}"  # Simulated decoding

    # Mock the model's behavior
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])  # Simulated logits
    mock_model.return_value = mock_outputs

    # Call the function with a specific `top_k` value
    result = predict_next_word("The cat sat on the", top_k=2)

    # Assert that the function returns the correct number of predictions
    assert len(result) == 2
    assert result == ["word4", "word3"]  # Expected decoded words
