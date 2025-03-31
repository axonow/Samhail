# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
import torch  # For tensor operations
from models.base_models.bert import predict_next_word_bert  # Import the function to test

# -------------------------------
# Test Suite for BERT Model
# -------------------------------

@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_valid_input(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with valid input text.

    Steps:
    1. Mock the tokenizer to simulate tokenization and decoding behavior.
    2. Mock the model to simulate inference and return logits.
    3. Call the `predict_next_word_bert` function with a sample input.
    4. Assert that the function returns the expected top-k predicted words.
    """

    # Mock the tokenizer's behavior
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.mask_token_id = 103
    mock_tokenizer.mask_token = "[MASK]"
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 2009, 2001, 103]])}
    mock_tokenizer.decode.side_effect = lambda token_id: f"word{token_id}"

    # Mock the model's behavior
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])  # Simulated logits for 4 tokens
    mock_model.return_value = mock_outputs

    # Call the function with a sample input
    result = predict_next_word_bert("The cat sat on the", top_k=3)

    # Assert that the function returns the expected top-k predicted words
    assert result == ["word3", "word2", "word1"]


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_empty_input(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with empty input text.

    Steps:
    1. Call the `predict_next_word_bert` function with an empty string.
    2. Assert that the function raises a ValueError with the appropriate error message.
    """

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_next_word_bert("", top_k=3)


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_top_k(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with different `top_k` values.

    Steps:
    1. Mock the tokenizer to simulate tokenization and decoding behavior.
    2. Mock the model to simulate inference and return logits.
    3. Call the `predict_next_word_bert` function with a specific `top_k` value.
    4. Assert that the function returns the correct number of predictions.
    """

    # Mock the tokenizer's behavior
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.mask_token_id = 103
    mock_tokenizer.mask_token = "[MASK]"
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 2009, 2001, 103]])}
    mock_tokenizer.decode.side_effect = lambda token_id: f"word{token_id}"

    # Mock the model's behavior
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])  # Simulated logits for 4 tokens
    mock_model.return_value = mock_outputs

    # Call the function with a specific `top_k` value
    result = predict_next_word_bert("The cat sat on the", top_k=2)

    # Assert that the function returns the correct number of predictions
    assert len(result) == 2
    assert result == ["word3", "word2"]


def test_predict_next_word_bert_invalid_top_k():
    """
    Test the `predict_next_word_bert` function with an invalid `top_k` value.

    Steps:
    1. Call the `predict_next_word_bert` function with an invalid `top_k` value (e.g., 0 or negative).
    2. Assert that the function raises a ValueError with the appropriate error message.
    """

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_next_word_bert("The cat sat on the", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_next_word_bert("The cat sat on the", top_k=-1)
