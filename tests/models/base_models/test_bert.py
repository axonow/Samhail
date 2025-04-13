import pytest
from unittest.mock import patch, MagicMock
import torch
from models.base_models.bert import predict_next_word_bert
import re  # Import the `re` module for escaping special characters


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_valid_input(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with valid input text.
    """
    # Mock the tokenizer's behavior
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.mask_token_id = 103
    mock_tokenizer.mask_token = "[MASK]"
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 2009, 2001, 103]])}
    # Update the decode method to return the correct format
    mock_tokenizer.decode.side_effect = lambda token_id: (
        f"word{token_id[0]}" if isinstance(token_id, list) else f"word{token_id}"
    )

    # Mock the model's behavior
    mock_outputs = MagicMock()
    # Ensure the logits tensor matches the input_ids structure
    mock_outputs.logits = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],  # Token 101
                [0.1, 0.2, 0.3, 0.4],  # Token 2009
                [0.1, 0.2, 0.3, 0.4],  # Token 2001
                # Token 103 ([MASK])
                [0.1, 0.2, 0.3, 0.4],
            ]
        ]
    )
    mock_model.return_value = mock_outputs

    # Call the function with a valid input
    result = predict_next_word_bert("The cat sat on the [MASK]", top_k=2)

    # Assert the expected output
    assert result == ["word3", "word2"]


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_top_k_exceeds_logits(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function when `top_k` exceeds the number of logits.
    """
    # Mock the tokenizer's behavior
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[101, 2009, 2001, 103]])}
    mock_tokenizer.mask_token_id = 103
    # Fix the decode method to return the correct format
    mock_tokenizer.decode.side_effect = lambda token_id: (
        f"word{token_id}" if isinstance(token_id, int) else f"word{token_id[0]}"
    )

    # Mock the model's behavior
    mock_outputs = MagicMock()
    # Ensure the logits tensor has enough values for the `[MASK]` token
    mock_outputs.logits = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],  # Token 101
                [0.1, 0.2, 0.3, 0.4],  # Token 2009
                [0.1, 0.2, 0.3, 0.4],  # Token 2001
                [0.1, 0.2, 0.3, 0.4],  # Token 103 ([MASK])
            ]
        ]
    )
    mock_model.return_value = mock_outputs

    # Call the function with a `top_k` value that exceeds the number of logits
    result = predict_next_word_bert("The cat sat on the [MASK]", top_k=5)

    # Assert the expected output
    assert result == ["word3", "word2", "word1", "word0"]


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_no_mask_token(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function when no [MASK] token is present in the input.
    """
    # Escape the [MASK] token in the regex pattern
    with pytest.raises(
        ValueError, match=re.escape("No [MASK] token found in the input")
    ):
        predict_next_word_bert("The cat sat on the", top_k=3)


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_empty_input(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with empty input text.
    """
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_next_word_bert("", top_k=3)


@patch("models.base_models.bert.tokenizer")
@patch("models.base_models.bert.model")
def test_predict_next_word_bert_invalid_top_k(mock_model, mock_tokenizer):
    """
    Test the `predict_next_word_bert` function with invalid `top_k` values.
    """
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_next_word_bert("The cat sat on the [MASK]", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        predict_next_word_bert("The cat sat on the [MASK]", top_k=-1)
