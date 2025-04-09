# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import patch, MagicMock  # For mocking external dependencies
import torch  # For tensor operations

# Import the function, tokenizer, and model to test
from models.base_models.bert import predict_next_word_bert, tokenizer, model

# -------------------------------
# Test Suite for BERT Model
# -------------------------------


def predict_next_word_bert(text, top_k=5):
    """
    Predicts the next word(s) for a given input text using the BERT model.

    Args:
        text (str): The input text for which the next word(s) are predicted.
        top_k (int): The number of top probable next words to return (default is 5).

    Returns:
        list of str: A list of the top `top_k` predicted next words.
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    if not isinstance(top_k, int) or top_k <= 0:
        # Ensure top_k is a positive integer
        raise ValueError("top_k must be a positive integer")

    # Append the [MASK] token to the input text
    masked_text = text + " [MASK]"

    # Tokenize the input text and convert it into tensors
    inputs = tokenizer(masked_text, return_tensors="pt")

    # Get the position of the [MASK] token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[
        1
    ].item()

    # Perform inference with the BERT model (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits for the [MASK] token
    logits = outputs.logits[0, mask_token_index, :]

    # Apply softmax to convert logits into probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top `top_k` tokens with the highest probabilities
    top_k_tokens = torch.topk(probabilities, top_k, dim=-1)

    # Decode the token IDs back into words
    next_words = [tokenizer.decode([token]) for token in top_k_tokens.indices]

    return next_words


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
