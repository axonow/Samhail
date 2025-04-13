# Import necessary libraries
# For loading the BERT model and tokenizer
import torch
from transformers import BertTokenizer, BertForMaskedLM
import torch  # For tensor operations and model inference
import logging  # For suppressing warnings

# -------------------------------
# Suppress Warnings from Transformers Library
# -------------------------------

# The warning occurs because the `bert-base-uncased` checkpoint includes weights for tasks beyond masked language modeling (MLM),
# such as Next Sentence Prediction (NSP) and the pooler layer used for sentence-level tasks.
# These weights are not used by the `BertForMaskedLM` model, which is specifically designed for MLM tasks.
# The warning is expected and does not affect the functionality of the model for MLM tasks.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# -------------------------------
# Load Pre-trained BERT Model and Tokenizer
# -------------------------------

# Load the pre-trained BERT tokenizer
# The tokenizer converts input text into token IDs that the model can process.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the pre-trained BERT model for masked language modeling
# The model predicts the masked token in the input text.
# Note: Some weights in the `bert-base-uncased` checkpoint (e.g., NSP and pooler weights) are not used by this model.
# This is expected behavior and does not affect the MLM functionality.
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

# -------------------------------
# Prediction Function
# -------------------------------


def predict_next_word_bert(input_text, top_k=5):
    """
    Predict the next word(s) using a BERT model.

    Args:
        input_text (str): The input text containing a [MASK] token.
        top_k (int): The number of top predictions to return.

    Returns:
        list: A list of the top-k predicted words.

    Raises:
        ValueError: If the input text is empty, does not contain a [MASK] token,
                    or if `top_k` is not a positive integer.
    """
    if not input_text:
        raise ValueError("Input text cannot be empty")
    if "[MASK]" not in input_text:
        raise ValueError("No [MASK] token found in the input")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    # Ensure the [MASK] token exists in the input
    if mask_token_index.numel() == 0:
        raise ValueError("No [MASK] token found in the input")

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Extract logits for the [MASK] token
    mask_token_logits = logits[0, mask_token_index, :].squeeze(0)

    # Limit top_k to the number of available logits
    top_k = min(top_k, mask_token_logits.size(0))

    # Get the top-k predictions
    top_k_indices = torch.topk(mask_token_logits, top_k).indices.tolist()

    # Decode the predicted token IDs to words
    predicted_words = [tokenizer.decode([idx]).strip() for idx in top_k_indices]

    return predicted_words


# -------------------------------
# Example Predictions
# -------------------------------


if __name__ == "__main__":
    # Predict the next word(s) for various input texts
    # The predictions are based on the context provided in the input text.
    # Example output: ["mat", "floor", "sofa", "chair", "bed"]
    print(predict_next_word_bert("The cat sat on the"))
    # Example output: ["transforming", "revolutionizing", "advancing", "changing", "reshaping"]
    print(predict_next_word_bert("Deep learning is"))
    # Example output: ["AI", "technology", "NLP", "research", "science"]
    print(predict_next_word_bert("Transformers are revolutionizing"))
