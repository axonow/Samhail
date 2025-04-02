# Import necessary libraries
from transformers import BertTokenizer, BertForMaskedLM  # For loading the BERT model and tokenizer
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

def predict_next_word_bert(text, top_k=5):
    """
    Predicts the next word(s) for a given input text using the BERT model.

    Args:
        text (str): The input text for which the next word(s) are predicted.
        top_k (int): The number of top probable next words to return (default is 5).

    Returns:
        list of str: A list of the top `top_k` predicted next words.

    Steps:
    1. Append a [MASK] token to the input text to indicate the missing word.
    2. Tokenize the input text using the BERT tokenizer.
    3. Pass the tokenized input to the BERT model to get the logits (raw predictions).
    4. Extract the logits for the [MASK] token.
    5. Apply the softmax function to convert logits into probabilities.
    6. Use `torch.topk` to get the top `top_k` tokens with the highest probabilities.
    7. Decode the token IDs back into words using the tokenizer.

    Example:
        Input: "The cat sat on the"
        Output: ["mat", "floor", "sofa", "chair", "bed"]
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer") # Ensure top_k is a positive integer

    # Append the [MASK] token to the input text
    masked_text = text + " [MASK]"

    # Tokenize the input text and convert it into tensors
    inputs = tokenizer(masked_text, return_tensors="pt")

    # Get the position of the [MASK] token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

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
    next_words = [tokenizer.decode([token]) for token in top_k_tokens.indices[0]]

    return next_words

# -------------------------------
# Example Predictions
# -------------------------------

if __name__ == "__main__":
    # Predict the next word(s) for various input texts
    # The predictions are based on the context provided in the input text.
    print(predict_next_word_bert("The cat sat on the"))  # Example output: ["mat", "floor", "sofa", "chair", "bed"]
    print(predict_next_word_bert("Deep learning is"))  # Example output: ["transforming", "revolutionizing", "advancing", "changing", "reshaping"]
    print(predict_next_word_bert("Transformers are revolutionizing"))  # Example output: ["AI", "technology", "NLP", "research", "science"]
