# Import necessary libraries
# For loading the GPT-2 model and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch  # For tensor operations and model inference

# -------------------------------
# Load Pre-trained GPT-2 Model and Tokenizer
# -------------------------------

# Load the pre-trained GPT-2 tokenizer
# The tokenizer converts input text into token IDs that the model can process.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the pre-trained GPT-2 language model
# The model generates text by predicting the next word based on the input context.
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

# -------------------------------
# Prediction Function
# -------------------------------


def predict_next_word(text, top_k=5):
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    """
    Predicts the next word(s) for a given input text using the GPT-2 model.

    Args:
        text (str): The input text for which the next word(s) are predicted.
        top_k (int): The number of top probable next words to return (default is 5).

    Returns:
        list of str: A list of the top `top_k` predicted next words.

    Steps:
    1. Tokenize the input text using the GPT-2 tokenizer.
    2. Pass the tokenized input to the GPT-2 model to get the logits (raw predictions).
    3. Extract the logits for the last token in the sequence.
    4. Apply the softmax function to convert logits into probabilities.
    5. Use `torch.topk` to get the top `top_k` tokens with the highest probabilities.
    6. Decode the token IDs back into words using the tokenizer.

    Example:
        Input: "The cat sat on the"
        Output: ["mat", "floor", "sofa", "chair", "bed"]
    """
    # Tokenize the input text and convert it into tensors
    inputs = tokenizer(text, return_tensors="pt")

    # Perform inference with the GPT-2 model (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits for the last token in the sequence
    logits = outputs.logits[:, -1, :]

    # Apply softmax to convert logits into probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top `top_k` tokens with the highest probabilities
    top_k_tokens = torch.topk(probabilities, top_k)

    # Decode the token IDs back into words
    next_words = [tokenizer.decode([token])
                  for token in top_k_tokens.indices[0]]

    return next_words


# -------------------------------
# Example Predictions
# -------------------------------


# Predict the next word(s) for various input texts
# The predictions are based on the context provided in the input text.
# Example output: ["mat", "floor", "sofa", "chair", "bed"]
print(predict_next_word("The cat sat on the"))
# Example output: ["transforming", "revolutionizing", "advancing", "changing", "reshaping"]
print(predict_next_word("Deep learning is"))
# Example output: ["AI", "technology", "NLP", "research", "science"]
print(predict_next_word("Transformers are revolutionizing"))
