# Import necessary libraries
import pytest  # For writing and running tests
from unittest.mock import MagicMock  # For mocking methods
from models.base_models.n_gram import NGramModel  # Import the NGramModel class

# -------------------------------
# Test Suite for N-Gram Model
# -------------------------------

def test_ngram_model_initialization():
    """
    Test the initialization of the NGramModel.

    Asserts:
        - The `n` attribute is correctly set.
        - The `ngram_counts` and `total_counts` dictionaries are initialized as empty.
    """
    ngram_model = NGramModel(n=3)
    assert ngram_model.n == 3
    assert isinstance(ngram_model.ngram_counts, dict)
    assert isinstance(ngram_model.total_counts, dict)
    assert len(ngram_model.ngram_counts) == 0
    assert len(ngram_model.total_counts) == 0

def test_ngram_model_training():
    """
    Test the training of the NGramModel.

    Asserts:
        - The `ngram_counts` and `total_counts` dictionaries are updated correctly.
    """
    ngram_model = NGramModel(n=3)
    text = "the cat sat on the mat"
    ngram_model.train(text)

    # Check n-gram counts
    assert ngram_model.ngram_counts[("the", "cat")]["sat"] == 1
    assert ngram_model.ngram_counts[("cat", "sat")]["on"] == 1
    assert ngram_model.ngram_counts[("sat", "on")]["the"] == 1

    # Check total counts
    assert ngram_model.total_counts[("the", "cat")] == 1
    assert ngram_model.total_counts[("cat", "sat")] == 1
    assert ngram_model.total_counts[("sat", "on")] == 1

def test_ngram_model_prediction():
    """
    Test the prediction functionality of the NGramModel.

    Asserts:
        - The predicted word is one of the possible next words based on probabilities.
    """
    ngram_model = NGramModel(n=3)
    text = "the cat sat on the mat"
    ngram_model.train(text)

    # Predict the next word
    predicted_word = ngram_model.predict(["the", "cat"])
    assert predicted_word in ["sat"]

def test_ngram_model_prediction_no_data():
    """
    Test the prediction functionality when no data is available for the given context.

    Asserts:
        - The function returns `None` when no prediction is possible.
    """
    ngram_model = NGramModel(n=3)
    predicted_word = ngram_model.predict(["unknown", "context"])
    assert predicted_word is None

def test_ngram_model_training_with_multiple_sentences():
    """
    Test the training of the NGramModel with multiple sentences.

    Asserts:
        - The `ngram_counts` and `total_counts` dictionaries are updated correctly for multiple sentences.
    """
    ngram_model = NGramModel(n=3)
    text = "the cat sat on the mat. the dog barked at the stranger."
    ngram_model.train(text)

    # Check n-gram counts
    assert ngram_model.ngram_counts[("the", "cat")]["sat"] == 1
    assert ngram_model.ngram_counts[("the", "dog")]["barked"] == 1
    assert ngram_model.ngram_counts[("dog", "barked")]["at"] == 1

    # Check total counts
    assert ngram_model.total_counts[("the", "cat")] == 1
    assert ngram_model.total_counts[("the", "dog")] == 1
    assert ngram_model.total_counts[("dog", "barked")] == 1
