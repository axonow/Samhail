# Import necessary libraries
import pytest  # For writing and running tests
from models.base_models.second_order_markov_chain import SecondOrderMarkovChain  # Import the class to test

# -------------------------------
# Test Suite for SecondOrderMarkovChain
# -------------------------------

def test_initialization():
    """
    Test the initialization of the SecondOrderMarkovChain.

    Asserts:
        - The `transitions` and `total_counts` dictionaries are initialized as empty.
    """
    mc = SecondOrderMarkovChain()
    assert isinstance(mc.transitions, dict)
    assert isinstance(mc.total_counts, dict)
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0

def test_training():
    """
    Test the training of the SecondOrderMarkovChain.

    Asserts:
        - The `transitions` and `total_counts` dictionaries are updated correctly.
    """
    mc = SecondOrderMarkovChain()
    text = "the cat sat on the mat"
    mc.train(text)

    # Check transitions
    assert mc.transitions["the"]["cat"]["sat"] == 1
    assert mc.transitions["cat"]["sat"]["on"] == 1
    assert mc.transitions["sat"]["on"]["the"] == 1

    # Check total counts
    assert mc.total_counts["the"]["cat"] == 1
    assert mc.total_counts["cat"]["sat"] == 1
    assert mc.total_counts["sat"]["on"] == 1

def test_prediction():
    """
    Test the prediction functionality of the SecondOrderMarkovChain.

    Asserts:
        - The predicted word is one of the possible next words based on probabilities.
    """
    mc = SecondOrderMarkovChain()
    text = "the cat sat on the mat"
    mc.train(text)

    # Predict the next word
    predicted_word = mc.predict("the", "cat")
    assert predicted_word in ["sat"]

def test_prediction_no_data():
    """
    Test the prediction functionality when no data is available for the given context.

    Asserts:
        - The function returns `None` when no prediction is possible.
    """
    mc = SecondOrderMarkovChain()
    predicted_word = mc.predict("unknown", "context")
    assert predicted_word is None

def test_training_with_multiple_sentences():
    """
    Test the training of the SecondOrderMarkovChain with multiple sentences.

    Asserts:
        - The `transitions` and `total_counts` dictionaries are updated correctly for multiple sentences.
    """
    mc = SecondOrderMarkovChain()
    text = "the cat sat on the mat. the dog barked at the stranger."
    mc.train(text)

    # Check transitions
    assert mc.transitions["the"]["cat"]["sat"] == 1
    assert mc.transitions["the"]["dog"]["barked"] == 1
    assert mc.transitions["dog"]["barked"]["at"] == 1

    # Check total counts
    assert mc.total_counts["the"]["cat"] == 1
    assert mc.total_counts["the"]["dog"] == 1
    assert mc.total_counts["dog"]["barked"] == 1

def test_prediction_with_multiple_options():
    """
    Test the prediction functionality when multiple next words are possible.

    Asserts:
        - The predicted word is one of the possible next words based on probabilities.
    """
    mc = SecondOrderMarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Predict the next word
    predicted_word = mc.predict("the", "cat")
    assert predicted_word in ["sat", "jumped"]
