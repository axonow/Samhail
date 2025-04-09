import pytest
from models.base_models.markov_chain import MarkovChain


def test_markov_chain_initialization():
    """
    Test the initialization of the MarkovChain class.
    Asserts:
        - The `transitions` dictionary is empty.
        - The `total_counts` dictionary is empty.
    """
    mc = MarkovChain()
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0


def test_markov_chain_training():
    """
    Test the `train` method of the MarkovChain class.
    Asserts:
        - The `transitions` and `total_counts` dictionaries are updated correctly.
    """
    mc = MarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Check transitions
    assert mc.transitions["the"]["cat"] == 2
    assert mc.transitions["cat"]["sat"] == 1
    assert mc.transitions["cat"]["jumped"] == 1
    assert mc.transitions["sat"]["on"] == 1
    assert mc.transitions["on"]["the"] == 1

    # Check total counts
    assert mc.total_counts["the"] == 3
    assert mc.total_counts["cat"] == 2
    assert mc.total_counts["sat"] == 1
    assert mc.total_counts["on"] == 1


def test_markov_chain_predict_valid_word():
    """
    Test the `predict` method with a valid word.
    Asserts:
        - The predicted word is one of the possible next words based on the learned probabilities.
    """
    mc = MarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Predict the next word for "cat"
    next_word = mc.predict("cat")
    assert next_word in {"sat", "jumped"}

    # Predict the next word for "the"
    next_word = mc.predict("the")
    assert next_word in {"cat", "mat"}


def test_markov_chain_predict_invalid_word():
    """
    Test the `predict` method with an invalid word.
    Asserts:
        - The method returns `None` for a word not in the model.
    """
    mc = MarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Predict the next word for a word not in the model
    next_word = mc.predict("dog")
    assert next_word is None


def test_markov_chain_predict_no_training():
    """
    Test the `predict` method without training the model.
    Asserts:
        - The method returns `None` for any word when the model is not trained.
    """
    mc = MarkovChain()

    # Predict the next word for any input without training
    next_word = mc.predict("cat")
    assert next_word is None


def test_markov_chain_training_empty_text():
    """
    Test the `train` method with an empty text.
    Asserts:
        - The `transitions` and `total_counts` dictionaries remain empty.
    """
    mc = MarkovChain()
    mc.train("")

    # Check that no transitions or counts are added
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0


def test_markov_chain_training_single_word():
    """
    Test the `train` method with a single word.
    Asserts:
        - The `transitions` and `total_counts` dictionaries remain empty.
    """
    mc = MarkovChain()
    mc.train("hello")

    # Check that no transitions or counts are added
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0


def test_markov_chain_training_repeated_word():
    """
    Test the `train` method with repeated words.
    Asserts:
        - The `transitions` and `total_counts` dictionaries are updated correctly.
    """
    mc = MarkovChain()
    mc.train("hello hello hello")

    # Check transitions
    assert mc.transitions["hello"]["hello"] == 2

    # Check total counts
    assert mc.total_counts["hello"] == 2
