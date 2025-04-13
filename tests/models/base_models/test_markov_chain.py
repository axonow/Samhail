import pytest
from models.base_models.markov_chain import MarkovChain


def test_markov_chain_training():
    mc = MarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Validate transitions
    assert mc.transitions["the"]["cat"] == 2
    assert mc.transitions["cat"]["sat"] == 1
    assert mc.transitions["cat"]["jumped"] == 1
    assert mc.transitions["sat"]["on"] == 1
    assert mc.transitions["on"]["the"] == 1
    assert mc.transitions["mat"]["the"] == 1

    # Validate total counts
    assert mc.total_counts["the"] == 4
    assert mc.total_counts["cat"] == 2
    assert mc.total_counts["sat"] == 1
    assert mc.total_counts["on"] == 1
    assert mc.total_counts["mat"] == 1


def test_markov_chain_training_repeated_word():
    mc = MarkovChain()
    mc.train("hello hello hello")

    # Validate transitions
    assert mc.transitions["hello"]["hello"] == 2

    # Validate total counts
    assert mc.total_counts["hello"] == 2


def test_markov_chain_training_single_word():
    mc = MarkovChain()
    mc.train("hello")

    # No transitions or counts should be added
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0


def test_markov_chain_predict_valid_word():
    mc = MarkovChain()
    text = "the cat sat on the mat the cat jumped over the mat"
    mc.train(text)

    # Validate predictions
    assert mc.predict("cat") in {"sat", "jumped"}
    assert mc.predict("the") in {"cat", "mat"}


def test_markov_chain_predict_invalid_word():
    mc = MarkovChain()
    mc.train("the cat sat on the mat the cat jumped over the mat")

    # Predict a word not in the model
    assert mc.predict("dog") is None


def test_markov_chain_predict_no_training():
    mc = MarkovChain()

    # Predict without training
    assert mc.predict("cat") is None


def test_markov_chain_training_empty_text():
    mc = MarkovChain()
    mc.train("")

    # No transitions or counts should be added
    assert len(mc.transitions) == 0
    assert len(mc.total_counts) == 0
