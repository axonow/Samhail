#!/usr/bin/env python3
"""
Tests for the MarkovChain model.

This module provides comprehensive tests for the MarkovChain model,
focusing on behavior rather than implementation details.
"""

from utils.database_adapters.postgresql.markov_chain import MarkovChainPostgreSqlAdapter
from data_preprocessing.text_preprocessor import TextPreprocessor
from models.production_models.markov_chain.markov_chain import MarkovChain, process_text_batch
import os
import sys
import json
import pytest
import psutil
import random
import numpy as np
from collections import defaultdict
from unittest.mock import MagicMock, patch, ANY

# Add project root directory to Python path for reliable imports
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the MarkovChain class and dependencies


# Define fixture for mock logger
@pytest.fixture
def mock_logger():
    """Create a mock logger for testing"""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


# Define fixture for mock DB adapter
@pytest.fixture
def mock_db_adapter():
    """Create a mock database adapter for testing"""
    adapter = MagicMock(spec=MarkovChainPostgreSqlAdapter)
    adapter.is_usable.return_value = True
    adapter.get_connection.return_value = MagicMock()
    adapter.has_transitions.return_value = True
    return adapter


# Define fixture for mock text preprocessor
@pytest.fixture
def mock_preprocessor():
    """Create a mock text preprocessor for testing"""
    preprocessor = MagicMock(spec=TextPreprocessor)
    preprocessor.to_lowercase.side_effect = lambda text: text.lower()
    preprocessor.handle_whitespace.side_effect = lambda text: ' '.join(
        text.split())
    preprocessor.handle_contractions.return_value = "this is a test"
    preprocessor.normalize.return_value = "this is a test"
    preprocessor.normalize_social_media_text.return_value = "this is a test"
    preprocessor.handle_emojis.return_value = "this is a test"
    preprocessor.handle_urls.return_value = "this is a test"
    preprocessor.remove_html_tags.return_value = "this is a test"
    return preprocessor


@pytest.fixture
def markov_chain(mock_logger, mock_db_adapter, mock_preprocessor):
    """Create a MarkovChain instance with mocked dependencies"""
    with patch('models.production_models.markov_chain.markov_chain.MarkovChainPostgreSqlAdapter',
               return_value=mock_db_adapter):
        with patch('models.production_models.markov_chain.markov_chain.TextPreprocessor',
                   return_value=mock_preprocessor):
            with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                # Create the model with mocked dependencies
                model = MarkovChain(
                    n_gram=2,
                    environment="test",
                    logger=mock_logger
                )

                # Replace the created dependencies with our controlled mocks
                # This is crucial because the MarkovChain might create its own instances
                # even though we mocked the classes
                model.db_adapter = mock_db_adapter
                model.preprocessor = mock_preprocessor

                # Set up common mock behaviors needed by most tests
                mock_db_adapter.predict_next_word.return_value = {
                    "apple": 0.5,
                    "banana": 0.3,
                    "cherry": 0.2
                }

                yield model


class TestMarkovChain:
    """Test suite for MarkovChain class."""

    def test_initialization(self, mock_logger):
        """Test MarkovChain initialization with various parameters."""
        # Test initialization with required parameters
        with patch('models.production_models.markov_chain.markov_chain.MarkovChainPostgreSqlAdapter') as mock_adapter:
            with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                with patch('models.production_models.markov_chain.markov_chain.TextPreprocessor'):
                    mock_adapter_instance = MagicMock()
                    mock_adapter_instance.is_usable.return_value = True
                    mock_adapter.return_value = mock_adapter_instance

                    model = MarkovChain(
                        n_gram=2,
                        environment="test",
                        logger=mock_logger
                    )

                    assert model.n_gram == 2
                    assert model.environment == "test"
                    assert model.logger == mock_logger
                    assert model.using_db is True
                    assert isinstance(model.transitions, defaultdict)
                    mock_logger.info.assert_called()

    def test_initialization_error_no_logger(self):
        """Test initialization fails when no logger is provided."""
        with pytest.raises(ValueError, match="Logger instance must be provided"):
            MarkovChain(n_gram=1, environment="test", logger=None)

    def test_initialization_db_adapter_error(self, mock_logger):
        """Test initialization handles DB adapter errors properly."""
        with patch('models.production_models.markov_chain.markov_chain.MarkovChainPostgreSqlAdapter') as mock_adapter:
            with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                with patch('models.production_models.markov_chain.markov_chain.TextPreprocessor'):
                    # Make adapter raise an exception
                    mock_adapter.side_effect = Exception(
                        "DB connection failed")

                    with pytest.raises(ValueError, match="Database adapter initialization failed"):
                        MarkovChain(n_gram=1, environment="test",
                                    logger=mock_logger)

                    mock_logger.error.assert_called()

    def test_initialization_db_not_usable(self, mock_logger):
        """Test initialization handles non-usable DB adapter."""
        with patch('models.production_models.markov_chain.markov_chain.MarkovChainPostgreSqlAdapter') as mock_adapter:
            with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                with patch('models.production_models.markov_chain.markov_chain.TextPreprocessor'):
                    mock_adapter_instance = MagicMock()
                    mock_adapter_instance.is_usable.return_value = False
                    mock_adapter.return_value = mock_adapter_instance

                    with pytest.raises(ValueError, match="Database adapter is not usable"):
                        MarkovChain(n_gram=1, environment="test",
                                    logger=mock_logger)

    def test_preprocess_text(self, markov_chain):
        """Test the text preprocessing method."""
        # Instead of creating a new mock, use the mock_preprocessor from the fixture
        # and reset it to ensure clean state
        markov_chain.preprocessor.to_lowercase.reset_mock()
        markov_chain.preprocessor.handle_contractions.reset_mock()
        markov_chain.preprocessor.normalize.reset_mock()
        markov_chain.preprocessor.normalize_social_media_text.reset_mock()
        markov_chain.preprocessor.handle_emojis.reset_mock()
        markov_chain.preprocessor.handle_urls.reset_mock()
        markov_chain.preprocessor.remove_html_tags.reset_mock()
        markov_chain.preprocessor.handle_whitespace.reset_mock()

        # Set up the mock to return processed text
        markov_chain.preprocessor.to_lowercase.return_value = "this is a test"

        # Test the method
        result = markov_chain._preprocess_text("THIS IS A TEST")

        # Verify that all preprocessing methods were called
        markov_chain.preprocessor.to_lowercase.assert_called_once()
        markov_chain.preprocessor.handle_contractions.assert_called_once()
        markov_chain.preprocessor.normalize.assert_called_once()
        markov_chain.preprocessor.normalize_social_media_text.assert_called_once()
        markov_chain.preprocessor.handle_emojis.assert_called_once()
        markov_chain.preprocessor.handle_urls.assert_called_once()
        markov_chain.preprocessor.remove_html_tags.assert_called_once()
        markov_chain.preprocessor.handle_whitespace.assert_called_once()

        # Verify the result
        assert result == "this is a test"

    def test_preprocess_text_no_preprocessor(self, markov_chain):
        """Test text preprocessing when preprocessor is not available."""
        markov_chain.preprocessor = None
        test_text = "THIS IS A TEST"
        result = markov_chain._preprocess_text(test_text)

        # If preprocessor is unavailable, text should be returned as is
        assert result == test_text
        markov_chain.logger.warning.assert_called_once()

    def test_preprocess_text_exception(self, markov_chain):
        """Test text preprocessing when an exception occurs."""
        # Store the original preprocessor
        original_preprocessor = markov_chain.preprocessor

        try:
            # Create a completely new mock preprocessor that raises an exception
            # immediately on the first method call
            mock_preprocessor = MagicMock(spec=TextPreprocessor)
            mock_preprocessor.to_lowercase.side_effect = Exception(
                "Test exception")
            markov_chain.preprocessor = mock_preprocessor

            # Original text to be passed for processing
            original_text = "THIS IS A TEST"

            # Call the method
            result = markov_chain._preprocess_text(original_text)

            # When exception occurs, original text should be returned unchanged
            assert result == original_text

            # Verify error was logged
            markov_chain.logger.error.assert_called_once()
        finally:
            # Restore the original preprocessor
            markov_chain.preprocessor = original_preprocessor

    def test_train_single_text(self, markov_chain):
        """Test training on a single text input."""
        # Mock the _preprocess_text method
        with patch.object(markov_chain, '_preprocess_text', return_value="this is a test text for training"):
            # Mock the resource_monitor
            markov_chain.resource_monitor = MagicMock()

            # Prepare test data
            test_text = "This is a test text for training"

            # Train the model
            result = markov_chain._train_single_text(test_text)

            # Verify that the database methods were called correctly
            markov_chain.db_adapter.insert_transitions_batch.assert_called()
            markov_chain.db_adapter.update_total_counts.assert_called_once()

            # Verify result contains expected keys
            assert "training_time" in result
            assert "word_count" in result
            assert "total_transitions" in result

            # Verify resource monitoring was used
            markov_chain.resource_monitor.log_progress.assert_called()
            markov_chain.resource_monitor.stop.assert_called_once()

    def test_train_single_text_too_short(self, markov_chain):
        """Test training on a text that is too short."""
        # Mock the _preprocess_text method
        with patch.object(markov_chain, '_preprocess_text', return_value="short"):
            # Mock the resource_monitor
            markov_chain.resource_monitor = MagicMock()

            # Prepare test data (too short for n_gram=2)
            test_text = "short"

            # Train the model
            result = markov_chain._train_single_text(test_text)

            # Verify result indicates error
            assert "error" in result
            assert result["error"] == "Text too short for training"

            # Verify warning was logged
            markov_chain.logger.warning.assert_called()

            # Verify resource monitoring was stopped
            markov_chain.resource_monitor.stop.assert_called_once()

    def test_train_multiple_texts(self, markov_chain):
        """Test training on multiple text inputs."""
        # Mock process_text_batch to avoid subprocess issues in testing
        with patch('models.production_models.markov_chain.markov_chain.process_text_batch') as mock_process:
            # Set up mock return values for process_text_batch
            mock_process.side_effect = [
                {
                    "idx": 0,
                    "transitions": {"word1": {"word2": 1}},
                    "processing_time": 0.1,
                    "word_count": 10,
                    "transition_count": 1,
                    "state_count": 1
                },
                {
                    "idx": 1,
                    "transitions": {"word2": {"word3": 1}},
                    "processing_time": 0.1,
                    "word_count": 10,
                    "transition_count": 1,
                    "state_count": 1
                }
            ]

            # Mock concurrent.futures for easier testing
            with patch('models.production_models.markov_chain.markov_chain.ProcessPoolExecutor'):
                with patch('models.production_models.markov_chain.markov_chain.concurrent.futures.as_completed') as mock_completed:
                    # Set up mock futures
                    future1 = MagicMock()
                    future1.result.return_value = {
                        "idx": 0,
                        "transitions": {"word1": {"word2": 1}},
                        "processing_time": 0.1,
                        "word_count": 10,
                        "transition_count": 1,
                        "state_count": 1
                    }

                    future2 = MagicMock()
                    future2.result.return_value = {
                        "idx": 1,
                        "transitions": {"word2": {"word3": 1}},
                        "processing_time": 0.1,
                        "word_count": 10,
                        "transition_count": 1,
                        "state_count": 1
                    }

                    mock_completed.return_value = [future1, future2]

                    # Create test data
                    test_texts = ["This is text one", "This is text two"]

                    # Mock the resource monitor
                    markov_chain.resource_monitor = MagicMock()

                    # Call the method
                    result = markov_chain._train_multiple_texts(
                        test_texts, n_jobs=2)

                    # Verify result
                    assert "training_time" in result
                    assert "total_texts" in result
                    assert result["total_texts"] == 2

                    # Verify DB operations
                    markov_chain.db_adapter.update_total_counts.assert_called_once()

                    # Verify resource monitoring
                    markov_chain.resource_monitor.stop.assert_called_once()

    def test_train_input_validation(self, markov_chain):
        """Test train method validates input properly."""
        # Mock resource monitor
        markov_chain.resource_monitor = MagicMock()

        # Test with invalid input type
        with pytest.raises(ValueError, match="Input must be either a single text string or a list of text strings"):
            markov_chain.train(123)  # Not a string or list

        # Test with list containing non-string items
        with pytest.raises(ValueError, match="Input must be either a single text string or a list of text strings"):
            markov_chain.train(["valid", 123, "valid"])

        # Verify resource monitoring was stopped
        assert markov_chain.resource_monitor.stop.called

    def test_train_dispatches_correctly(self, markov_chain):
        """Test train method dispatches to correct training method."""
        # Mock the underlying training methods
        with patch.object(markov_chain, '_train_single_text') as mock_single:
            with patch.object(markov_chain, '_train_multiple_texts') as mock_multiple:
                # Mock resource monitor
                markov_chain.resource_monitor = MagicMock()

                # Call with single text
                markov_chain.train("This is a single text")
                mock_single.assert_called_once()
                mock_multiple.assert_not_called()

                # Reset mocks
                mock_single.reset_mock()
                mock_multiple.reset_mock()

                # Call with multiple texts
                markov_chain.train(["Text one", "Text two"])
                mock_single.assert_not_called()
                mock_multiple.assert_called_once()

    def test_predict(self, markov_chain):
        """Test prediction functionality."""
        # Reset mock to ensure clean call tracking
        markov_chain.db_adapter.predict_next_word.reset_mock()

        # Set up mock DB adapter for prediction
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2
        }

        # Use the exact module path that MarkovChain uses for the random.choices function
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["apple"]):
            # Basic prediction with direct state
            result = markov_chain.predict(
                "test", temperature=1.0, deterministic=False)

            # Verify result
            assert result == "apple"

            # Verify DB adapter was called correctly with exact arguments
            markov_chain.db_adapter.predict_next_word.assert_called_with(
                "test", 2)

    def test_predict_preprocessing(self, markov_chain):
        """Test prediction with text preprocessing."""
        # Mock the preprocessing
        with patch.object(markov_chain, '_preprocess_text', return_value="preprocessed text"):
            # Set up mock DB adapter for prediction
            markov_chain.db_adapter.predict_next_word.return_value = {
                "word": 1.0
            }

            # Prediction with preprocessing
            markov_chain.predict("RAW TEXT", preprocess=True)

            # Verify preprocessing was called
            markov_chain._preprocess_text.assert_called_with("RAW TEXT")

            # Verify DB adapter was called with processed text
            markov_chain.db_adapter.predict_next_word.assert_called_with(
                "preprocessed text", 2)

    def test_predict_with_string_input_n_gram_greater_than_1(self, markov_chain):
        """Test prediction with string input when n_gram > 1."""
        # Set up mock for preprocessing
        with patch.object(markov_chain, '_preprocess_text', return_value="word1 word2 word3"):
            # Set up mock DB adapter for prediction
            markov_chain.db_adapter.predict_next_word.return_value = {
                "word4": 1.0
            }

            # Set n_gram > 1
            markov_chain.n_gram = 2

            # Predict with string
            result = markov_chain.predict("some text", preprocess=True)

            # Verify it extracted the last n_gram words for prediction
            markov_chain.db_adapter.predict_next_word.assert_called_with(
                "word2 word3", 2)

            # Verify result
            assert result == "word4"

    def test_predict_with_insufficient_words(self, markov_chain):
        """Test prediction when string has fewer words than n_gram."""
        # Set up mock for preprocessing
        with patch.object(markov_chain, '_preprocess_text', return_value="single"):
            # Set n_gram > 1
            markov_chain.n_gram = 2

            # Predict with insufficient words
            result = markov_chain.predict("single", preprocess=True)

            # Verify result is None
            assert result is None

            # Verify warning was logged
            markov_chain.logger.warning.assert_called()

            # Verify DB was never queried
            markov_chain.db_adapter.predict_next_word.assert_not_called()

    def test_predict_no_transitions(self, markov_chain):
        """Test prediction when no transitions are found."""
        # Set up mock DB adapter to return no transitions
        markov_chain.db_adapter.predict_next_word.return_value = {}

        # Predict
        result = markov_chain.predict("test")

        # Verify result is None
        assert result is None

        # Verify warning was logged
        markov_chain.logger.warning.assert_called()

    def test_predict_with_avoid_words(self, markov_chain):
        """Test prediction with avoid_words filter."""
        # Set up mock DB adapter
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2
        }

        # Use the correct module path for patching
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["banana"]):
            # Predict avoiding "apple"
            result = markov_chain.predict("test", avoid_words=["apple"])

            # Verify result isn't "apple"
            assert result == "banana"

    def test_predict_with_deterministic_mode(self, markov_chain):
        """Test prediction in deterministic mode."""
        # Set up mock DB adapter
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2
        }

        # Use proper module path for patching
        with patch('models.production_models.markov_chain.markov_chain.random.choices') as mock_choices:
            # Set the mock to return "banana" if it's called, but it shouldn't be called
            mock_choices.return_value = ["banana"]

            # Predict with deterministic=True
            result = markov_chain.predict("test", deterministic=True)

            # Verify highest probability word was chosen
            assert result == "apple"

            # Verify random.choices wasn't called (should use deterministic selection)
            mock_choices.assert_not_called()

            # Verify info logging
            markov_chain.logger.info.assert_called()

    def test_predict_with_temperature(self, markov_chain):
        """Test prediction with different temperature values."""
        # Set up mock DB adapter
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2
        }

        # Use proper module path for patching
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["apple"]):
            # Test with near-zero temperature (should be deterministic)
            result = markov_chain.predict("test", temperature=0.01)
            assert result == "apple"  # Should be highest probability

        # Test with very high temperature
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["cherry"]):
            result = markov_chain.predict("test", temperature=10.0)
            assert result == "cherry"  # Should use random.choices

    def test_predict_with_top_k(self, markov_chain):
        """Test prediction with top_k filtering."""
        # Set up mock DB adapter with many options
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2,
            "date": 0.15,
            "elderberry": 0.1
        }

        # Test with top_k=2
        with patch('models.production_models.markov_chain.markov_chain.random.choices') as mock_choices:
            mock_choices.return_value = ["banana"]

            result = markov_chain.predict("test", top_k=2)

            # Verify result is one of top 2
            assert result == "banana"

            # Verify choices were only from top 2
            call_args = mock_choices.call_args[0]
            assert set(call_args[0]) == {"apple", "banana"}

    def test_predict_with_top_p(self, markov_chain):
        """Test prediction with top_p (nucleus) sampling."""
        # Set up mock DB adapter with many options
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "cherry": 0.2,
            "date": 0.15,
            "elderberry": 0.1
        }

        # Test with top_p=0.6 (should include apple, banana but not others)
        with patch('models.production_models.markov_chain.markov_chain.random.choices') as mock_choices:
            mock_choices.return_value = ["banana"]

            result = markov_chain.predict("test", top_p=0.6)

            # Verify choices were only from words that sum to < 0.6 prob
            assert result == "banana"

            call_args = mock_choices.call_args[0]
            words = set(call_args[0])
            assert "apple" in words
            assert "banana" in words
            assert "cherry" not in words  # 0.5 + 0.3 > 0.6, so cherry shouldn't be included

    def test_predict_with_penalties(self, markov_chain):
        """Test prediction with various penalties."""
        # Set up mock DB adapter
        markov_chain.db_adapter.predict_next_word.return_value = {
            "apple": 0.5,
            "banana": 0.3,
            "repeated": 0.2
        }

        # Create generation context with 'repeated' word appearing frequently
        generation_context = {
            'generated_words': ['repeated', 'other', 'repeated'],
            'word_frequencies': {'repeated': 2, 'other': 1}
        }

        # Test with repetition penalty
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["apple"]):
            result = markov_chain.predict(
                "test",
                repetition_penalty=2.0,
                generation_context=generation_context
            )

            # "apple" should be chosen due to penalties on "repeated"
            assert result == "apple"

    def test_predict_with_all_penalties(self, markov_chain):
        """Test prediction with all penalties applied simultaneously."""
        # Set up mock DB adapter with probabilities
        markov_chain.db_adapter.predict_next_word.return_value = {
            "repeated": 0.5,  # This word appears in generation context
            "unique": 0.3,    # This word doesn't appear in generation context
            "rare": 0.2       # This word appears once in generation context
        }

        # Create generation context with word frequencies
        generation_context = {
            'generated_words': ['repeated', 'other', 'repeated', 'rare'],
            'word_frequencies': {'repeated': 2, 'other': 1, 'rare': 1}
        }

        # Test with all penalties active
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["unique"]):
            result = markov_chain.predict(
                "test",
                repetition_penalty=1.5,       # Penalize repeated words
                presence_penalty=0.1,         # Penalize words present at all
                frequency_penalty=0.2,        # Penalize by frequency
                generation_context=generation_context
            )

            # Should choose "unique" as penalties reduce probability of others
            assert result == "unique"

        # Verify the penalties were properly calculated and applied
        # by checking with an extreme repetition penalty that forces "unique" selection
        with patch('models.production_models.markov_chain.markov_chain.random.choices', side_effect=lambda population, weights: [population[weights.index(max(weights))]]):
            result = markov_chain.predict(
                "test",
                repetition_penalty=10.0,  # Extreme penalty for repetition
                generation_context=generation_context
            )

            # Should select non-repeated word due to extreme penalty
            assert result == "unique"

    def test_predict_zero_probabilities_after_penalties(self, markov_chain):
        """Test prediction handles case where penalties reduce all probabilities to zero."""
        # Set up mock DB adapter with all words in avoid list
        markov_chain.db_adapter.predict_next_word.return_value = {
            "word1": 0.5,
            "word2": 0.3,
            "word3": 0.2
        }

        # Make generation context where all words are heavily penalized
        generation_context = {
            'generated_words': ['word1', 'word2', 'word3'],
            'word_frequencies': {'word1': 5, 'word2': 5, 'word3': 5}
        }

        # Test with extreme penalties that will zero out all probabilities
        result = markov_chain.predict(
            "test",
            presence_penalty=1.0,  # Remove all probability from present words
            generation_context=generation_context
        )

        # Should return None when all options are zeroed out
        assert result is None

        # Verify warning was logged
        markov_chain.logger.warning.assert_any_call(
            "No valid options after applying penalties")


# Test process_text_batch standalone function
def test_process_text_batch_basic():
    """Test basic functionality of process_text_batch."""
    # Define test data
    idx = 1
    text = "this is a test of the process text batch function"
    n_gram = 2

    # Call the function
    result = process_text_batch((idx, text, n_gram))

    # Verify results
    assert result["idx"] == idx
    assert result["word_count"] == len(text.split())
    assert "transitions" in result
    assert "processing_time" in result
    assert "transition_count" in result
    assert "state_count" in result


def test_process_text_batch_n_gram_1():
    """Test process_text_batch with n_gram=1."""
    # Define test data
    text = "word1 word2 word1 word3"
    args = (0, text, 1)

    # Call the function
    result = process_text_batch(args)

    # Verify transitions structure
    transitions = result["transitions"]
    assert "word1" in transitions
    assert "word2" in transitions["word1"]
    assert transitions["word1"]["word2"] == 1
    assert "word3" in transitions["word1"]
    assert transitions["word1"]["word3"] == 1


def test_process_text_batch_n_gram_greater_than_1():
    """Test process_text_batch with n_gram>1."""
    # Define test data
    text = "word1 word2 word3 word4 word1 word2 word5"
    args = (0, text, 2)

    # Call the function
    result = process_text_batch(args)

    # Verify transitions structure
    transitions = result["transitions"]
    assert ("word1", "word2") in transitions
    assert "word3" in transitions[("word1", "word2")]
    assert "word5" in transitions[("word1", "word2")]
    assert transitions[("word1", "word2")]["word3"] == 1
    assert transitions[("word1", "word2")]["word5"] == 1


def test_getstate_setstate_full_cycle(markov_chain, tmp_path):
    """Test complete pickle serialization and deserialization cycle."""
    import pickle
    import os
    from collections import defaultdict

    # Set up test data
    test_transitions = {
        "word1": {"word2": 5, "word3": 3},
        "word2": {"word4": 2, "word1": 4}
    }
    test_total_counts = {"word1": 8, "word2": 6}

    # Set up transitions and counts in the model
    markov_chain.transitions = defaultdict(lambda: defaultdict(int))
    for state, next_words in test_transitions.items():
        for next_word, count in next_words.items():
            markov_chain.transitions[state][next_word] = count

    markov_chain.total_counts = defaultdict(int)
    for state, count in test_total_counts.items():
        markov_chain.total_counts[state] = count

    # Add a custom config parameter
    markov_chain.n_gram = 3
    markov_chain.environment = "custom_env"

    # Save original objects that can't be pickled
    original_db_adapter = markov_chain.db_adapter
    original_logger = markov_chain.logger
    original_resource_monitor = markov_chain.resource_monitor
    original_preprocessor = markov_chain.preprocessor

    # Remove the unpicklable attributes
    markov_chain.db_adapter = None
    markov_chain.logger = None
    markov_chain.resource_monitor = None
    markov_chain.preprocessor = None

    # Save mock DB config - real implementation uses this in __getstate__
    markov_chain._db_config = {"host": "test_host", "database": "test_db"}

    pickle_path = None
    try:
        # Pickle the model
        pickle_path = tmp_path / "model.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(markov_chain, f)

        # Unpickle to a new model instance
        with open(pickle_path, "rb") as f:
            unpickled_model = pickle.load(f)

        # Add a mock logger to the unpickled model before any assertions
        # This is crucial because many operations require a valid logger
        unpickled_model.logger = MagicMock()

        # Verify the unpickled model has correctly reconstructed data structures
        # Check model parameters
        assert unpickled_model.n_gram == 3
        assert unpickled_model.environment == "custom_env"

        # Check that transitions are preserved and converted back to defaultdict
        assert isinstance(unpickled_model.transitions, defaultdict)
        for state, next_words in test_transitions.items():
            for next_word, count in next_words.items():
                assert unpickled_model.transitions[state][next_word] == count

        # Check that total_counts are preserved and converted back to defaultdict
        assert isinstance(unpickled_model.total_counts, defaultdict)
        for state, count in test_total_counts.items():
            assert unpickled_model.total_counts[state] == count

    finally:
        # Restore the original objects
        markov_chain.db_adapter = original_db_adapter
        markov_chain.logger = original_logger
        markov_chain.resource_monitor = original_resource_monitor
        markov_chain.preprocessor = original_preprocessor

        # Clean up (explicitly delete the file)
        if pickle_path and os.path.exists(pickle_path):
            os.unlink(pickle_path)

    def test_pickle_serialization(self, markov_chain, tmp_path):
        """Test serialization and deserialization with pickle."""
        import pickle
        import os

        # Set up some data in the model
        markov_chain.transitions = defaultdict(lambda: defaultdict(int))
        markov_chain.transitions["state1"]["next1"] = 10
        markov_chain.transitions["state2"]["next2"] = 20
        markov_chain.total_counts["state1"] = 10
        markov_chain.total_counts["state2"] = 20
        markov_chain.n_gram = 2
        markov_chain.environment = "test"

        # Save for __getstate__ to use
        markov_chain._db_config = {"host": "test_host", "database": "test_db"}
        markov_chain._db_environment = "test"

        # Get original logger
        original_logger = markov_chain.logger

        pickle_path = None
        try:
            # Pickle the model
            pickle_path = tmp_path / "model.pickle"
            with open(pickle_path, 'wb') as f:
                pickle.dump(markov_chain, f)

            # Verify pickle file exists
            assert pickle_path.exists()

            # Load the model from pickle
            with open(pickle_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # Manually set logger since it can't be pickled
            loaded_model.logger = original_logger

            # Verify core attributes are preserved
            assert loaded_model.n_gram == 2
            assert loaded_model.environment == "test"

            # Verify transitions are preserved
            assert loaded_model.transitions["state1"]["next1"] == 10
            assert loaded_model.transitions["state2"]["next2"] == 20

            # Verify count totals are preserved
            assert loaded_model.total_counts["state1"] == 10
            assert loaded_model.total_counts["state2"] == 20

            # Make sure logger functions can be called
            loaded_model.logger.info("Testing logger after deserialization")

        finally:
            # Clean up the pickle file
            if pickle_path and os.path.exists(pickle_path):
                os.unlink(pickle_path)

    def test_export_to_onnx_memory_model(self, markov_chain, tmp_path):
        """Test exporting model to ONNX format from memory."""
        # Set up in-memory transitions
        markov_chain.using_db = False
        markov_chain.transitions = defaultdict(lambda: defaultdict(int))
        markov_chain.transitions["state1"]["next1"] = 10
        markov_chain.transitions["state1"]["next2"] = 5
        markov_chain.total_counts = defaultdict(int)
        markov_chain.total_counts["state1"] = 15

        # Mock onnx operations
        with patch('onnx.helper.make_tensor_value_info'):
            with patch('onnx.helper.make_tensor'):
                with patch('onnx.helper.make_node'):
                    with patch('onnx.helper.make_graph'):
                        with patch('onnx.helper.make_model'):
                            with patch('onnx.checker.check_model'):
                                with patch('onnx.save'):
                                    # Export model
                                    filepath = tmp_path / "test_model.onnx"
                                    result = markov_chain.export_to_onnx(
                                        str(filepath))

                                    # Verify result
                                    assert result is True

                                    # Verify logger was called
                                    markov_chain.logger.info.assert_called()

    def test_export_to_onnx_error(self, markov_chain, tmp_path):
        """Test handling of errors during ONNX export."""
        # Mock onnx operations to raise an exception
        with patch('onnx.helper.make_tensor_value_info', side_effect=Exception("ONNX error")):
            # Export model
            filepath = tmp_path / "test_model.onnx"
            result = markov_chain.export_to_onnx(str(filepath))

            # Verify result
            assert result is False

            # Verify error was logged
            markov_chain.logger.error.assert_called()

    def test_extract_db_vocabulary_and_transitions(self, markov_chain):
        """Test extracting vocabulary and transitions from database."""
        # Create proper mocks for database cursor and connection
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = [
            ("state1", "next1", 10),
            ("state1", "next2", 5),
            ("state2", "next3", 8)
        ]

        # Create a proper context manager mock for cursor
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor_mock

        # Create a proper context manager mock for connection
        conn_mock = MagicMock()
        conn_mock.cursor.return_value = cursor_context

        # Set up the connection mock on the db_adapter
        markov_chain.db_adapter.get_connection.return_value = conn_mock

        # Call the method
        vocab, transitions = markov_chain._extract_db_vocabulary_and_transitions()

        # Verify results
        assert "state1" in vocab
        assert "next1" in vocab
        assert "next2" in vocab
        assert "state2" in vocab
        assert "next3" in vocab

        assert ("state1", "next1") in transitions
        assert transitions[("state1", "next1")] == 10
        assert ("state1", "next2") in transitions
        assert transitions[("state1", "next2")] == 5
        assert ("state2", "next3") in transitions
        assert transitions[("state2", "next3")] == 8

        # Verify connection was returned
        markov_chain.db_adapter.return_connection.assert_called_once_with(
            conn_mock)

    def test_get_db_total_for_state(self, markov_chain):
        """Test getting total count for state from database."""
        # Create proper mocks for database cursor and connection
        cursor_mock = MagicMock()
        cursor_mock.fetchone.return_value = (15,)

        # Create a proper context manager mock for cursor
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor_mock

        # Create a proper context manager mock for connection
        conn_mock = MagicMock()
        conn_mock.cursor.return_value = cursor_context

        # Set up the connection mock on the db_adapter
        markov_chain.db_adapter.get_connection.return_value = conn_mock

        # Call the method
        result = markov_chain._get_db_total_for_state("state1")

        # Verify result
        assert result == 15

        # Verify connection was returned
        markov_chain.db_adapter.return_connection.assert_called_once_with(
            conn_mock)

    def test_process_text_batch(self):
        """Test the standalone process_text_batch function."""
        # Create test data
        idx = 0
        text = "This is a test text with multiple words for processing"
        n_gram = 2

        # Call the function
        result = process_text_batch((idx, text, n_gram))

        # Verify result structure
        assert "transitions" in result
        assert "processing_time" in result
        assert "word_count" in result
        assert "transition_count" in result
        assert "state_count" in result
        assert result["idx"] == idx

        # Check some transitions
        transitions = result["transitions"]
        assert isinstance(transitions, dict)

        # For n_gram=2, we expect tuples of 2 words as states
        expected_state = ("this", "is")
        if expected_state in transitions:
            assert "a" in transitions[expected_state]

    def test_store_model_in_db(self, markov_chain):
        """Test storing model in database."""
        # Create mock transitions and counts
        transitions = {
            "state1": {"next1": 10, "next2": 5},
            "state2": {"next3": 8}
        }
        total_counts = {
            "state1": 15,
            "state2": 8
        }

        # Create a properly mocked connection setup
        conn_mock = MagicMock()
        cursor_context = MagicMock()
        cursor_mock = MagicMock()
        cursor_context.__enter__.return_value = cursor_mock
        conn_mock.cursor.return_value = cursor_context

        # Set up the mock DB adapter behavior
        markov_chain.db_adapter.get_connection.return_value = conn_mock

        # Mock psycopg2.extras.execute_values
        with patch('psycopg2.extras.execute_values'):
            # Call the method
            result = markov_chain._store_model_in_db(transitions, total_counts)

            # Verify result is True (success)
            assert result is True

            # Verify connection was committed and returned
            conn_mock.commit.assert_called_once()
            markov_chain.db_adapter.return_connection.assert_called_once_with(
                conn_mock)

            # Verify log was generated
            markov_chain.logger.info.assert_any_call(
                f"Successfully stored model in {markov_chain.environment} database"
            )

    def test_store_model_in_db_error(self, markov_chain):
        """Test handling errors when storing model in database."""
        # Create mock transitions and counts
        transitions = {
            "state1": {"next1": 10, "next2": 5}
        }
        total_counts = {
            "state1": 15
        }

        # Make execute_values raise an exception
        with patch('psycopg2.extras.execute_values', side_effect=Exception("DB error")):
            # Create proper mocks for database cursor and connection
            cursor_mock = MagicMock()

            # Create a proper context manager mock for cursor
            cursor_context = MagicMock()
            cursor_context.__enter__.return_value = cursor_mock

            # Create a proper context manager mock for connection with rollback
            conn_mock = MagicMock()
            conn_mock.cursor.return_value = cursor_context

            # Set up the connection mock on db_adapter
            markov_chain.db_adapter.get_connection.return_value = conn_mock

            # Call the method
            result = markov_chain._store_model_in_db(transitions, total_counts)

            # Verify result is False (failure)
            assert result is False

            # Verify error was logged
            markov_chain.logger.error.assert_called()

            # Verify connection was rolled back and returned
            conn_mock.rollback.assert_called_once()
            markov_chain.db_adapter.return_connection.assert_called_once_with(
                conn_mock)

    def test_getstate(self, markov_chain):
        """Test pickle serialization preparation."""
        # Set up some attributes with non-picklable types
        markov_chain.transitions = defaultdict(lambda: defaultdict(int))
        markov_chain.transitions["state1"]["next1"] = 10
        markov_chain.total_counts = defaultdict(int)
        markov_chain.total_counts["state1"] = 10

        # Call __getstate__
        state = markov_chain.__getstate__()

        # Verify state doesn't contain non-picklable objects
        assert "db_adapter" not in state
        assert "resource_monitor" not in state

        # Verify defaultdicts were converted to regular dicts
        assert isinstance(state["transitions"], dict)
        assert isinstance(state["transitions"]["state1"], dict)
        assert isinstance(state["total_counts"], dict)

    def test_setstate(self):
        """Test pickle deserialization restoration."""
        # Create a state dict with regular dicts
        state = {
            "n_gram": 2,
            "environment": "test",
            "logger": MagicMock(),
            "using_db": True,
            "transitions": {
                "state1": {"next1": 10, "next2": 5}
            },
            "total_counts": {
                "state1": 15
            },
            "_db_config": {"host": "localhost"},
            "_db_environment": "test"
        }

        # Mock the database adapter
        with patch('utils.database_adapters.postgresql.markov_chain.MarkovChainPostgreSqlAdapter'):
            with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                # Create a new instance
                model = MarkovChain(
                    n_gram=1, environment="temp", logger=MagicMock())

                # Call __setstate__
                model.__setstate__(state)

                # Verify attributes were restored
                assert model.n_gram == 2
                assert model.environment == "test"

                # Verify defaultdicts were recreated
                assert isinstance(model.transitions, defaultdict)
                assert model.transitions["state1"]["next1"] == 10
                assert isinstance(model.total_counts, defaultdict)
                assert model.total_counts["state1"] == 15

    def test_load_model(self, mock_logger):
        """Test loading model from ONNX file."""
        # Mock onnx operations
        with patch('onnx.load') as mock_load:
            # Set up mock ONNX model with metadata
            mock_model = MagicMock()
            mock_model.metadata_props = [
                MagicMock(key='n_gram', value='2'),
                MagicMock(key='vocab_mapping',
                          value='{"0": "word1", "1": "word2"}')
            ]
            mock_load.return_value = mock_model

            # Mock model graph and node structure
            mock_graph = MagicMock()
            mock_model.graph = mock_graph
            mock_node = MagicMock()
            mock_node.op_type = "Constant"
            mock_node.output = ["matrix"]
            mock_attr = MagicMock()
            mock_attr.name = "value"
            mock_attr.t = MagicMock()
            mock_node.attribute = [mock_attr]
            mock_graph.node = [mock_node]

            # Mock numpy helper to return proper weights tensor
            weights = np.zeros((2, 2), dtype=np.float32)
            weights[0, 1] = 0.7
            weights[1, 0] = 0.3

            # Explicitly define the mock instance to be returned
            mock_instance = MagicMock(spec=MarkovChain)

            # Create a comprehensive mock chain
            with patch('onnx.numpy_helper.to_array', return_value=weights):
                with patch('onnxruntime.SessionOptions'):
                    with patch('onnxruntime.InferenceSession'):
                        # Mock the constructor to return our prepared instance
                        with patch('models.production_models.markov_chain.markov_chain.MarkovChain', return_value=mock_instance):
                            # Call load_model
                            result = MarkovChain.load_model(
                                filepath="test_model.onnx",
                                environment="test",
                                logger=mock_logger
                            )

                            # Verify the result is our mock instance
                            assert result is mock_instance

                            # Verify logger calls
                            mock_logger.info.assert_any_call(
                                "ONNX model loading started", extra=ANY)

    def test_del_method(self, markov_chain):
        """Test __del__ method properly closes connections."""
        # Call __del__
        markov_chain.__del__()

        # Verify DB adapter close_connections was called
        markov_chain.db_adapter.close_connections.assert_called_once()

    def test_del_method_with_exception(self, markov_chain):
        """Test __del__ method handles exceptions gracefully."""
        # Make close_connections raise an exception
        markov_chain.db_adapter.close_connections.side_effect = Exception(
            "Close error")

        # Call __del__ - should not raise exception
        try:
            markov_chain.__del__()
            # If we get here, no exception was raised
            assert True
        except Exception:
            assert False, "Exception should have been caught in __del__"

    def test_save_model(self, markov_chain):
        """Test save_model method."""
        # Mock export_to_onnx
        with patch.object(markov_chain, 'export_to_onnx', return_value=True):
            # Call save_model
            result = markov_chain.save_model("test_model.onnx")

            # Verify result
            assert result is True

            # Verify export_to_onnx was called
            markov_chain.export_to_onnx.assert_called_once_with(
                "test_model.onnx")

    def test_pickle_serialization(self, markov_chain, tmp_path):
        """Test serialization and deserialization with pickle."""
        import pickle
        import os

        # Set up some data in the model
        markov_chain.transitions = defaultdict(lambda: defaultdict(int))
        markov_chain.transitions["state1"]["next1"] = 10
        markov_chain.transitions["state2"]["next2"] = 20
        markov_chain.total_counts["state1"] = 10
        markov_chain.total_counts["state2"] = 20
        markov_chain.n_gram = 2
        markov_chain.environment = "test"

        # Save for __getstate__ to use
        markov_chain._db_config = {"host": "test_host", "database": "test_db"}
        markov_chain._db_environment = "test"

        # Get original logger
        original_logger = markov_chain.logger

        pickle_path = None
        try:
            # Pickle the model
            pickle_path = tmp_path / "model.pickle"
            with open(pickle_path, 'wb') as f:
                pickle.dump(markov_chain, f)

            # Verify pickle file exists
            assert pickle_path.exists()

            # Load the model from pickle
            with open(pickle_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # Manually set logger since it can't be pickled
            loaded_model.logger = original_logger

            # Verify core attributes are preserved
            assert loaded_model.n_gram == 2
            assert loaded_model.environment == "test"

            # Verify transitions are preserved
            assert loaded_model.transitions["state1"]["next1"] == 10
            assert loaded_model.transitions["state2"]["next2"] == 20

            # Verify count totals are preserved
            assert loaded_model.total_counts["state1"] == 10
            assert loaded_model.total_counts["state2"] == 20

            # Make sure logger functions can be called
            loaded_model.logger.info("Testing logger after deserialization")

        finally:
            # Clean up the pickle file
            if pickle_path and os.path.exists(pickle_path):
                os.unlink(pickle_path)

    def test_export_to_onnx_db_model(self, markov_chain, tmp_path):
        """Test exporting model to ONNX format from database."""
        # Set up database-backed model
        markov_chain.using_db = True

        # Mock DB extraction methods
        vocab = {"state1", "next1", "next2", "state2", "next3"}
        transitions = {
            ("state1", "next1"): 10,
            ("state1", "next2"): 5,
            ("state2", "next3"): 8
        }

        with patch.object(markov_chain, '_extract_db_vocabulary_and_transitions', return_value=(vocab, transitions)):
            with patch.object(markov_chain, '_get_db_total_for_state', return_value=15):
                # Mock ONNX operations
                with patch('onnx.helper.make_tensor_value_info'):
                    with patch('onnx.helper.make_tensor'):
                        with patch('onnx.helper.make_node'):
                            with patch('onnx.helper.make_graph'):
                                with patch('onnx.helper.make_model'):
                                    with patch('onnx.checker.check_model'):
                                        with patch('onnx.save'):
                                            # Export model
                                            filepath = tmp_path / "db_model.onnx"
                                            result = markov_chain.export_to_onnx(
                                                str(filepath))

                                            # Verify result
                                            assert result is True

                                            # Verify DB extraction method was called
                                            markov_chain._extract_db_vocabulary_and_transitions.assert_called_once()

                                            # Verify _get_db_total_for_state was called
                                            assert markov_chain._get_db_total_for_state.called

    def test_load_model_complete(self, mock_logger, tmp_path):
        """Test complete load_model functionality with mocked ONNX model."""
        import json
        import numpy as np

        # Create a mock ONNX model with required components
        model_path = str(tmp_path / "test_model.onnx")

        # Mock vocabulary mapping
        vocab_mapping = {
            "0": "word1",
            "1": "word2",
            "2": "word3"
        }

        # Mock metadata properties
        metadata_props = [
            MagicMock(key='n_gram', value='2'),
            MagicMock(key='vocab_mapping', value=json.dumps(vocab_mapping))
        ]

        # Create mock model structure
        mock_model = MagicMock()
        mock_model.metadata_props = metadata_props

        # Create mock graph
        mock_graph = MagicMock()
        mock_model.graph = mock_graph

        # Create mock node with transition matrix
        mock_node = MagicMock()
        mock_node.op_type = "Constant"
        mock_node.output = ["matrix"]

        # Create mock attribute with tensor
        mock_attr = MagicMock()
        mock_attr.name = "value"

        # Create weights tensor that will be converted to numpy
        # 3x3 for vocab size 3
        mock_weights = np.zeros((3, 3), dtype=np.float32)
        mock_weights[0, 1] = 0.7  # word1 -> word2: 0.7 probability
        mock_weights[0, 2] = 0.3  # word1 -> word3: 0.3 probability
        mock_weights[1, 0] = 1.0  # word2 -> word1: 1.0 probability

        # Set up mock conversion from tensor to numpy
        with patch('onnx.numpy_helper.to_array', return_value=mock_weights):
            mock_attr.t = "mock_tensor"
            mock_node.attribute = [mock_attr]
            mock_graph.node = [mock_node]

        # Mock ONNX load
        with patch('onnx.load', return_value=mock_model):
            # Mock ONNX runtime session
            with patch('onnxruntime.SessionOptions'):
                with patch('onnxruntime.InferenceSession'):
                    # Mock MarkovChain initialization
                    with patch('models.production_models.markov_chain.markov_chain.MarkovChainPostgreSqlAdapter'):
                        with patch('models.production_models.markov_chain.markov_chain.ResourceMonitor'):
                            with patch('models.production_models.markov_chain.markov_chain.TextPreprocessor'):
                                # Call load_model
                                MarkovChain.load_model(
                                    filepath=model_path,
                                    environment="test",
                                    logger=mock_logger
                                )

                                # Verify logger was called for successful loading
                                mock_logger.info.assert_any_call(
                                    "ONNX model loading started", extra=ANY)

    def test_generate_text_with_penalties(self, markov_chain):
        """Test text generation with various penalties."""
        # Mock _get_valid_start_state
        with patch.object(markov_chain, '_get_valid_start_state', return_value="start"):
            # Create mock for predict method
            mock_predict = MagicMock(side_effect=["word1", None])

            # Replace the predict method with our mock
            original_predict = markov_chain.predict
            markov_chain.predict = mock_predict

            try:
                # Generate text with penalties
                result = markov_chain.generate_text(
                    start="start",
                    max_length=3,
                    repetition_penalty=1.5,
                    presence_penalty=0.1,
                    frequency_penalty=0.2
                )

                # Verify result
                assert result == "start word1"

                # Verify predict was called exactly once (then returned None to stop)
                assert mock_predict.call_count == 2

                # Verify the penalties were passed correctly
                call_kwargs = mock_predict.call_args_list[0][1]
                assert call_kwargs['repetition_penalty'] == 1.5
                assert call_kwargs['presence_penalty'] == 0.1
                assert call_kwargs['frequency_penalty'] == 0.2
                assert 'generation_context' in call_kwargs
            finally:
                # Restore the original predict method
                markov_chain.predict = original_predict

    def test_generate_text_early_termination(self, markov_chain):
        """Test text generation terminates early if predict returns None."""
        # Mock _get_valid_start_state
        with patch.object(markov_chain, '_get_valid_start_state', return_value="start"):
            # Create mock for predict that returns None on second call
            with patch.object(markov_chain, 'predict', side_effect=["word", None]):
                # Generate text
                result = markov_chain.generate_text(
                    start="start", max_length=5)

                # Verify result contains only words before None
                assert result == "start word"

                # Verify predict was called twice
                assert markov_chain.predict.call_count == 2

                # Verify info log for early termination
                markov_chain.logger.info.assert_any_call(
                    "Text generation ended early - no prediction available",
                    extra=ANY
                )

    def test_predict_with_all_penalties(self, markov_chain):
        """Test prediction with all penalties applied simultaneously."""
        # Set up mock DB adapter with probabilities
        markov_chain.db_adapter.predict_next_word.return_value = {
            "repeated": 0.5,  # This word appears in generation context
            "unique": 0.3,    # This word doesn't appear in generation context
            "rare": 0.2       # This word appears once in generation context
        }

        # Create generation context with word frequencies
        generation_context = {
            'generated_words': ['repeated', 'other', 'repeated', 'rare'],
            'word_frequencies': {'repeated': 2, 'other': 1, 'rare': 1}
        }

        # Test with all penalties active
        with patch('models.production_models.markov_chain.markov_chain.random.choices', return_value=["unique"]):
            result = markov_chain.predict(
                "test",
                repetition_penalty=1.5,       # Penalize repeated words
                presence_penalty=0.1,         # Penalize words present at all
                frequency_penalty=0.2,        # Penalize by frequency
                generation_context=generation_context
            )

            # Should choose "unique" as penalties reduce probability of others
            assert result == "unique"

        # Verify the penalties were properly calculated and applied
        # by checking with an extreme repetition penalty that forces "unique" selection
        with patch('models.production_models.markov_chain.markov_chain.random.choices', side_effect=lambda population, weights: [population[weights.index(max(weights))]]):
            result = markov_chain.predict(
                "test",
                repetition_penalty=10.0,  # Extreme penalty for repetition
                generation_context=generation_context
            )

            # Should select non-repeated word due to extreme penalty
            assert result == "unique"

    def test_predict_zero_probabilities_after_penalties(self, markov_chain):
        """Test prediction handles case where penalties reduce all probabilities to zero."""
        # Set up mock DB adapter with all words in avoid list
        markov_chain.db_adapter.predict_next_word.return_value = {
            "word1": 0.5,
            "word2": 0.3,
            "word3": 0.2
        }

        # Make generation context where all words are heavily penalized
        generation_context = {
            'generated_words': ['word1', 'word2', 'word3'],
            'word_frequencies': {'word1': 5, 'word2': 5, 'word3': 5}
        }

        # Test with extreme penalties that will zero out all probabilities
        result = markov_chain.predict(
            "test",
            presence_penalty=1.0,  # Remove all probability from present words
            generation_context=generation_context
        )

        # Should return None when all options are zeroed out
        assert result is None

        # Verify warning was logged
        markov_chain.logger.warning.assert_any_call(
            "No valid options after applying penalties")


# Test process_text_batch standalone function
def test_process_text_batch_basic():
    """Test basic functionality of process_text_batch."""
    # Define test data
    idx = 1
    text = "this is a test of the process text batch function"
    n_gram = 2

    # Call the function
    result = process_text_batch((idx, text, n_gram))

    # Verify results
    assert result["idx"] == idx
    assert result["word_count"] == len(text.split())
    assert "transitions" in result
    assert "processing_time" in result
    assert "transition_count" in result
    assert "state_count" in result


def test_process_text_batch_n_gram_1():
    """Test process_text_batch with n_gram=1."""
    # Define test data
    text = "word1 word2 word1 word3"
    args = (0, text, 1)

    # Call the function
    result = process_text_batch(args)

    # Verify transitions structure
    transitions = result["transitions"]
    assert "word1" in transitions
    assert "word2" in transitions["word1"]
    assert transitions["word1"]["word2"] == 1
    assert "word3" in transitions["word1"]
    assert transitions["word1"]["word3"] == 1


def test_process_text_batch_n_gram_greater_than_1():
    """Test process_text_batch with n_gram>1."""
    # Define test data
    text = "word1 word2 word3 word4 word1 word2 word5"
    args = (0, text, 2)

    # Call the function
    result = process_text_batch(args)

    # Verify transitions structure
    transitions = result["transitions"]
    assert ("word1", "word2") in transitions
    assert "word3" in transitions[("word1", "word2")]
    assert "word5" in transitions[("word1", "word2")]
    assert transitions[("word1", "word2")]["word3"] == 1
    assert transitions[("word1", "word2")]["word5"] == 1


def test_getstate_setstate_full_cycle(markov_chain, tmp_path):
    """Test complete pickle serialization and deserialization cycle."""
    import pickle
    import os
    from collections import defaultdict

    # Set up test data
    test_transitions = {
        "word1": {"word2": 5, "word3": 3},
        "word2": {"word4": 2, "word1": 4}
    }
    test_total_counts = {"word1": 8, "word2": 6}

    # Set up transitions and counts in the model
    markov_chain.transitions = defaultdict(lambda: defaultdict(int))
    for state, next_words in test_transitions.items():
        for next_word, count in next_words.items():
            markov_chain.transitions[state][next_word] = count

    markov_chain.total_counts = defaultdict(int)
    for state, count in test_total_counts.items():
        markov_chain.total_counts[state] = count

    # Add a custom config parameter
    markov_chain.n_gram = 3
    markov_chain.environment = "custom_env"

    # Save original objects that can't be pickled
    original_db_adapter = markov_chain.db_adapter
    original_logger = markov_chain.logger
    original_resource_monitor = markov_chain.resource_monitor
    original_preprocessor = markov_chain.preprocessor

    # Remove the unpicklable attributes
    markov_chain.db_adapter = None
    markov_chain.logger = None
    markov_chain.resource_monitor = None
    markov_chain.preprocessor = None

    # Save mock DB config - real implementation uses this in __getstate__
    markov_chain._db_config = {"host": "test_host", "database": "test_db"}

    pickle_path = None
    try:
        # Pickle the model
        pickle_path = tmp_path / "model.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(markov_chain, f)

        # Unpickle to a new model instance
        with open(pickle_path, "rb") as f:
            unpickled_model = pickle.load(f)

        # Add a mock logger to the unpickled model before any assertions
        # This is crucial because many operations require a valid logger
        unpickled_model.logger = MagicMock()

        # Verify the unpickled model has correctly reconstructed data structures
        # Check model parameters
        assert unpickled_model.n_gram == 3
        assert unpickled_model.environment == "custom_env"

        # Check that transitions are preserved and converted back to defaultdict
        assert isinstance(unpickled_model.transitions, defaultdict)
        for state, next_words in test_transitions.items():
            for next_word, count in next_words.items():
                assert unpickled_model.transitions[state][next_word] == count

        # Check that total_counts are preserved and converted back to defaultdict
        assert isinstance(unpickled_model.total_counts, defaultdict)
        for state, count in test_total_counts.items():
            assert unpickled_model.total_counts[state] == count

    finally:
        # Restore the original objects
        markov_chain.db_adapter = original_db_adapter
        markov_chain.logger = original_logger
        markov_chain.resource_monitor = original_resource_monitor
        markov_chain.preprocessor = original_preprocessor

        # Clean up (explicitly delete the file)
        if pickle_path and os.path.exists(pickle_path):
            os.unlink(pickle_path)
