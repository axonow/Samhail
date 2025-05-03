import os
import time
import json
import random
import datetime
import multiprocessing
import concurrent.futures
import numpy as np
import psutil
import sys
from math import ceil

from utils.loggers.json_logger import get_logger
from utils.system_monitoring import ResourceMonitor
from utils.database_adapters.postgresql.markov_chain import MarkovChainPostgreSqlAdapter
from collections import defaultdict
from psycopg2.extras import execute_values
from onnx import helper, TensorProto
import onnx
import onnxruntime as ort
from concurrent.futures import ProcessPoolExecutor
from data_preprocessing.text_preprocessor import TextPreprocessor

# Define a standalone preprocessing function for parallel processing


def process_text_batch(args):
    """
    Process a text batch for distributed training.
    This is a standalone function outside of the class to avoid pickling issues.

    Args:
        args (tuple): A tuple containing (idx, text, n_gram)

    Returns:
        dict: A dictionary with processed transitions and statistics
    """
    idx, text, n_gram = args
    start_time = time.time()

    # Always preprocess text with basic operations that can be pickled
    # Convert to lowercase and handle basic whitespace
    text = text.lower()
    text = ' '.join(text.split())

    words = text.split()

    # Extract transitions but don't update model yet - using standard dict instead of defaultdict
    local_transitions = {}
    for i in range(len(words) - n_gram):
        if n_gram == 1:
            state = words[i]
        else:
            state = tuple(words[i: i + n_gram])

        next_word = words[i + n_gram]

        # Create nested dictionary structure without using defaultdict
        if state not in local_transitions:
            local_transitions[state] = {}

        if next_word not in local_transitions[state]:
            local_transitions[state][next_word] = 0

        local_transitions[state][next_word] += 1

    # Progress info
    processing_time = time.time() - start_time
    transition_count = sum(len(next_words)
                           for next_words in local_transitions.values())

    return {
        "idx": idx,
        "transitions": local_transitions,
        "processing_time": processing_time,
        "word_count": len(words),
        "transition_count": transition_count,
        "state_count": len(local_transitions)
    }


class MarkovChain:
    """
    A hybrid implementation of a Markov Chain for text generation.

    This class implements an n-gram Markov model that can use either in-memory
    storage or PostgreSQL database storage based on data size and configuration.
    """

    def __init__(
        self,
        n_gram=1,
        db_config=None,
        environment="development",
        logger=None
    ):
        """
        Initializes the Markov Chain model with database storage for large-scale data.

        Args:
            n_gram (int): Number of words to consider as a state (default: 1)
            db_config (dict, optional): PostgreSQL configuration dictionary
            environment (str): Which environment to use ('development' or 'test')
            logger (Logger, required): Logger instance for logging model activities
        """
        self.n_gram = n_gram
        self.environment = environment
        self.db_adapter = None

        # Add default attributes needed by various methods
        # Use standard defaultdict initialization - __getstate__ will handle serialization
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.using_db = True  # Default to database storage

        # Ensure logger is provided
        if logger is None:
            raise ValueError("Logger instance must be provided")
        self.logger = logger

        # Initialize resource monitor for system monitoring
        self.resource_monitor = ResourceMonitor(
            logger=self.logger,
            monitoring_interval=10.0  # Log metrics every 10 seconds
        )

        # Initialize text preprocessor
        self.preprocessor = None
        try:
            self.preprocessor = TextPreprocessor()
            self.logger.info("TextPreprocessor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize TextPreprocessor: {e}")
            raise ValueError(
                "TextPreprocessor initialization failed, which is required for processing")

        # Log initialization details with system info
        self.logger.info("MarkovChain initialized", extra={
            "metrics": {
                "n_gram": n_gram,
                "environment": environment,
                "preprocessor_available": self.preprocessor is not None,
                "system": self.resource_monitor.get_resource_usage()
            }
        })

        # Initialize database adapter - this is mandatory
        try:
            # Initialize the database adapter
            if db_config is not None:
                self.db_adapter = MarkovChainPostgreSqlAdapter(
                    environment=environment,
                    logger=logger,
                    db_config=db_config
                )
            else:
                self.db_adapter = MarkovChainPostgreSqlAdapter(
                    environment=environment,
                    logger=logger
                )

            # Check if the adapter is usable
            if self.db_adapter and self.db_adapter.is_usable():
                self.logger.info("Database adapter initialized and ready to use", extra={
                    "metrics": {
                        "environment": self.environment,
                    }
                })
            else:
                error_msg = "Database adapter is not usable - required for large-scale model training"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Database adapter initialization failed: {str(e)}"
            self.logger.error(error_msg, extra={
                "metrics": {"error": str(e)}
            })
            raise ValueError(error_msg)

    def train(self, text_input, clear_previous=True, n_jobs=-1):
        """
        Trains the Markov Chain model using database storage. 
        Automatically handles single text or multiple texts with parallel processing.

        Args:
            text_input (str or list): Either a single text string or a list of text strings to train on
            clear_previous (bool): Whether to clear previous training data
            n_jobs (int): Number of parallel jobs when training on multiple texts (-1 for all cores)

        Returns:
            dict: Training statistics
        """
        # Start resource monitoring for training
        self.resource_monitor.start("markov_training")

        # Determine if we're dealing with a single text or multiple texts
        if isinstance(text_input, str):
            return self._train_single_text(text_input, clear_previous)
        elif isinstance(text_input, list) and all(isinstance(t, str) for t in text_input):
            return self._train_multiple_texts(text_input, clear_previous, n_jobs)
        else:
            error_msg = "Input must be either a single text string or a list of text strings"
            self.logger.error(error_msg)
            self.resource_monitor.stop()
            raise ValueError(error_msg)

    def _train_single_text(self, text, clear_previous=True):
        """
        Internal method to train on a single text string.

        Args:
            text (str): The input text used to train the model
            clear_previous (bool): Whether to clear previous training data

        Returns:
            dict: Training statistics
        """
        # Always preprocess the text
        text = self._preprocess_text(text)

        words = text.split()
        if len(words) < self.n_gram + 1:
            # If there aren't enough words, no transitions can be learned
            self.logger.warning("Text too short for training", extra={
                "metrics": {
                    "n_gram": self.n_gram,
                    "text_length": len(words),
                    "min_required": self.n_gram + 1,
                    "system": self.resource_monitor.get_resource_usage()
                }
            })
            self.resource_monitor.stop()
            return {"error": "Text too short for training"}

        # Log training parameters with system metrics
        unique_words_estimate = len(set(words))
        estimated_transitions = unique_words_estimate**2

        self.resource_monitor.log_progress(
            "Training started",
            operation="markov_training_start",
            extra_metrics={
                "text_sample": text[:100] + ("..." if len(text) > 100 else ""),
                "word_count": len(words),
                "unique_words": unique_words_estimate,
                "estimated_transitions": estimated_transitions,
                "storage": "database",
                "n_gram": self.n_gram,
                "clear_previous": clear_previous
            }
        )

        # Clear previous data if requested
        if clear_previous:
            self.db_adapter.clear_tables(self.n_gram)
            self.logger.info("Cleared previous database training data", extra={
                "metrics": {"environment": self.environment, "n_gram": self.n_gram}
            })

        # Process transitions in batches
        batch_size = 5000
        transitions_batch = []
        total_transitions = 0
        start_time = time.time()

        for i in range(len(words) - self.n_gram):
            if self.n_gram == 1:
                current_state = words[i]
                next_word = words[i + 1]
            else:
                current_state = " ".join(words[i: i + self.n_gram])
                next_word = words[i + self.n_gram]

            # Add to batch
            transitions_batch.append((current_state, next_word, 1))

            # Process batch when it reaches the threshold
            if len(transitions_batch) >= batch_size:
                self.db_adapter.insert_transitions_batch(
                    transitions_batch, self.n_gram)
                total_transitions += len(transitions_batch)
                transitions_batch = []

                # Log batch processing
                if total_transitions % (batch_size * 5) == 0:
                    self.logger.info("Database training progress", extra={
                        "metrics": {
                            "transitions_processed": total_transitions,
                            "environment": self.environment,
                            "n_gram": self.n_gram,
                        }
                    })

        # Insert any remaining transitions
        if transitions_batch:
            self.db_adapter.insert_transitions_batch(
                transitions_batch, self.n_gram)
            total_transitions += len(transitions_batch)

        # Update total counts
        self.db_adapter.update_total_counts(self.n_gram)

        # Calculate statistics
        training_time = time.time() - start_time
        transitions_per_second = total_transitions / \
            training_time if training_time > 0 else 0

        # Get database statistics
        try:
            db_stats = self.db_adapter.get_model_statistics(self.n_gram)
            total_states = db_stats.get("states_count", 0)
        except Exception as e:
            self.logger.error(f"Error getting DB statistics: {e}")
            total_states = 0

        # Log database training completion
        self.logger.info("Database training completed", extra={
            "metrics": {
                "environment": self.environment,
                "n_gram": self.n_gram,
                "total_transitions": total_transitions,
                "total_states": total_states,
                "training_time_seconds": training_time
            }
        })

        # Log training completion with final system metrics
        self.resource_monitor.log_progress(
            "Training completed",
            operation="markov_training_complete",
            extra_metrics={
                "storage": "database",
                "n_gram": self.n_gram,
                "word_count": len(words),
                "total_transitions": total_transitions,
                "training_time_seconds": training_time
            }
        )

        # Stop resource monitoring
        self.resource_monitor.stop()

        return {
            "training_time": training_time,
            "word_count": len(words),
            "total_states": total_states,
            "total_transitions": total_transitions,
            "transitions_per_second": transitions_per_second
        }

    def _train_multiple_texts(self, texts, clear_previous=True, n_jobs=-1):
        """
        Internal method to train on multiple texts using parallel processing.

        Args:
            texts (list): List of text strings to train on
            clear_previous (bool): Whether to clear previous training data
            n_jobs (int): Number of parallel jobs (-1 for all cores)

        Returns:
            dict: Training statistics
        """
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()

        # Clear if requested
        if clear_previous:
            self.db_adapter.clear_tables(self.n_gram)
            self.logger.info("Cleared database tables for training", extra={
                "metrics": {"environment": self.environment, "n_gram": self.n_gram}
            })

        # Log start of parallel training
        total_texts = len(texts)
        total_chars = sum(len(text) for text in texts)
        avg_chars = total_chars // total_texts if total_texts > 0 else 0

        self.logger.info("Parallel training started", extra={
            "metrics": {
                "n_jobs": n_jobs,
                "total_texts": total_texts,
                "total_chars": total_chars,
                "avg_chars_per_text": avg_chars,
                "n_gram": self.n_gram,
                "storage": "database"
            }
        })

        # Estimate time based on text volume (very rough estimate)
        estimated_time_per_text_ms = 500  # milliseconds per text
        estimated_total_time = (
            estimated_time_per_text_ms * total_texts) / (1000 * n_jobs)
        self.logger.info(
            f"Estimated training time: {estimated_total_time:.1f} seconds")

        # Create batch tracking variables
        batch_size = max(100, ceil(total_texts / 100))
        last_report_time = time.time()
        report_interval = 2.0  # seconds between progress reports
        processed_count = 0
        total_transitions = 0
        total_states = 0
        start_time = time.time()

        # Create argument tuples for the standalone processing function
        process_args = [
            (i, text, self.n_gram)
            for i, text in enumerate(texts)
        ]

        # Process texts in parallel with progress reporting
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_text_batch, arg)
                       for arg in process_args]

            # Track and combine results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()

                    # Update model with this batch's transitions
                    for state, next_words in result["transitions"].items():
                        for next_word, count in next_words.items():
                            if isinstance(state, tuple):
                                db_state = " ".join(state)
                            else:
                                db_state = str(state)
                            self.db_adapter.increment_transition(
                                db_state, next_word, count, self.n_gram)

                    # Update progress counters
                    processed_count += 1
                    total_transitions += result["transition_count"]
                    total_states += result["state_count"]

                    # Report progress at intervals
                    current_time = time.time()
                    if (processed_count % batch_size == 0 or processed_count == total_texts) and \
                       (current_time - last_report_time >= report_interval):

                        elapsed = current_time - start_time
                        progress = processed_count / total_texts
                        estimated_remaining = (
                            elapsed / progress) - elapsed if progress > 0 else 0

                        # Get memory usage
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)

                        self.logger.info("Training progress", extra={
                            "metrics": {
                                "processed": processed_count,
                                "total": total_texts,
                                "progress_percent": f"{progress*100:.1f}%",
                                "elapsed_seconds": elapsed,
                                "estimated_remaining_seconds": estimated_remaining,
                                "transitions_collected": total_transitions,
                                "states_collected": total_states,
                                "memory_usage_mb": f"{memory_mb:.1f}"
                            }
                        })

                        last_report_time = current_time

                except Exception as e:
                    self.logger.error("Text processing error", extra={
                        "metrics": {"error": str(e), "error_type": type(e).__name__}
                    })

        # Log training completion
        training_time = time.time() - start_time
        transitions_per_second = total_transitions / \
            training_time if training_time > 0 else 0

        # Update total counts in database
        self.db_adapter.update_total_counts(self.n_gram)

        # Get database statistics
        try:
            db_stats = self.db_adapter.get_model_statistics(self.n_gram)
            if "transitions_count" in db_stats:
                total_transitions = db_stats["transitions_count"]
            if "states_count" in db_stats:
                total_states = db_stats["states_count"]
        except Exception as e:
            self.logger.error(f"Error getting final DB statistics: {e}")

        self.logger.info("Parallel training completed", extra={
            "metrics": {
                "training_time_seconds": training_time,
                "total_texts_processed": total_texts,
                "total_states": total_states,
                "total_transitions": total_transitions,
                "transitions_per_second": transitions_per_second,
                "storage": "database"
            }
        })

        # Stop resource monitoring
        self.resource_monitor.stop()

        return {
            "training_time": training_time,
            "total_texts": total_texts,
            "total_states": total_states,
            "total_transitions": total_transitions,
            "transitions_per_second": transitions_per_second
        }

    def _preprocess_text(self, text):
        """
        Perform comprehensive text normalization using TextPreprocessor.

        This method applies a sequence of normalization steps to improve text quality
        for both training and generation:

        1. Converts to lowercase
        2. Expands contractions
        3. Normalizes for accents/special characters
        4. Normalizes social media text (hashtags, mentions)
        5. Handles emojis
        6. Removes URLs
        7. Removes HTML tags
        8. Handles whitespace

        Args:
            text (str): Raw input text

        Returns:
            str: Normalized text ready for processing
        """
        if self.preprocessor is None:
            self.logger.warning("Text normalization skipped - preprocessor not available", extra={
                "metrics": {"text_sample": text[:50] + ("..." if len(text) > 50 else "")}
            })
            return text
        try:
            self.logger.info("Text normalization started", extra={
                "metrics": {
                    "original_text_sample": text[:50] + ("..." if len(text) > 50 else ""),
                    "text_length": len(text),
                }
            })

            preprocessor = TextPreprocessor()

            # Apply comprehensive normalization pipeline
            text = preprocessor.to_lowercase(text)
            text = preprocessor.handle_contractions(text)
            text = preprocessor.normalize(text)
            text = preprocessor.normalize_social_media_text(text)
            text = preprocessor.handle_emojis(text)
            text = preprocessor.handle_urls(text)
            text = preprocessor.remove_html_tags(text)
            text = preprocessor.handle_whitespace(text)

            # Log normalization result
            self.logger.info("Text normalization completed", extra={
                "metrics": {
                    "normalized_text_sample": text[:50] + ("..." if len(text) > 50 else ""),
                    "text_length": len(text),
                }
            })

            return text

        except Exception as e:
            # Log normalization error
            self.logger.error("Text normalization error", extra={
                "metrics": {"error": str(e), "error_type": type(e).__name__}
            })

            return text

    def predict(self, current_state, preprocess=True, temperature=1.0, top_k=None, top_p=None,
                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                avoid_words=None, generation_context=None, deterministic=False):
        """
        Predicts the next word based on the current state using learned probabilities from database.

        Args:
            current_state: The current state (word or tuple of words) for prediction.
            preprocess (bool): Whether to preprocess the input state
            temperature (float): Controls randomness in word selection. Higher values (e.g., 1.5) increase randomness
                                while lower values (e.g., 0.5) make the model more deterministic.
                                Default: 1.0 (standard probability distribution)
            top_k (int, optional): If set, only sample from the top k most likely next words.
                                   Default: None (use all possible transitions)
            top_p (float, optional): If set, sample from the smallest set of words whose cumulative probability
                                     exceeds top_p. Range: [0.0, 1.0]
                                     Default: None (use all possible transitions)
            repetition_penalty (float): Penalize words that have already appeared. Values > 1.0 reduce
                                        repetition, while values < 1.0 encourage it.
                                        Default: 1.0 (no penalty)
            presence_penalty (float): Penalty for specific words being present at all in the generated text.
                                      Higher values make these words less likely.
                                      Default: 0.0 (no penalty)
            frequency_penalty (float): Penalty for words based on their frequency in the generated text.
                                       Higher values discourage frequent words more strongly.
                                       Default: 0.0 (no penalty)
            avoid_words (list, optional): List of words to avoid in generation
                                          Default: None
            generation_context (dict, optional): Context from the generation process, including generated words
                                                 and their frequencies for penalties
                                                 Default: None
            deterministic (bool): If True, always select the highest probability word
                                  Default: False (use probabilistic selection)

        Returns:
            str or None: The predicted next word, or None if the current state is not in the model.
        """
        # Log prediction attempt with control parameters
        self.logger.info("Prediction attempt", extra={
            "metrics": {
                "current_state": (
                    str(current_state)
                    if not isinstance(current_state, tuple)
                    else " ".join(current_state)
                ),
                "n_gram": self.n_gram,
                "storage": "database",
                "preprocess": preprocess,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "avoid_words_count": len(avoid_words) if avoid_words else 0,
                "deterministic": deterministic
            }
        })

        # Handle string input with preprocessing if requested
        if isinstance(current_state, str):
            if preprocess:
                current_state = self._preprocess_text(current_state)

            # Handle n-gram conversion for string input
            if self.n_gram > 1:
                words = current_state.split()
                if len(words) >= self.n_gram:
                    current_state = tuple(words[-self.n_gram:])
                else:
                    # Log insufficient words
                    self.logger.warning("Insufficient words for prediction", extra={
                        "metrics": {
                            "words_provided": len(words),
                            "words_required": self.n_gram,
                        }
                    })
                    return None

        # Convert tuple to string for database query if needed
        if isinstance(current_state, tuple):
            db_state = " ".join(current_state)
        else:
            db_state = str(current_state)

        try:
            # Get all transition probabilities for this state
            probabilities = self.db_adapter.predict_next_word(
                db_state, self.n_gram)

            # If no probabilities returned, log and return None
            if not probabilities:
                self.logger.warning("No transitions found for state", extra={
                    "metrics": {"state": db_state}
                })
                return None

            # Apply avoid_words filter if provided
            if avoid_words:
                probabilities = {
                    k: v for k, v in probabilities.items() if k not in avoid_words}

            if not probabilities:
                self.logger.warning("No transitions available after filtering")
                return None

            # If deterministic mode is enabled or we're using near-zero temperature,
            # return the highest probability word
            if deterministic or temperature <= 0.01:
                max_prob_word = max(probabilities.items(),
                                    key=lambda x: x[1])[0]
                self.logger.info("Deterministic prediction result", extra={
                    "metrics": {
                        "current_state": db_state,
                        "next_word": max_prob_word
                    }
                })
                return max_prob_word

            # Apply generation context penalties if provided
            if generation_context and isinstance(generation_context, dict):
                generated_words = generation_context.get('generated_words', [])
                word_frequencies = generation_context.get(
                    'word_frequencies', {})

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for word in probabilities:
                        if word in generated_words:
                            probabilities[word] /= repetition_penalty

                # Apply frequency penalty
                if frequency_penalty != 0.0:
                    for word in probabilities:
                        frequency = word_frequencies.get(word, 0)
                        if frequency > 0:
                            probabilities[word] -= frequency_penalty * frequency / \
                                len(generated_words) if generated_words else 0

                # Apply presence penalty
                if presence_penalty != 0.0:
                    for word in probabilities:
                        if word in word_frequencies:
                            probabilities[word] -= presence_penalty

            # Ensure no negative probabilities
            probabilities = {word: max(0.0, prob)
                             for word, prob in probabilities.items()}

            # Check if we have any valid options left
            if not any(probabilities.values()):
                self.logger.warning(
                    "No valid options after applying penalties")
                return None

            # Apply temperature
            if temperature != 1.0:
                # Handle near-zero temperature - pick the most likely option deterministically
                if temperature <= 0.01:
                    max_word = max(probabilities.items(),
                                   key=lambda x: x[1])[0]
                    return max_word

                # For temperature > 0.01: Apply temperature scaling
                probabilities = {
                    word: (prob ** (1.0 / temperature))
                    for word, prob in probabilities.items() if prob > 0
                }

            # Normalize probabilities after transformations
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                probabilities = {
                    word: prob / total_prob for word, prob in probabilities.items()}
            else:
                self.logger.warning(
                    "No valid probabilities after temperature scaling")
                return None

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                sorted_words = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True)
                top_k_words = sorted_words[:top_k]
                probabilities = dict(top_k_words)

                # Re-normalize probabilities
                total_prob = sum(probabilities.values())
                if total_prob > 0:
                    probabilities = {
                        word: prob / total_prob for word, prob in probabilities.items()}

            # Apply nucleus (top-p) sampling
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_words = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True)

                cumulative_prob = 0.0
                nucleus_words = []

                for word, prob in sorted_words:
                    nucleus_words.append((word, prob))
                    cumulative_prob += prob
                    if cumulative_prob >= top_p:
                        break

                probabilities = dict(nucleus_words)

                # Re-normalize probabilities
                total_prob = sum(probabilities.values())
                if total_prob > 0:
                    probabilities = {
                        word: prob / total_prob for word, prob in probabilities.items()}

            # Choose next word based on probabilities
            next_words = list(probabilities.keys())
            prob_values = list(probabilities.values())

            if not next_words:
                self.logger.warning("No words available for selection")
                return None

            try:
                selected_word = random.choices(
                    next_words, weights=prob_values)[0]

                # Log prediction result
                self.logger.info("Prediction result", extra={
                    "metrics": {
                        "current_state": db_state,
                        "next_word": selected_word,
                        "success": True,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty
                    }
                })

                return selected_word
            except ValueError:
                # Fallback if weights are invalid
                self.logger.warning(
                    "Invalid probability distribution, using uniform selection")
                selected_word = random.choice(next_words)

                self.logger.info("Prediction result (fallback)", extra={
                    "metrics": {
                        "current_state": db_state,
                        "next_word": selected_word,
                        "success": True,
                        "method": "uniform_fallback"
                    }
                })

                return selected_word

        except Exception as e:
            self.logger.error(f"Error predicting next word: {e}", extra={
                "metrics": {"error": str(e)}
            })
            return None

    def generate_text(self, start=None, max_length=100, temperature=1.0, top_k=None, top_p=None,
                      repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, avoid_words=None):
        """
        Generates text starting from a given state with control parameters.
        Uses database storage for transition probabilities.

        Args:
            start: Starting state (word or tuple). If None, a random state is chosen.
            max_length (int): Maximum number of words to generate.
            temperature (float): Controls randomness. Higher values (e.g., 1.5) increase randomness,
                                lower values (e.g., 0.5) make generation more deterministic.
            top_k (int, optional): If set, only sample from top k most probable words.
            top_p (float, optional): If set, sample from smallest set of words whose cumulative 
                                    probability exceeds top_p (nucleus sampling).
            repetition_penalty (float): Penalty for repeated words. Values > 1.0 discourage repetition.
            presence_penalty (float): Penalty for specific words appearing at all.
            frequency_penalty (float): Penalty based on word frequency in generated text.
            avoid_words (list, optional): List of words to avoid in generation.

        Returns:
            str: Generated text.
        """
        # Log generation start with control parameters
        self.logger.info("Text generation started", extra={
            "metrics": {
                "start": str(start) if start else "random",
                "max_length": max_length,
                "n_gram": self.n_gram,
                "storage": "database",
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "avoid_words_count": len(avoid_words) if avoid_words else 0
            }
        })

        # Check if the model has any transitions
        if not self.db_adapter.has_transitions(self.n_gram):
            self.logger.warning("Text generation failed - model not trained", extra={
                "metrics": {"storage": "database"}
            })
            return "Model not trained"

        # Always preprocess the starting state if it's a string
        if start and isinstance(start, str):
            start = self._preprocess_text(start)

        # Get a valid starting state
        current_state = self._get_valid_start_state(start)
        if current_state is None:
            # Log no valid starting state
            self.logger.warning("Text generation failed - no valid starting state", extra={
                "metrics": {"requested_start": str(start) if start else "random"}
            })
            return "Could not find valid starting state"

        # Initialize text generation
        text = []

        # Convert tuple to list of words for output
        if isinstance(current_state, tuple):
            text.extend(list(current_state))
        else:
            text.append(current_state)

        # Calculate how many more words to generate
        remaining = max_length - (
            self.n_gram if isinstance(current_state, tuple) else 1
        )

        # Initialize generation context to track generated words for penalties
        generation_context = {
            'generated_words': text.copy(),  # Start with the initial words
            'word_frequencies': {word: 1 for word in text}
        }

        # Generate remaining words
        for i in range(remaining):
            next_word = self.predict(
                current_state,
                preprocess=False,  # Already preprocessed
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                avoid_words=avoid_words,
                generation_context=generation_context
            )

            if next_word is None:
                # Log early termination
                self.logger.info("Text generation ended early - no prediction available", extra={
                    "metrics": {
                        "words_generated": len(text),
                        "last_state": (
                            str(current_state)
                            if not isinstance(current_state, tuple)
                            else " ".join(current_state)
                        ),
                    }
                })
                break

            text.append(next_word)

            # Update generation context
            generation_context['generated_words'].append(next_word)
            generation_context['word_frequencies'][next_word] = generation_context['word_frequencies'].get(
                next_word, 0) + 1

            # Update current state for n-grams
            if self.n_gram > 1:
                if isinstance(current_state, tuple):
                    current_state = tuple(
                        list(current_state)[1:] + [next_word])
                else:
                    # Handle case where current_state might be a string
                    words = (
                        current_state.split()
                        if isinstance(current_state, str)
                        else [current_state]
                    )
                    words = words + [next_word]
                    current_state = tuple(words[-self.n_gram:])
            else:
                current_state = next_word

            # Periodically log progress for long sequences
            if i > 0 and i % 50 == 0:
                self.logger.info("Text generation progress", extra={
                    "metrics": {
                        "words_generated": len(text),
                        "percentage_complete": f"{i/remaining*100:.1f}%",
                    }
                })

        generated_text = " ".join(text)

        # Log generation completion
        self.logger.info("Text generation completed", extra={
            "metrics": {
                "text": generated_text[:500] + ("..." if len(generated_text) > 500 else ""),
                "n_gram": self.n_gram,
                "start": str(start) if start else "random",
                "words_generated": len(text),
                "storage": "database",
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "unique_words_ratio": len(set(text)) / len(text) if text else 0
            }
        })

        return generated_text

    def _get_valid_start_state(self, start):
        """
        Get a valid starting state from database storage, either the provided one or a random one.

        Args:
            start: The requested starting state or None for random

        Returns:
            A valid starting state from the database or None if unavailable
        """
        # If start is None or empty string, choose randomly
        if start is None:
            return self.db_adapter.get_random_state(self.n_gram)

        # Explicitly handle empty strings - they should not generate text
        if isinstance(start, str) and start.strip() == "":
            self.logger.warning(
                "Empty string provided as start state - rejected")
            return None

        # Process the provided start state
        if isinstance(start, str) and self.n_gram > 1:
            words = start.split()
            if len(words) >= self.n_gram:
                start = tuple(words[: self.n_gram])
            else:
                # Not enough words, get a random state
                self.logger.warning("Not enough words in start state, using random state", extra={
                    "metrics": {"words_provided": len(words), "words_required": self.n_gram}
                })
                return self.db_adapter.get_random_state(self.n_gram)

        # Validate that start exists in model
        if isinstance(start, tuple):
            db_state = " ".join(start)
        else:
            db_state = str(start)

        if not self.db_adapter.check_state_exists(db_state, self.n_gram):
            self.logger.warning("Requested start state not found in model, using random state", extra={
                "metrics": {"requested_state": db_state}
            })
            return self.db_adapter.get_random_state(self.n_gram)

        # Return the original tuple or string state
        return start

    def __del__(self):
        """Cleanup method to close database connections"""
        if self.db_adapter:
            try:
                self.db_adapter.close_connections()
            except:
                pass

    def save_model(self, filepath):
        """
        Save the model to ONNX format.
        This is a standardized wrapper for export_to_onnx.

        Args:
            filepath (str): Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        return self.export_to_onnx(filepath)

    @classmethod
    def load_model(cls, filepath, db_config=None, environment="development", memory_threshold=10000, logger=None):
        """
        Load a Markov Chain model from an ONNX file.

        Args:
            filepath (str): Path to the ONNX model file
            db_config (dict, optional): PostgreSQL configuration if using database storage
            environment (str): Which environment to use ('development' or 'test')
            memory_threshold (int): Threshold for in-memory vs DB storage
            logger (logging.Logger, optional): Logger instance to use. If None, a new one will be created.

        Returns:
            MarkovChain: A new MarkovChain instance with the loaded model data
        """
        # Ensure we have a logger
        if logger is None:
            logger = get_logger("markov_chain_loader")

        try:
            logger.info("ONNX model loading started", extra={
                "metrics": {
                    "filepath": filepath,
                    "environment": environment
                }
            })

            # Load the ONNX model
            onnx_model = onnx.load(filepath)

            # Extract metadata
            metadata = {
                prop.key: prop.value for prop in onnx_model.metadata_props}

            # Get n-gram value from metadata
            n_gram = int(metadata.get('n_gram', '1'))

            # Get vocabulary mapping
            vocab_mapping = json.loads(metadata.get('vocab_mapping', '{}'))
            idx_to_word = {int(idx): word for idx,
                           word in vocab_mapping.items()}

            # Create a new model instance
            model = cls(
                n_gram=n_gram,
                db_config=db_config,
                environment=environment,
                logger=logger
            )

            # Create an ONNX Runtime session to access the model
            session_options = ort.SessionOptions()
            session = ort.InferenceSession(
                filepath, sess_options=session_options)

            # Get model weights (transition matrix) - depends on ONNX model structure
            # This assumes the exported model has weights stored in a node named 'transition_matrix'
            for node in onnx_model.graph.node:
                if (node.op_type == 'Constant' and len(node.output) > 0 and
                        node.output[0] == 'matrix'):
                    # Extract the weights tensor
                    weights_tensor = None
                    for attr in node.attribute:
                        if attr.name == 'value':
                            weights_tensor = onnx.numpy_helper.to_array(attr.t)
                            break

                    if weights_tensor is not None:
                        # Convert matrix to dictionary format
                        transitions = defaultdict(lambda: defaultdict(int))
                        total_counts = defaultdict(int)

                        # Get vocabulary size
                        vocab_size = weights_tensor.shape[0]

                        # Populate transitions and total_counts
                        for i in vocab_size:
                            if i not in idx_to_word:
                                continue

                            state = idx_to_word[i]
                            state_vector = weights_tensor[i]

                            # Count total transitions for this state
                            non_zero_transitions = np.where(
                                state_vector > 0)[0]
                            if len(non_zero_transitions) == 0:
                                continue

                            # Calculate total based on probabilities
                            # For simplicity, we'll normalize probabilities to counts
                            # by using a base count of 100 for the most likely transition
                            max_prob = np.max(state_vector)
                            scaling_factor = 100 / max_prob if max_prob > 0 else 0

                            for j in non_zero_transitions:
                                if j not in idx_to_word:
                                    continue

                                next_word = idx_to_word[j]
                                probability = state_vector[j]

                                # Convert probability to count
                                count = int(probability * scaling_factor)
                                if count > 0:
                                    transitions[state][next_word] = count
                                    total_counts[state] += count

                        # Update model with the extracted transitions
                        model.transitions = transitions
                        model.total_counts = total_counts
                        model.using_db = False

                        # Log successful loading
                        logger.info("ONNX model loaded successfully", extra={
                            "metrics": {
                                "filepath": filepath,
                                "states_count": len(transitions),
                                "n_gram": n_gram,
                                "environment": environment,
                            }
                        })

                        # Option to store in database if needed
                        if db_config and len(transitions) > model.memory_threshold:
                            logger.info(
                                "Converting loaded model to database storage")
                            model._store_model_in_db(transitions, total_counts)

                        return model

            # If we get here, we couldn't extract the transition matrix
            logger.error("ONNX model loading failed", extra={
                "metrics": {
                    "filepath": filepath,
                    "error": "Failed to extract transition matrix",
                }
            })

            return None

        except Exception as e:
            logger.error("ONNX model loading failed", extra={
                "metrics": {
                    "filepath": filepath,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            })
            return None

    def export_to_onnx(self, filepath):
        """
        Export the Markov Chain model to ONNX format.

        This method transforms the Markov Chain transition matrix into an ONNX model
        where transition probabilities are represented as weights in a neural network structure.

        Args:
            filepath (str): Path where the ONNX model will be saved

        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            # Log start of export process
            self.logger.info("ONNX export started", extra={
                "metrics": {
                    "filepath": filepath,
                    "storage": "database" if self.using_db else "memory",
                    "n_gram": self.n_gram,
                }
            })

            # Create vocabulary and index mapping
            vocab = set()
            transition_matrix = {}

            # Extract vocabulary and transitions based on storage type
            if self.using_db:
                vocab, transition_matrix = self._extract_db_vocabulary_and_transitions()
            else:
                # Extract from memory
                for state in self.transitions:
                    if isinstance(state, tuple):
                        for word in state:
                            vocab.add(word)
                    else:
                        vocab.add(state)

                    for next_word in self.transitions[state]:
                        vocab.add(next_word)

                transition_matrix = self.transitions

            # Create word to index mapping
            word_to_idx = {word: i for i, word in enumerate(sorted(vocab))}
            idx_to_word = {i: word for word, i in word_to_idx.items()}
            vocab_size = len(vocab)

            # Create transition probability matrix as numpy array
            matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

            # Fill the transition matrix
            if self.using_db:
                # For database-backed model
                for (state, next_word), count in transition_matrix.items():
                    if self.n_gram == 1:
                        # Single word state
                        state_idx = word_to_idx[state]
                    else:
                        # Multi-word state, use last word as state (simplification)
                        state_parts = state.split()
                        state_idx = word_to_idx[state_parts[-1]]

                    next_idx = word_to_idx[next_word]
                    total = self._get_db_total_for_state(state)
                    if total:
                        matrix[state_idx, next_idx] = count / total
            else:
                # For in-memory model
                for state, next_words in transition_matrix.items():
                    if isinstance(state, tuple):
                        # Multi-word state, use last word as state (simplification)
                        state_idx = word_to_idx[state[-1]]
                    else:
                        state_idx = word_to_idx[state]

                    total = self.total_counts[state]
                    for next_word, count in next_words.items():
                        next_idx = word_to_idx[next_word]
                        matrix[state_idx, next_idx] = count / total

            # Create ONNX model components
            input_name = 'state_idx'
            output_name = 'next_word_probabilities'

            # Define the ONNX graph
            # Input: state index
            input_tensor = helper.make_tensor_value_info(
                input_name, TensorProto.INT64, [None, 1]  # Batch dimension
            )

            # Output: probability distribution over next words
            output_tensor = helper.make_tensor_value_info(
                output_name, TensorProto.FLOAT, [
                    None, vocab_size]  # Batch dimension, vocab size
            )

            # Matrix weights
            weights = helper.make_tensor(
                'transition_matrix', TensorProto.FLOAT, matrix.shape, matrix.flatten().tolist()
            )

            # Create weight node (convert to constant)
            weights_node = helper.make_node(
                'Constant', [], ['matrix'], value=weights
            )

            # Create gather node to look up the right row of the transition matrix
            gather_node = helper.make_node(
                'Gather', ['matrix', input_name], ['gathered'], axis=0
            )

            # Create identity node for output
            identity_node = helper.make_node(
                'Identity', ['gathered'], [output_name]
            )

            # Define the graph
            graph = helper.make_graph(
                [weights_node, gather_node, identity_node],
                'markov_chain_model',
                [input_tensor],
                [output_tensor],
            )

            # Add vocabulary mapping as model metadata
            model_metadata = {
                'n_gram': str(self.n_gram),
                'vocab_size': str(vocab_size),
                'vocab_mapping': json.dumps(word_to_idx),
                'export_date': datetime.datetime.now().isoformat(),
                'model_type': 'markov_chain'
            }

            # Create the model
            model = helper.make_model(
                graph,
                producer_name='Samhail',
                doc_string='Markov Chain model converted to ONNX format'
            )

            # Add metadata
            for key, value in model_metadata.items():
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = value

            # Set opset version
            opset = model.opset_import.add()
            opset.version = 14

            # Check model validity
            onnx.checker.check_model(model)

            # Save the model
            onnx.save(model, filepath)

            # Log successful export
            self.logger.info("ONNX export completed", extra={
                "metrics": {
                    "filepath": filepath,
                    "vocab_size": vocab_size,
                    "n_gram": self.n_gram,
                }
            })

            return True

        except Exception as e:
            # Log export error
            self.logger.error("ONNX export failed", extra={
                "metrics": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "filepath": filepath,
                }
            })

            return False

    def _extract_db_vocabulary_and_transitions(self):
        """Extract vocabulary and transitions from database"""
        if not self.db_adapter:
            self.logger.error(
                "Failed to get database adapter for ONNX export")
            return set(), {}

        try:
            vocab = set()
            transitions = {}

            # Use the adapter to get a connection
            conn = self.db_adapter.get_connection()
            if not conn:
                self.logger.error(
                    "Failed to get database connection for vocabulary extraction")
                return set(), {}

            try:
                # Add environment suffix to table names
                table_prefix = f"markov_{self.environment}"

                with conn.cursor() as cur:
                    # Get all transitions
                    cur.execute(
                        f"""
                        SELECT state, next_word, count 
                        FROM {table_prefix}_transitions
                        WHERE n_gram = %s
                    """,
                        (self.n_gram,),
                    )

                    for state, next_word, count in cur.fetchall():
                        # Add words to vocabulary
                        if self.n_gram == 1:
                            vocab.add(state)
                        else:
                            for word in state.split():
                                vocab.add(word)

                        vocab.add(next_word)

                        # Store transition probability
                        transitions[(state, next_word)] = count

                return vocab, transitions
            finally:
                # Return the connection when done
                self.db_adapter.return_connection(conn)
        except Exception as e:
            self.logger.error(
                f"Database error during vocabulary extraction: {e}")
            return set(), {}

    def _get_db_total_for_state(self, state):
        """Get total count for a state from database"""
        if not self.db_adapter:
            return 0

        # Use the adapter to get a connection
        conn = self.db_adapter.get_connection()
        if not conn:
            return 0

        try:
            # Add environment suffix to table names
            table_prefix = f"markov_{self.environment}"

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT count FROM {table_prefix}_total_counts
                    WHERE state = %s AND n_gram = %s
                """,
                    (state, self.n_gram,),
                )

                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Database error getting total count: {e}")
            return 0
        finally:
            # Return the connection when done
            self.db_adapter.return_connection(conn)

    def _store_model_in_db(self, transitions, total_counts):
        """
        Store the loaded model in database storage

        Args:
            transitions: Dictionary of transitions to store
            total_counts: Dictionary of total counts to store
        """
        if not self.db_adapter:
            self.logger.warning(
                "No database adapter available, keeping model in memory")
            return False

        # Use the adapter to get a connection
        conn = self.db_adapter.get_connection()
        if not conn:
            self.logger.warning(
                "Failed to get database connection, keeping model in memory")
            return False

        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"

        try:
            # First clear existing data for this n-gram
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_prefix}_transitions WHERE n_gram = %s
                """,
                    (self.n_gram,),
                )
                cur.execute(
                    f"""
                    DELETE FROM {table_prefix}_total_counts WHERE n_gram = %s
                """,
                    (self.n_gram,),
                )

            # Prepare data for batch insertion
            transitions_batch = []
            total_counts_data = []

            # Process transitions
            for state, next_words in transitions.items():
                # Handle tuple states for n-gram > 1
                if isinstance(state, tuple):
                    db_state = " ".join(state)
                else:
                    db_state = str(state)

                # Add total count
                total_counts_data.append(
                    (db_state, total_counts[state], self.n_gram))

                # Add transitions
                for next_word, count in next_words.items():
                    transitions_batch.append(
                        (db_state, next_word, count, self.n_gram))

            # Insert transitions in batches
            batch_size = 1000
            for i in range(0, len(transitions_batch), batch_size):
                batch = transitions_batch[i:i+batch_size]

                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {table_prefix}_transitions (state, next_word, count, n_gram)
                        VALUES %s
                    """,
                        batch
                    )

            # Insert total counts in batches
            for i in range(0, len(total_counts_data), batch_size):
                batch = total_counts_data[i:i+batch_size]

                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {table_prefix}_total_counts (state, count, n_gram)
                        VALUES %s
                    """,
                        batch
                    )

            conn.commit()

            # Update model state
            self.using_db = True
            self.transitions.clear()
            self.total_counts.clear()

            self.logger.info(
                f"Successfully stored model in {self.environment} database")
            return True

        except Exception as e:
            self.logger.error(f"Error storing model in database: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            # Return the connection when done
            self.db_adapter.return_connection(conn)

    def __getstate__(self):
        """
        Prepare object for pickle serialization by converting non-picklable attributes
        to serializable formats.

        Returns:
            dict: State dictionary for pickle
        """
        state = self.__dict__.copy()

        # Handle the database adapter which contains non-picklable connections
        if 'db_adapter' in state:
            # Store only the connection parameters, not the connection itself
            if state['db_adapter'] is not None:
                state['_db_config'] = getattr(
                    state['db_adapter'], 'db_config', None)
                state['_db_environment'] = getattr(
                    state['db_adapter'], 'environment', None)

            # Remove the adapter instance which contains the non-picklable connection
            del state['db_adapter']

        # Handle resource monitor which may have non-picklable elements
        if 'resource_monitor' in state:
            # No need to store this as it will be recreated
            del state['resource_monitor']

        # Convert defaultdict with lambda to regular dictionaries for serialization
        if 'transitions' in state:
            # Convert nested defaultdict to regular nested dictionaries
            regular_dict = {}
            for state_key, next_words in state['transitions'].items():
                regular_dict[state_key] = dict(next_words)
            state['transitions'] = regular_dict

        if 'total_counts' in state:
            # Convert defaultdict to regular dictionary
            state['total_counts'] = dict(state['total_counts'])

        return state

    def __setstate__(self, state):
        """
        Restore object after unpickling by recreating necessary attributes
        including defaultdict structures.

        Args:
            state (dict): State dictionary from pickle
        """
        # Extract database config from the state
        db_config = state.pop(
            '_db_config', None) if '_db_config' in state else None
        db_environment = state.pop(
            '_db_environment', None) if '_db_environment' in state else 'development'

        # Convert regular dictionaries back to defaultdict with appropriate factory functions
        if 'transitions' in state:
            regular_dict = state['transitions']
            nested_dict = defaultdict(lambda: defaultdict(int))

            # Populate the nested defaultdict
            for state_key, next_words in regular_dict.items():
                for next_word, count in next_words.items():
                    nested_dict[state_key][next_word] = count

            state['transitions'] = nested_dict

        if 'total_counts' in state:
            regular_dict = state['total_counts']
            counts_dict = defaultdict(int)

            # Populate the defaultdict
            for key, value in regular_dict.items():
                counts_dict[key] = value

            state['total_counts'] = counts_dict

        # Restore the state
        self.__dict__.update(state)

        # Recreate resource monitor
        self.resource_monitor = ResourceMonitor(
            logger=self.logger,
            monitoring_interval=10.0
        )

        # Recreate database adapter if needed
        if getattr(self, 'using_db', False) and db_config:
            try:
                from utils.database_adapters.postgresql.markov_chain import MarkovChainPostgreSqlAdapter
                self.db_adapter = MarkovChainPostgreSqlAdapter(
                    environment=db_environment or self.environment,
                    logger=self.logger,
                    db_config=db_config
                )

                self.logger.info("Database adapter recreated after unpickling", extra={
                    "metrics": {
                        "environment": self.environment
                    }
                })
            except Exception as e:
                self.db_adapter = None
                self.using_db = False
                self.logger.warning(f"Failed to recreate database adapter after unpickling: {e}", extra={
                    "metrics": {
                        "error": str(e),
                        "fallback": "in-memory only"
                    }
                })
