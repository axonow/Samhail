import sys
import math
import random
import time
import uuid
import os
import threading
from datetime import datetime

# Add project root to Python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.system_monitoring import ResourceMonitor
from utils.database_adapters.postgresql.markov_chain import MarkovChainPostgreSqlAdapter
from utils.loggers.json_logger import get_logger

class MarkovChainAnalytics:
    """
    Analytics module for Markov Chain models with JSON-formatted metrics.

    This class provides various methods for analyzing Markov Chain models,
    including model statistics, sequence scoring, and transition probability
    analysis. All metrics are logged in JSON format for compatibility with
    monitoring tools like Grafana.
    """

    def __init__(self, markov_chain, logger=None):
        """
        Initialize with a reference to a MarkovChain instance.

        Args:
            markov_chain: An instance of the MarkovChain class
            logger: A logger instance for logging analytics activities
        """
        # Set up the logger if not provided
        if logger is None:
            # Create log directory if it doesn't exist
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            # Use specific log file
            log_file = os.path.join(log_dir, "analytics.log")
            self.logger = get_logger("markov_analytics", log_file=log_file)
        else:
            self.logger = logger

        self.markov_chain = markov_chain

        # Generate unique identifier for this analytics instance
        self.analytics_id = str(uuid.uuid4())[:8]

        # Access the database adapter from the markov chain if available
        self.db_adapter = getattr(self.markov_chain, 'db_adapter', None)

        # Initialize resource monitor for system metrics tracking
        self.resource_monitor = ResourceMonitor(
            logger=self.logger,
            monitoring_interval=10.0  # Log system metrics every 10 seconds
        )

        # Initialize system metrics
        self.system_metrics = self.resource_monitor.get_resource_usage()

        # Log initialization with system metrics
        self.logger.info("MarkovChainAnalytics initialized", extra={
            "metrics": {
                "model_type": "markov_chain",
                "n_gram": self.markov_chain.n_gram,
                "storage_type": ("database" if self.markov_chain.using_db else "memory"),
                "environment": self.markov_chain.environment,
                "analytics_id": self.analytics_id,
                "system": self.system_metrics
            }
        })

    def get_system_metrics(self):
        """
        Get current system resource metrics including memory usage, CPU usage, and thread count.

        Returns:
            dict: Dictionary containing system resource metrics
        """
        return self.resource_monitor.get_resource_usage()

    def log_operation_with_metrics(self, message, operation_name, metrics=None, log_level="info"):
        """
        Log an operation with system metrics and custom metrics.

        Args:
            message (str): Message to log
            operation_name (str): Name of the operation being performed
            metrics (dict, optional): Custom metrics to include
            log_level (str): Logging level (info, debug, warning, error)
        """
        system_metrics = self.get_system_metrics()

        combined_metrics = {
            "system": system_metrics,
            "model_id": self.analytics_id,
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name
        }

        # Add custom metrics if provided
        if metrics:
            combined_metrics.update(metrics)

        # Log with appropriate level
        if log_level == "debug":
            self.logger.debug(message, extra={"metrics": combined_metrics})
        elif log_level == "warning":
            self.logger.warning(message, extra={"metrics": combined_metrics})
        elif log_level == "error":
            self.logger.error(message, extra={"metrics": combined_metrics})
        else:
            self.logger.info(message, extra={"metrics": combined_metrics})

    def analyze_model(self):
        """
        Analyze model characteristics and provide comprehensive statistics.

        Returns:
            dict: A dictionary containing various model statistics including:
                - storage_type: 'database' or 'memory'
                - environment: The model's environment setting
                - n_gram: The n-gram size
                - transitions_count: Total number of transitions
                - states_count: Number of unique states
                - top_transitions: Top 5 most common transitions
                - vocabulary_size: Number of unique words in the model
                - perplexity: An estimate of the model's perplexity (if applicable)
                - system_metrics: Memory, CPU, and thread usage
        """
        start_time = time.time()

        # Start resource monitoring for this operation
        self.resource_monitor.start()

        # Log operation start with initial system metrics
        self.log_operation_with_metrics(
            "Starting model analysis",
            "analyze_model_start",
            {"analysis_stage": "beginning"}
        )

        stats = {
            "storage_type": "database" if self.markov_chain.using_db else "memory",
            "environment": self.markov_chain.environment,
            "n_gram": self.markov_chain.n_gram,
        }

        if self.markov_chain.using_db and self.db_adapter:
            # Use the database adapter to get model statistics
            db_stats = self.db_adapter.get_model_statistics(
                self.markov_chain.n_gram)
            stats.update(db_stats)

            # Log intermediate metrics during database analysis
            self.log_operation_with_metrics(
                "Database statistics retrieved",
                "analyze_model_progress",
                {
                    "analysis_stage": "db_stats_retrieved",
                    "db_stats_count": len(db_stats)
                },
                "debug"
            )
        else:
            # In-memory analytics
            stats["transitions_count"] = sum(
                len(next_words) for next_words in self.markov_chain.transitions.values()
            )
            stats["states_count"] = len(self.markov_chain.transitions)

            # Get top 5 most common state transitions
            all_transitions = []
            for state, next_words in self.markov_chain.transitions.items():
                for next_word, count in next_words.items():
                    all_transitions.append((state, next_word, count))

            top_transitions = sorted(all_transitions, key=lambda x: x[2], reverse=True)[
                :5
            ]
            stats["top_transitions"] = [
                {"state": str(state), "next_word": next_word, "count": count}
                for state, next_word, count in top_transitions
            ]

            # Calculate vocabulary size
            unique_words = set()
            for next_words in self.markov_chain.transitions.values():
                unique_words.update(next_words.keys())
            stats["vocabulary_size"] = len(unique_words)

            # Average transitions per state
            total_transitions = sum(
                len(next_words) for next_words in self.markov_chain.transitions.values()
            )
            if stats["states_count"] > 0:
                stats["avg_transitions_per_state"] = (
                    total_transitions / stats["states_count"]
                )
            else:
                stats["avg_transitions_per_state"] = 0

            # Get memory usage estimate
            import sys

            memory_size = sum(
                sys.getsizeof(state) + sys.getsizeof(next_words)
                for state, next_words in self.markov_chain.transitions.items()
            )
            stats["memory_usage_bytes"] = memory_size

            # Get state distribution metrics
            if stats["states_count"] > 0:
                counts = list(self.markov_chain.total_counts.values())
                counts.sort()
                stats["state_count_distribution"] = {
                    "min": min(counts),
                    "max": max(counts),
                    "avg": sum(counts) / len(counts),
                    "median": counts[len(counts) // 2],
                }
            else:
                stats["state_count_distribution"] = {
                    "min": 0,
                    "max": 0,
                    "avg": 0,
                    "median": 0,
                }

            # Log memory analytics progress with system metrics
            self.log_operation_with_metrics(
                "In-memory statistics calculated",
                "analyze_model_progress",
                {
                    "analysis_stage": "memory_stats_calculated",
                    "transitions_count": stats["transitions_count"],
                    "states_count": stats["states_count"]
                },
                "debug"
            )

        # Calculate execution time
        execution_time = time.time() - start_time
        stats["analysis_execution_time"] = execution_time

        # Add final system metrics to stats
        final_system_metrics = self.resource_monitor.get_resource_usage()
        stats["system_metrics"] = final_system_metrics

        # Stop resource monitoring
        self.resource_monitor.stop()

        # Log results with final system metrics
        self.log_operation_with_metrics(
            "Model analysis completed",
            "analyze_model_complete",
            {
                "model_metrics": stats,
                "execution_time": execution_time,
            }
        )

        return stats

    def get_transition_probability(self, current_state, next_word):
        """
        Calculate the probability of transitioning from current_state to next_word.

        Args:
            current_state: The current state (word or tuple)
            next_word: The next word to transition to

        Returns:
            float: Probability of the transition (between 0 and 1)
        """
        start_time = time.time()

        if self.markov_chain.using_db and self.db_adapter:
            prob = self.db_adapter.get_transition_probability(
                current_state, next_word, self.markov_chain.n_gram)
        else:
            prob = self._get_memory_transition_probability(
                current_state, next_word)

        # Log metrics with system resources
        execution_time = time.time() - start_time

        self.log_operation_with_metrics(
            f"Transition probability: {current_state} → {next_word} = {prob}",
            "get_transition_probability",
            {
                "current_state": str(current_state),
                "next_word": next_word,
                "probability": prob,
                "execution_time": execution_time,
            },
            "debug"
        )

        return prob

    def _get_memory_transition_probability(self, current_state, next_word):
        """Calculate transition probability using in-memory storage"""
        if current_state not in self.markov_chain.transitions:
            return 0.0

        next_words = self.markov_chain.transitions[current_state]
        total = self.markov_chain.total_counts[current_state]

        # Return probability if transition exists, otherwise 0
        return next_words[next_word] / total if next_word in next_words else 0.0

    def score_sequence(self, sequence, preprocess=True):
        """
        Calculate log probability score for a given sequence

        Args:
            sequence (str): Text sequence to score
            preprocess (bool): Whether to preprocess the sequence

        Returns:
            float: Log probability score (higher is better fit)
        """
        start_time = time.time()
        sequence_id = str(uuid.uuid4())[:8]

        # Log start of sequence scoring with initial system metrics
        self.log_operation_with_metrics(
            "Starting sequence scoring",
            "score_sequence_start",
            {
                "sequence": sequence[:50] + ("..." if len(sequence) > 50 else ""),
                "sequence_length": len(sequence),
                "word_count": len(sequence.split()),
                "preprocess": preprocess,
                "sequence_id": sequence_id
            },
            "debug"
        )

        if preprocess and hasattr(self.markov_chain, "_preprocess_text"):
            preprocess_start = time.time()
            sequence = self.markov_chain._preprocess_text(sequence)
            preprocess_time = time.time() - preprocess_start

            # Log preprocessing completion with metrics
            self.log_operation_with_metrics(
                "Sequence preprocessing complete",
                "score_sequence_preprocess",
                {
                    "sequence_id": sequence_id,
                    "preprocess_time": preprocess_time,
                    "processed_length": len(sequence),
                    "processed_word_count": len(sequence.split())
                },
                "debug"
            )
        else:
            preprocess_time = 0

        words = sequence.split()
        if len(words) <= self.markov_chain.n_gram:
            # Log early return with system metrics
            self.log_operation_with_metrics(
                "Sequence too short for scoring",
                "score_sequence_error",
                {
                    "score": 0.0,
                    "reason": "sequence_too_short",
                    "execution_time": time.time() - start_time,
                    "sequence_id": sequence_id
                }
            )

            return 0.0

        log_prob = 0.0
        count = 0
        zero_prob_transitions = 0

        # Track transition stats
        transitions = []

        # Calculate progress thresholds for logging
        words_to_process = len(words) - self.markov_chain.n_gram
        # Log 4 times during processing
        progress_threshold = max(1, words_to_process // 4)

        for i in range(words_to_process):
            if self.markov_chain.n_gram == 1:
                current_state = words[i]
            else:
                current_state = tuple(words[i: i + self.markov_chain.n_gram])

            next_word = words[i + self.markov_chain.n_gram]
            transition_prob = self.get_transition_probability(
                current_state, next_word)

            # Track this transition
            transitions.append(
                {
                    "state": str(current_state),
                    "next_word": next_word,
                    "probability": transition_prob,
                }
            )

            if transition_prob > 0:
                log_prob += math.log(transition_prob)
                count += 1
            else:
                zero_prob_transitions += 1

            # Log progress for long sequences with system metrics
            if i > 0 and i % progress_threshold == 0:
                progress_percent = (i / words_to_process) * 100
                self.log_operation_with_metrics(
                    f"Sequence scoring progress: {progress_percent:.1f}%",
                    "score_sequence_progress",
                    {
                        "sequence_id": sequence_id,
                        "progress_percent": progress_percent,
                        "transitions_processed": i,
                        "total_transitions": words_to_process,
                        "current_log_prob": log_prob,
                        "zero_transitions_so_far": zero_prob_transitions
                    },
                    "debug"
                )

        # Normalize by sequence length for fair comparison between different sequences
        final_score = log_prob / max(1, count) if count > 0 else float("-inf")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log results with final system metrics
        self.log_operation_with_metrics(
            f"Sequence scored with result {final_score:.4f}",
            "score_sequence_complete",
            {
                "sequence_id": sequence_id,
                "score": final_score,
                "transitions_checked": count + zero_prob_transitions,
                "zero_probability_transitions": zero_prob_transitions,
                "execution_time": execution_time,
                "preprocess_time": preprocess_time,
                "transitions_sample": transitions[:5] if transitions else [],
            }
        )

        return final_score

    def perplexity(self, text, preprocess=True):
        """
        Calculate the perplexity of the model on a given text.

        Perplexity is a measure of how well a model predicts a sample.
        Lower perplexity indicates better prediction performance.

        Args:
            text (str): The text to evaluate
            preprocess (bool): Whether to preprocess the text

        Returns:
            float: The perplexity score (lower is better)
        """
        start_time = time.time()
        text_id = str(uuid.uuid4())[:8]

        # Log start of perplexity calculation with initial system metrics
        self.log_operation_with_metrics(
            "Starting perplexity calculation",
            "perplexity_start",
            {
                "text_sample": text[:50] + ("..." if len(text) > 50 else ""),
                "text_length": len(text),
                "word_count": len(text.split()),
                "preprocess": preprocess,
                "text_id": text_id
            }
        )

        if preprocess and hasattr(self.markov_chain, "_preprocess_text"):
            preprocess_start = time.time()
            text = self.markov_chain._preprocess_text(text)
            preprocess_time = time.time() - preprocess_start

            # Log preprocessing metrics
            self.log_operation_with_metrics(
                "Text preprocessing complete",
                "perplexity_preprocess",
                {
                    "text_id": text_id,
                    "preprocess_time": preprocess_time,
                    "processed_text_length": len(text),
                    "processed_word_count": len(text.split())
                },
                "debug"
            )
        else:
            preprocess_time = 0

        words = text.split()
        if len(words) <= self.markov_chain.n_gram:
            # Log early return for text too short with system metrics
            self.log_operation_with_metrics(
                "Text too short for perplexity calculation",
                "perplexity_error",
                {
                    "perplexity": float("inf"),
                    "reason": "text_too_short",
                    "execution_time": time.time() - start_time,
                    "text_id": text_id
                }
            )

            return float("inf")

        log_prob = 0.0
        token_count = 0
        zero_prob_count = 0

        # Calculate progress thresholds for logging
        words_to_process = len(words) - self.markov_chain.n_gram
        # Log 5 times during processing
        progress_threshold = max(1, words_to_process // 5)

        for i in range(words_to_process):
            if self.markov_chain.n_gram == 1:
                current_state = words[i]
            else:
                current_state = tuple(words[i: i + self.markov_chain.n_gram])

            next_word = words[i + self.markov_chain.n_gram]
            transition_prob = self.get_transition_probability(
                current_state, next_word)

            if transition_prob <= 0:
                zero_prob_count += 1

            # Smooth probability to avoid log(0)
            smooth_prob = max(transition_prob, 1e-10)
            log_prob += math.log2(smooth_prob)
            token_count += 1

            # Log progress for long texts with system metrics
            if i > 0 and i % progress_threshold == 0:
                progress_percent = (i / words_to_process) * 100
                current_perplexity = 2 ** (-log_prob /
                                           token_count) if token_count > 0 else float("inf")

                self.log_operation_with_metrics(
                    f"Perplexity calculation progress: {progress_percent:.1f}%",
                    "perplexity_progress",
                    {
                        "text_id": text_id,
                        "progress_percent": progress_percent,
                        "tokens_processed": i,
                        "total_tokens": words_to_process,
                        "current_perplexity": current_perplexity,
                        "zero_prob_tokens_so_far": zero_prob_count
                    },
                    "debug"
                )

        # Calculate perplexity: 2^(-average log probability)
        if token_count > 0:
            perplexity = 2 ** (-log_prob / token_count)
        else:
            perplexity = float("inf")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log results with final system metrics
        self.log_operation_with_metrics(
            f"Perplexity calculation complete: {perplexity:.4f}",
            "perplexity_complete",
            {
                "text_id": text_id,
                "perplexity": perplexity,
                "tokens_processed": token_count,
                "zero_probability_tokens": zero_prob_count,
                "execution_time": execution_time,
                "preprocess_time": preprocess_time
            }
        )

        return perplexity

    def find_high_probability_sequences(self, length=3, top_n=10):
        """
        Find the most likely sequences of words in the model.

        Args:
            length (int): The length of sequences to find
            top_n (int): Number of top sequences to return

        Returns:
            list: Top sequences with their probabilities
        """
        start_time = time.time()

        # Log start of sequence finding with initial system metrics
        self.log_operation_with_metrics(
            f"Finding high probability sequences of length {length}",
            "find_sequences_start",
            {
                "sequence_length": length,
                "top_n": top_n,
                "storage_type": ("database" if self.markov_chain.using_db else "memory")
            }
        )

        if self.markov_chain.using_db and self.db_adapter:
            # Track system metrics before database operation
            before_db_metrics = self.get_system_metrics()

            results = self.db_adapter.find_high_probability_sequences(
                self.markov_chain.n_gram, length, top_n)

            # Log database operation completion with metrics
            self.log_operation_with_metrics(
                f"Database query for high probability sequences completed",
                "find_sequences_db_complete",
                {
                    "results_count": len(results),
                    "before_metrics": before_db_metrics,
                    "storage_type": "database"
                },
                "debug"
            )
        else:
            results = self._find_high_probability_sequences_memory(
                length, top_n)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare result metrics for logging
        result_metrics = []
        for i, (sequence, probability) in enumerate(
            results[:5]
        ):  # Log only top 5 for brevity
            result_metrics.append(
                {"rank": i + 1, "sequence": sequence, "probability": probability}
            )

        # Log results with final system metrics
        self.log_operation_with_metrics(
            f"Found {len(results)} high probability sequences in {execution_time:.3f}s",
            "find_sequences_complete",
            {
                "execution_time": execution_time,
                "results_count": len(results),
                "top_results": result_metrics
            }
        )

        return results

    def _find_high_probability_sequences_memory(self, length=3, top_n=10):
        """
        Find high probability sequences in memory-based Markov Chain model.

        Args:
            length (int): The length of sequences to find
            top_n (int): Number of top sequences to return

        Returns:
            list: Top sequences with their probabilities
        """
        # Track progress metrics
        start_time = time.time()
        sequences_checked = 0
        sequences_found = 0
        sequences_list = []

        # Log memory search start with system metrics
        self.log_operation_with_metrics(
            "Starting in-memory sequence search",
            "find_sequences_memory_start",
            {
                "state_count": len(self.markov_chain.transitions),
                "target_length": length,
                "storage_type": "memory"
            },
            "debug"
        )

        # Get all states as potential starting points
        all_states = list(self.markov_chain.transitions.keys())
        total_states = len(all_states)
        # Log 10 times during processing
        progress_threshold = max(1, total_states // 10)

        # For each starting state, find sequences by following transitions
        for state_idx, start_state in enumerate(all_states):
            # Log progress for large models
            if state_idx > 0 and state_idx % progress_threshold == 0:
                progress_percent = (state_idx / total_states) * 100
                self.log_operation_with_metrics(
                    f"Sequence search progress: {progress_percent:.1f}%",
                    "find_sequences_memory_progress",
                    {
                        "progress_percent": progress_percent,
                        "states_processed": state_idx,
                        "total_states": total_states,
                        "sequences_found": sequences_found,
                        "sequences_checked": sequences_checked
                    },
                    "debug"
                )

            # For single word state
            if self.markov_chain.n_gram == 1:
                current_state = start_state

                # Try to build sequences of desired length
                for _ in range(top_n * 2):  # Sample more than needed and take top_n
                    sequence = [current_state]
                    current_probability = 1.0

                    # Build a sequence of required length by following transitions
                    for _ in range(length - 1):
                        if current_state not in self.markov_chain.transitions:
                            break

                        next_words = self.markov_chain.transitions[current_state]
                        if not next_words:
                            break

                        # Choose next word based on probabilities
                        total = self.markov_chain.total_counts[current_state]
                        next_word_items = list(next_words.items())
                        next_word_probs = [
                            count / total for _, count in next_word_items
                        ]
                        next_word_idx = self._weighted_choice(next_word_probs)

                        if next_word_idx is None:
                            break

                        next_word, count = next_word_items[next_word_idx]
                        transition_prob = count / total

                        sequence.append(next_word)
                        current_probability *= transition_prob
                        current_state = next_word

                    # If we successfully built a sequence of required length
                    if len(sequence) == length:
                        sequences_list.append(
                            (" ".join(sequence), current_probability)
                        )
                        sequences_found += 1

                    sequences_checked += 1
                    current_state = start_state  # Reset for next attempt
            else:
                # For multi-word state (tuple)
                current_state = start_state
                sequence = list(current_state)  # Start with initial n-gram

                # Try to build sequences of required length
                for _ in range(
                    min(top_n * 2, 50)
                ):  # Sample some sequences from each start state
                    sequence = list(current_state)  # Reset to start state
                    current_probability = 1.0

                    # Build a sequence of required additional words
                    for _ in range(length - len(sequence)):
                        if current_state not in self.markov_chain.transitions:
                            break

                        next_words = self.markov_chain.transitions[current_state]
                        if not next_words:
                            break

                        # Choose next word based on probabilities
                        total = self.markov_chain.total_counts[current_state]
                        next_word_items = list(next_words.items())
                        next_word_probs = [
                            count / total for _, count in next_word_items
                        ]
                        next_word_idx = self._weighted_choice(next_word_probs)

                        if next_word_idx is None:
                            break

                        next_word, count = next_word_items[next_word_idx]
                        transition_prob = count / total

                        sequence.append(next_word)
                        current_probability *= transition_prob

                        # Update current state by shifting words for n-gram
                        if len(sequence) >= self.markov_chain.n_gram:
                            current_state = tuple(
                                sequence[-(self.markov_chain.n_gram):]
                            )

                    # If we successfully built a sequence of required length
                    if len(sequence) >= length:
                        sequences_list.append(
                            (" ".join(sequence), current_probability)
                        )
                        sequences_found += 1

                    sequences_checked += 1

        # Log memory search completion with metrics before sorting
        self.log_operation_with_metrics(
            f"In-memory sequence search completed, found {sequences_found} sequences",
            "find_sequences_memory_complete",
            {
                "sequences_found": sequences_found,
                "sequences_checked": sequences_checked,
                "execution_time": time.time() - start_time
            },
            "debug"
        )

        # Sort by probability (descending) and take top_n
        top_sequences = sorted(
            sequences_list, key=lambda x: x[1], reverse=True)[:top_n]

        return top_sequences

    def _weighted_choice(self, probabilities):
        """
        Select an index randomly according to the weights in probabilities.

        Args:
            probabilities (list): List of probability weights

        Returns:
            int: Selected index or None if probabilities is empty
        """
        if not probabilities:
            return None

        total = sum(probabilities)
        if total == 0:
            return None

        r = random.random() * total
        cumulative_sum = 0

        for i, prob in enumerate(probabilities):
            cumulative_sum += prob
            if r <= cumulative_sum:
                return i

        # Fallback to avoid edge cases
        return len(probabilities) - 1


# Example usage
def example_analytics(logger=None):
    """
    Example function demonstrating analytics usage

    Args:
        logger: A logger instance for logging. If None, a default logger will be used.
    """
    from markov_chain import MarkovChain
    from utils.loggers.json_logger import get_logger

    # Ensure we have a logger
    if logger is None:
        logger = get_logger("markov_chain_analytics_example")

    # Create and train a simple model
    markov = MarkovChain(n_gram=2, logger=logger)
    text = "the cat sat on the mat. the dog sat on the floor. the cat saw the dog."
    markov.train(text)

    # Create analytics object
    analytics = MarkovChainAnalytics(markov, logger=logger)

    # Get model statistics
    stats = analytics.analyze_model()
    print(f"\033[1m\nModel Analysis:\033[0m")
    for key, value in stats.items():
        if key == "top_transitions":
            print(f"\033[1m{key}: \033[0m")
            for transition in value:
                print(f"\033[1m  {transition}\033[0m")
        elif key == "system_metrics":
            print(f"\033[1m{key}: \033[0m")
            print(
                f"\033[1m  Memory: {value['memory']['current_mb']:.2f} MB (Peak: {value['memory']['peak_mb']:.2f} MB)\033[0m")
            print(
                f"\033[1m  CPU: {value['cpu']['process_percent']:.1f}% (System: {value['cpu']['system_percent']:.1f}%)\033[0m")
            print(
                f"\033[1m  Threads: {value['threads']['active_count']}\033[0m")
        else:
            print(f"\033[1m{key}: {value}\033[0m")

    # Calculate sequence probability
    seq = "the cat sat"
    score = analytics.score_sequence(seq)
    print(f"\033[1m\nSequence '{seq}' score: {score}\033[0m")

    # Get high probability sequences
    top_sequences = analytics.find_high_probability_sequences(
        length=3, top_n=5)
    print(f"\033[1m\nTop sequences:\033[0m")
    for seq, prob in top_sequences:
        print(f"\033[1m  {seq}: {prob:.4f}\033[0m")

    # Calculate perplexity
    test_text = "the cat sat on the floor"
    perplexity = analytics.perplexity(test_text)
    print(f"\033[1m\nPerplexity on '{test_text}': {perplexity:.4f}\033[0m")

    # Print final system resource usage
    system_metrics = analytics.get_system_metrics()
    print(f"\033[1m\nFinal System Resource Usage:\033[0m")
    print(
        f"\033[1m  Memory: {system_metrics['memory']['current_mb']:.2f} MB (Peak: {system_metrics['memory']['peak_mb']:.2f} MB)\033[0m")
    print(
        f"\033[1m  CPU: {system_metrics['cpu']['process_percent']:.1f}% (System: {system_metrics['cpu']['system_percent']:.1f}%)\033[0m")
    print(
        f"\033[1m  Cores: {system_metrics['cpu']['physical_cores']} physical, {system_metrics['cpu']['total_cores']} logical\033[0m")
    print(
        f"\033[1m  Threads: {system_metrics['threads']['active_count']}\033[0m")
    print(
        f"\033[1m  Load Average: {system_metrics['cpu']['load_avg_1min']:.2f} (1 min), {system_metrics['cpu']['load_avg_5min']:.2f} (5 min)\033[0m")
