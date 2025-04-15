import math
import random
import time
import uuid
import os
import sys

from utils.loggers.json_logger import log_json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)


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
        self.markov_chain = markov_chain

        # Ensure logger is provided
        if logger is None:
            raise ValueError("Logger instance must be provided")
        self.logger = logger

        # Generate unique identifier for this analytics instance
        self.analytics_id = str(uuid.uuid4())[:8]

        # Log initialization
        self.logger.info("MarkovChainAnalytics initialized", extra={
            "metrics": {
                "model_type": "markov_chain",
                "n_gram": self.markov_chain.n_gram,
                "storage_type": ("database" if self.markov_chain.using_db else "memory"),
                "environment": self.markov_chain.environment,
                "analytics_id": self.analytics_id,
            }
        })

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
        """
        start_time = time.time()

        stats = {
            "storage_type": "database" if self.markov_chain.using_db else "memory",
            "environment": self.markov_chain.environment,
            "n_gram": self.markov_chain.n_gram,
        }

        if self.markov_chain.using_db:
            conn = self.markov_chain._get_connection()
            try:
                with conn.cursor() as cur:
                    # Get total number of transitions
                    table_prefix = f"markov_{self.markov_chain.environment}"
                    cur.execute(
                        f"""
                        SELECT COUNT(*) FROM {table_prefix}_transitions 
                        WHERE n_gram = %s
                    """,
                        (self.markov_chain.n_gram,),
                    )
                    stats["transitions_count"] = cur.fetchone()[0]

                    # Get unique states count
                    cur.execute(
                        f"""
                        SELECT COUNT(*) FROM {table_prefix}_total_counts
                        WHERE n_gram = %s
                    """,
                        (self.markov_chain.n_gram,),
                    )
                    stats["states_count"] = cur.fetchone()[0]

                    # Get top 5 most common state transitions
                    cur.execute(
                        f"""
                        SELECT state, next_word, count 
                        FROM {table_prefix}_transitions
                        WHERE n_gram = %s
                        ORDER BY count DESC
                        LIMIT 5
                    """,
                        (self.markov_chain.n_gram,),
                    )
                    top_transitions = cur.fetchall()
                    stats["top_transitions"] = [
                        {"state": state, "next_word": next_word, "count": count}
                        for state, next_word, count in top_transitions
                    ]

                    # Get vocabulary size (unique words)
                    cur.execute(
                        f"""
                        SELECT COUNT(DISTINCT next_word) 
                        FROM {table_prefix}_transitions
                        WHERE n_gram = %s
                    """,
                        (self.markov_chain.n_gram,),
                    )
                    stats["vocabulary_size"] = cur.fetchone()[0]

                    # Get average transitions per state
                    cur.execute(
                        f"""
                        SELECT AVG(transition_count) FROM (
                            SELECT state, COUNT(*) as transition_count
                            FROM {table_prefix}_transitions
                            WHERE n_gram = %s
                            GROUP BY state
                        ) as state_counts
                    """,
                        (self.markov_chain.n_gram,),
                    )
                    stats["avg_transitions_per_state"] = cur.fetchone()[0]

                    # Get database size
                    cur.execute(
                        f"""
                        SELECT pg_size_pretty(pg_total_relation_size('{table_prefix}_transitions'))
                    """
                    )
                    stats["transitions_table_size"] = cur.fetchone()[0]

                    # Get state distribution metrics
                    cur.execute(
                        f"""
                        SELECT 
                            MIN(count) as min_count,
                            MAX(count) as max_count,
                            AVG(count) as avg_count,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY count) as median_count
                        FROM {table_prefix}_total_counts
                        WHERE n_gram = %s
                    """,
                        (self.markov_chain.n_gram,),
                    )

                    dist_metrics = cur.fetchone()
                    stats["state_count_distribution"] = {
                        "min": dist_metrics[0],
                        "max": dist_metrics[1],
                        "avg": dist_metrics[2],
                        "median": dist_metrics[3],
                    }

            finally:
                self.markov_chain._return_connection(conn)
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

        # Calculate execution time
        execution_time = time.time() - start_time
        stats["analysis_execution_time"] = execution_time

        # Log results
        self.logger.info("Model analysis completed", extra={
            "metrics": {
                "model_metrics": stats,
                "execution_time": execution_time,
                "model_id": self.analytics_id,
            }
        })

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

        if self.markov_chain.using_db:
            prob = self._get_db_transition_probability(
                current_state, next_word)
        else:
            prob = self._get_memory_transition_probability(
                current_state, next_word)

        # Log metrics
        execution_time = time.time() - start_time

        self.logger.debug(f"Transition probability: {current_state} → {next_word} = {prob}", extra={
            "metrics": {
                "current_state": str(current_state),
                "next_word": next_word,
                "probability": prob,
                "execution_time": execution_time,
            },
            "model_id": self.analytics_id,
            "operation": "get_transition_probability",
        })

        return prob

    def _get_memory_transition_probability(self, current_state, next_word):
        """Calculate transition probability using in-memory storage"""
        if current_state not in self.markov_chain.transitions:
            return 0.0

        next_words = self.markov_chain.transitions[current_state]
        total = self.markov_chain.total_counts[current_state]

        # Return probability if transition exists, otherwise 0
        return next_words[next_word] / total if next_word in next_words else 0.0

    def _get_db_transition_probability(self, current_state, next_word):
        """Calculate transition probability using database storage"""
        conn = self.markov_chain._get_connection()
        if not conn:
            return 0.0

        table_prefix = f"markov_{self.markov_chain.environment}"

        try:
            # Convert tuple to string for database query
            if isinstance(current_state, tuple):
                db_state = " ".join(current_state)
            else:
                db_state = str(current_state)

            with conn.cursor() as cur:
                # Get the count for this specific transition
                cur.execute(
                    f"""
                    SELECT count FROM {table_prefix}_transitions 
                    WHERE state = %s AND next_word = %s AND n_gram = %s
                """,
                    (db_state, next_word, self.markov_chain.n_gram),
                )

                transition_result = cur.fetchone()
                if not transition_result:
                    return 0.0

                transition_count = transition_result[0]

                # Get the total count for this state
                cur.execute(
                    f"""
                    SELECT count FROM {table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, self.markov_chain.n_gram),
                )

                total_result = cur.fetchone()
                if not total_result:
                    return 0.0

                total_count = total_result[0]

                # Calculate probability
                return transition_count / total_count

        except Exception as e:
            self.logger.error(f"Error calculating transition probability: {e}", extra={
                "metrics": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                "model_id": self.analytics_id,
                "operation": "get_db_transition_probability",
            })
            return 0.0
        finally:
            self.markov_chain._return_connection(conn)

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

        # Log start of sequence scoring
        self.logger.debug("Starting sequence scoring", extra={
            "metrics": {
                "sequence": sequence[:50] + ("..." if len(sequence) > 50 else ""),
                "sequence_length": len(sequence),
                "word_count": len(sequence.split()),
                "preprocess": preprocess,
                "sequence_id": sequence_id,
            },
            "model_id": self.analytics_id,
            "operation": "score_sequence_start",
        })

        if preprocess and hasattr(self.markov_chain, "_preprocess_text"):
            preprocess_start = time.time()
            sequence = self.markov_chain._preprocess_text(sequence)
            preprocess_time = time.time() - preprocess_start
        else:
            preprocess_time = 0

        words = sequence.split()
        if len(words) <= self.markov_chain.n_gram:
            # Log early return
            self.logger.debug("Sequence too short for scoring", extra={
                "metrics": {
                    "score": 0.0,
                    "reason": "sequence_too_short",
                    "execution_time": time.time() - start_time,
                    "sequence_id": sequence_id,
                    "model_id": self.analytics_id,
                }
            })

            return 0.0

        log_prob = 0.0
        count = 0
        zero_prob_transitions = 0

        # Track transition stats
        transitions = []

        for i in range(len(words) - self.markov_chain.n_gram):
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

        # Normalize by sequence length for fair comparison between different sequences
        final_score = log_prob / max(1, count) if count > 0 else float("-inf")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log results
        self.logger.info(f"Sequence scored with result {final_score:.4f}", extra={
            "metrics": {
                "sequence_id": sequence_id,
                "score": final_score,
                "transitions_checked": count + zero_prob_transitions,
                "zero_probability_transitions": zero_prob_transitions,
                "execution_time": execution_time,
                "preprocess_time": preprocess_time,
                "transitions_sample": transitions[:5] if transitions else [],
            },
            "model_id": self.analytics_id,
            "operation": "score_sequence_complete",
        })

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

        # Log start of perplexity calculation
        self.logger.info("Starting perplexity calculation", extra={
            "metrics": {
                "text_sample": text[:50] + ("..." if len(text) > 50 else ""),
                "text_length": len(text),
                "word_count": len(text.split()),
                "preprocess": preprocess,
                "text_id": text_id,
            },
            "model_id": self.analytics_id,
            "operation": "perplexity_start",
        })

        if preprocess and hasattr(self.markov_chain, "_preprocess_text"):
            preprocess_start = time.time()
            text = self.markov_chain._preprocess_text(text)
            preprocess_time = time.time() - preprocess_start
        else:
            preprocess_time = 0

        words = text.split()
        if len(words) <= self.markov_chain.n_gram:
            # Log early return for text too short
            self.logger.info("Text too short for perplexity calculation", extra={
                "metrics": {
                    "perplexity": float("inf"),
                    "reason": "text_too_short",
                    "execution_time": time.time() - start_time,
                    "text_id": text_id,
                    "model_id": self.analytics_id,
                }
            })

            return float("inf")

        log_prob = 0.0
        token_count = 0
        zero_prob_count = 0

        for i in range(len(words) - self.markov_chain.n_gram):
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

        # Calculate perplexity: 2^(-average log probability)
        if token_count > 0:
            perplexity = 2 ** (-log_prob / token_count)
        else:
            perplexity = float("inf")

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log results
        self.logger.info(f"Perplexity calculation complete: {perplexity:.4f}", extra={
            "metrics": {
                "text_id": text_id,
                "perplexity": perplexity,
                "tokens_processed": token_count,
                "zero_probability_tokens": zero_prob_count,
                "execution_time": execution_time,
                "preprocess_time": preprocess_time,
            },
            "model_id": self.analytics_id,
            "operation": "perplexity_complete",
        })

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

        # Log start of sequence finding
        self.logger.info(f"Finding high probability sequences of length {length}", extra={
            "metrics": {
                "sequence_length": length,
                "top_n": top_n,
                "storage_type": ("database" if self.markov_chain.using_db else "memory"),
            },
            "model_id": self.analytics_id,
            "operation": "find_sequences_start",
        })

        if self.markov_chain.using_db:
            results = self._find_high_probability_sequences_db(length, top_n)
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

        # Log results
        self.logger.info(f"Found {len(results)} high probability sequences in {execution_time:.3f}s", extra={
            "metrics": {
                "execution_time": execution_time,
                "results_count": len(results),
                "top_results": result_metrics,
            },
            "model_id": self.analytics_id,
            "operation": "find_sequences_complete",
        })

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
        sequences_found = []

        # Get all states as potential starting points
        all_states = list(self.markov_chain.transitions.keys())

        # For each starting state, find sequences by following transitions
        for start_state in all_states:
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
                        sequences_found.append(
                            (" ".join(sequence), current_probability)
                        )

                    sequences_checked += 1
                    current_state = start_state  # Reset for next attempt
            else:
                # For multi-word state (tuple)
                current_state = start_state
                sequence = list(current_state)  # Start with initial n-gram

                # Try to build sequences of desired length
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
                        sequences_found.append(
                            (" ".join(sequence), current_probability)
                        )

                    sequences_checked += 1

        # Sort by probability (descending) and take top_n
        top_sequences = sorted(sequences_found, key=lambda x: x[1], reverse=True)[
            :top_n
        ]

        return top_sequences

    def _find_high_probability_sequences_db(self, length=3, top_n=10):
        """
        Find high probability sequences in database-stored Markov Chain model.

        Args:
            length (int): The length of sequences to find
            top_n (int): Number of top sequences to return

        Returns:
            list: Top sequences with their probabilities
        """
        # Track progress metrics
        start_time = time.time()
        sequences_checked = 0
        sequences_found = []

        conn = self.markov_chain._get_connection()
        try:
            table_prefix = f"markov_{self.markov_chain.environment}"

            with conn.cursor() as cur:
                # Get sample of states to use as starting points
                cur.execute(
                    f"""
                    SELECT state 
                    FROM {table_prefix}_total_counts
                    WHERE n_gram = %s
                    ORDER BY count DESC
                    LIMIT 100
                """,
                    (self.markov_chain.n_gram,),
                )

                start_states = [row[0] for row in cur.fetchall()]

                # For each starting state, find sequences by following transitions
                for db_state in start_states:
                    # Convert string state to appropriate format
                    if self.markov_chain.n_gram > 1:
                        start_state = tuple(db_state.split())
                    else:
                        start_state = db_state

                    # Try to build sequences starting from this state
                    current_state = start_state

                    for _ in range(
                        min(top_n, 10)
                    ):  # Sample some sequences from each start state
                        if self.markov_chain.n_gram == 1:
                            sequence = [current_state]
                        else:
                            sequence = list(current_state)

                        current_probability = 1.0

                        # Build a sequence of required length
                        for _ in range(length - len(sequence)):
                            # Get possible next words and their probabilities
                            if self.markov_chain.n_gram > 1:
                                db_current_state = " ".join(current_state)
                            else:
                                db_current_state = current_state

                            cur.execute(
                                f"""
                                SELECT next_word, count 
                                FROM {table_prefix}_transitions
                                WHERE state = %s AND n_gram = %s
                                ORDER BY count DESC
                                LIMIT 10
                            """,
                                (db_current_state, self.markov_chain.n_gram),
                            )

                            next_words = cur.fetchall()
                            if not next_words:
                                break

                            # Get total count for current state
                            cur.execute(
                                f"""
                                SELECT count 
                                FROM {table_prefix}_total_counts
                                WHERE state = %s AND n_gram = %s
                            """,
                                (db_current_state, self.markov_chain.n_gram),
                            )

                            total_result = cur.fetchone()
                            if not total_result:
                                break

                            total = total_result[0]

                            # Calculate probabilities and choose next word
                            next_word_probs = [
                                count / total for _, count in next_words]
                            next_word_idx = self._weighted_choice(
                                next_word_probs)

                            if next_word_idx is None:
                                break

                            next_word, count = next_words[next_word_idx]
                            transition_prob = count / total

                            sequence.append(next_word)
                            current_probability *= transition_prob

                            # Update current state by shifting words for n-gram
                            if self.markov_chain.n_gram > 1:
                                if len(sequence) >= self.markov_chain.n_gram:
                                    current_state = tuple(
                                        sequence[-(self.markov_chain.n_gram):]
                                    )
                            else:
                                current_state = next_word

                        # If we successfully built a sequence of required length
                        if len(sequence) >= length:
                            sequences_found.append(
                                (" ".join(sequence), current_probability)
                            )

                        sequences_checked += 1

        finally:
            self.markov_chain._return_connection(conn)

        # Sort by probability (descending) and take top_n
        top_sequences = sorted(sequences_found, key=lambda x: x[1], reverse=True)[
            :top_n
        ]

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
