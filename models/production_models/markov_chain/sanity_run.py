# Example usage for text generation
import os
import sys
import datetime
import uuid
import time

# Add project root to Python path directly to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import modules after Python path is set up
from models.production_models.markov_chain.analytics import MarkovChainAnalytics
from models.production_models.markov_chain.markov_chain import MarkovChain
from utils.loggers.json_logger import get_logger


class MarkovChainSanityRun:
    """
    A class to perform sanity testing of Markov Chain functionality
    with structured JSON logging similar to the analytics module.
    """

    def __init__(self, logger=None):
        """
        Initialize the sanity run class with a logger.

        Args:
            logger: A logger instance for logging activities (optional)
        """
        # Ensure logger is provided
        if logger is None:
            self.logger = get_logger("markov_chain_sanity_run")
        else:
            self.logger = logger

        # Generate unique identifier for this run
        self.run_id = str(uuid.uuid4())[:8]

        # Log initialization
        self.logger.info("MarkovChainSanityRun initialized", extra={
            "metrics": {
                "run_type": "sanity_test",
                "run_id": self.run_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
        })

    def run_text_generation_example(self):
        """Run a basic text generation example"""
        start_time = time.time()

        self.logger.info("Starting text generation example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "text_generation_example"
            }
        })

        print("\033[1mExample usage for generating text\033[0m\n")
        markov_chain = MarkovChain(
            n_gram=2, memory_threshold=10000, environment="test", logger=self.logger)
        text = "It was a bright cold day in April, and the clocks were striking thirteen."
        markov_chain.train(text)
        generated_text = markov_chain.generate_text(
            start="It was", max_length=50)
        print(f"\033[1m{generated_text}\033[0m")
        print("\n")

        execution_time = time.time() - start_time
        self.logger.info("Text generation example completed", extra={
            "metrics": {
                "input_text_length": len(text),
                "generated_text": generated_text,
                "execution_time": execution_time,
                "n_gram": 2,
                "start_text": "It was",
                "run_id": self.run_id
            }
        })

        return generated_text

    def run_next_word_prediction_example(self):
        """Run a word prediction example"""
        start_time = time.time()

        self.logger.info("Starting next word prediction example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "word_prediction_example"
            }
        })

        print("\033[1mExample usage for predicting next word\033[0m\n")
        markov_chain = MarkovChain(
            n_gram=1, memory_threshold=10000, environment="test", logger=self.logger)
        text = "It was a bright cold day in April, and the clocks were striking thirteen."
        markov_chain.train(text)
        predicted_word = markov_chain.predict("striking")
        print(f"\033[1m{predicted_word}\033[0m")
        print("\n")

        execution_time = time.time() - start_time
        self.logger.info("Word prediction example completed", extra={
            "metrics": {
                "input_text_length": len(text),
                "predicted_word": predicted_word,
                "execution_time": execution_time,
                "n_gram": 1,
                "input_word": "striking",
                "run_id": self.run_id
            }
        })

        return predicted_word

    def run_postgres_generation_example(self):
        """Run a text generation example using PostgreSQL storage"""
        start_time = time.time()

        self.logger.info("Starting PostgreSQL text generation example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "postgres_generation_example",
                "storage": "postgresql"
            }
        })

        print(
            "\033[1mExample usage for generating text using PostgreSQL using test environment\033[0m\n")
        markov_chain_test = MarkovChain(
            n_gram=2, memory_threshold=10000, environment="test", logger=self.logger)
        text = "It was a bright cold day in April, and the clocks were striking thirteen."
        markov_chain_test.train(text)
        generated_text_test = markov_chain_test.generate_text(
            start="It was", max_length=50)
        print(f"\033[1m{generated_text_test}\033[0m")
        print("\n")

        execution_time = time.time() - start_time
        self.logger.info("PostgreSQL text generation example completed", extra={
            "metrics": {
                "input_text_length": len(text),
                "generated_text": generated_text_test,
                "execution_time": execution_time,
                "n_gram": 2,
                "start_text": "It was",
                "storage": "postgresql",
                "run_id": self.run_id
            }
        })

        return generated_text_test

    def run_postgres_prediction_example(self):
        """Run a word prediction example using PostgreSQL storage"""
        start_time = time.time()

        self.logger.info("Starting PostgreSQL word prediction example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "postgres_prediction_example",
                "storage": "postgresql"
            }
        })

        print(
            "\033[1mExample usage for predicting next word using PostgreSQL using test environment\033[0m\n")
        markov_chain_test = MarkovChain(
            n_gram=2, memory_threshold=10000, environment="test", logger=self.logger)
        text = "It was a bright cold day in April, and the clocks were striking thirteen."
        markov_chain_test.train(text)
        predicted_word_test = markov_chain_test.predict("It was")
        print(f"\033[1m{predicted_word_test}\033[0m")
        print("\n")

        execution_time = time.time() - start_time
        self.logger.info("PostgreSQL word prediction example completed", extra={
            "metrics": {
                "input_text_length": len(text),
                "predicted_word": predicted_word_test,
                "execution_time": execution_time,
                "n_gram": 2,
                "input_phrase": "It was",
                "storage": "postgresql",
                "run_id": self.run_id
            }
        })

        return predicted_word_test

    def run_preprocessed_example(self):
        """Run an example with text preprocessing"""
        start_time = time.time()

        self.logger.info("Starting preprocessed text example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "preprocessed_text_example"
            }
        })

        print("\033[1mExample usage with preprocessing\033[0m\n")
        markov_chain = MarkovChain(
            n_gram=2, memory_threshold=10000, environment="test", logger=self.logger)

        # Raw text with various issues that preprocessing will handle
        raw_text = """It was a bright cold day in April, and the clocks were striking thirteen. 
        Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, 
        slipped quickly through the glass doors of Victory Mansions, though not quickly 
        enough to prevent a swirl of gritty dust from entering along with him.
        http://example.com/test?page=1 
        <b>HTML tags</b> should be removed!
        Don't forget about contractions :) 😊"""

        # Train with preprocessing
        markov_chain.train(raw_text, preprocess=True)

        # Generate text
        generated_text = markov_chain.generate_text(
            start="It was", max_length=50)
        print("\033[1mGenerated text with preprocessed training:\033[0m")
        print(f"\033[1m{generated_text}\033[0m")
        print("\n")

        # Compare with specific normalization
        normalized_start = markov_chain._preprocess_text(
            "It's cold in April, don't you think? 🥶"
        )
        print("\033[1mNormalized input:\033[0m",
              f"\033[1m{normalized_start}\033[0m")
        generated_normalized = markov_chain.generate_text(
            start=normalized_start, max_length=30)
        print("\033[1mGenerated from normalized input:\033[0m")
        print(f"\033[1m{generated_normalized}\033[0m")
        print("\n")

        # Create a log entry for the normalized input generation
        self.logger.info("Generated from normalized input", extra={
            "metrics": {
                "original_input": "It's cold in April, don't you think? 🥶",
                "normalized_input": normalized_start,
                "generated_text": generated_normalized,
                "execution_time": time.time() - start_time,
                "run_id": self.run_id
            }
        })

        return generated_normalized

    def run_analytics_example(self):
        """Run analytics on a model"""
        start_time = time.time()

        self.logger.info("Starting analytics example", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "analytics_example"
            }
        })

        print("\033[1mAnalytics of the model\033[0m\n")
        print("\033[1mRunning example analytics...\033[0m")

        # Create and train a simple model
        markov = MarkovChain(n_gram=2, logger=self.logger)
        text = "the cat sat on the mat. the dog sat on the floor. the cat saw the dog."
        markov.train(text)

        # Log the analytics model training
        self.logger.info("Analytics model trained", extra={
            "metrics": {
                "text": text,
                "n_gram": 2,
                "run_id": self.run_id
            }
        })

        # Create analytics object
        analytics = MarkovChainAnalytics(markov, logger=self.logger)

        # Get model statistics
        stats = analytics.analyze_model()
        print("\033[1m\nModel Analysis:\033[0m")
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
        print("\033[1m\nTop sequences:\033[0m")
        for seq, prob in top_sequences:
            print(f"\033[1m  {seq}: {prob:.4f}\033[0m")

        # Calculate perplexity
        test_text = "the cat sat on the floor"
        perplexity = analytics.perplexity(test_text)
        print(f"\033[1m\nPerplexity on '{test_text}': {perplexity:.4f}\033[0m")

        execution_time = time.time() - start_time
        self.logger.info("Analytics example completed", extra={
            "metrics": {
                "execution_time": execution_time,
                "test_sequence": seq,
                "sequence_score": score,
                "test_text": test_text,
                "perplexity": perplexity,
                "run_id": self.run_id
            }
        })

        return {
            "stats": stats,
            "score": score,
            "top_sequences": top_sequences,
            "perplexity": perplexity
        }

    def run_all_examples(self):
        """Run all examples in sequence"""
        overall_start_time = time.time()

        self.logger.info("Starting complete sanity run", extra={
            "metrics": {
                "run_id": self.run_id,
                "operation": "complete_sanity_run"
            }
        })

        # Run all examples
        self.run_text_generation_example()
        self.run_next_word_prediction_example()
        self.run_postgres_generation_example()
        self.run_postgres_prediction_example()
        self.run_preprocessed_example()
        self.run_analytics_example()

        # Log completion
        execution_time = time.time() - overall_start_time
        self.logger.info("Complete sanity run finished", extra={
            "metrics": {
                "total_execution_time": execution_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "run_id": self.run_id
            }
        })

        print(
            f"\033[1mTest run completed in {execution_time:.2f} seconds.\033[0m")


# Main execution
if __name__ == "__main__":
    # Initialize the logger for this test run
    logger = get_logger("markov_chain_sanity_run")

    # Create and run the sanity test
    sanity_run = MarkovChainSanityRun(logger=logger)
    sanity_run.run_all_examples()
