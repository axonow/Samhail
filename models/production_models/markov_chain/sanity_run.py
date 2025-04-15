# Example usage for text generation
from utils.loggers.json_logger import get_logger, log_json
from models.production_models.markov_chain.markov_chain import MarkovChain
from models.production_models.markov_chain.analytics import MarkovChainAnalytics
import os
import sys
import datetime

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Import the MarkovChain and analytics modules

# Initialize the logger for this test run
logger = get_logger("test_run")

print("\033[1mExample usage for generating text\033[0m\n")
markov_chain = MarkovChain(
    n_gram=2, memory_threshold=10000, environment="test", logger=logger)
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain.train(text)
generated_text = markov_chain.generate_text(start="It was", max_length=50)
print(f"\033[1m{generated_text}\033[0m")
print("\n")

print("\033[1mExample usage for predicting next word\033[0m\n")
markov_chain = MarkovChain(
    n_gram=1, memory_threshold=10000, environment="test", logger=logger)
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain.train(text)
predicted_word = markov_chain.predict("striking")
print(f"\033[1m{predicted_word}\033[0m")
print("\n")

print(
    "\033[1mExample usage for generating text using PostgreSQL using test environment\033[0m\n"
)
markov_chain_test = MarkovChain(
    n_gram=2, memory_threshold=10000, environment="test", logger=logger)
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain_test.train(text)
generated_text_test = markov_chain_test.generate_text(
    start="It was", max_length=50)
print(f"\033[1m{generated_text_test}\033[0m")
print("\n")

print(
    "\033[1mExample usage for predicting next word using PostgreSQL using test environment\033[0m\n"
)
markov_chain_test = MarkovChain(
    n_gram=2, memory_threshold=10000, environment="test", logger=logger)
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain_test.train(text)
predicted_word_test = markov_chain_test.predict("It was")
print(f"\033[1m{predicted_word_test}\033[0m")
print("\n")

print("\033[1mExample usage with preprocessing\033[0m\n")
markov_chain = MarkovChain(
    n_gram=2, memory_threshold=10000, environment="test", logger=logger)

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
generated_text = markov_chain.generate_text(start="It was", max_length=50)
print("\033[1mGenerated text with preprocessed training:\033[0m")
print(f"\033[1m{generated_text}\033[0m")
print("\n")

# Compare with specific normalization
normalized_start = markov_chain._preprocess_text(
    "It's cold in April, don't you think? 🥶"
)
print("\033[1mNormalized input:\033[0m", f"\033[1m{normalized_start}\033[0m")
generated_normalized = markov_chain.generate_text(
    start=normalized_start, max_length=30)
print("\033[1mGenerated from normalized input:\033[0m")
print(f"\033[1m{generated_normalized}\033[0m")
print("\n")

# Create a log entry for the normalized input generation
log_json(
    logger,
    "Generated from normalized input",
    {
        "original_input": "It's cold in April, don't you think? 🥶",
        "normalized_input": normalized_start,
        "generated_text": generated_normalized,
    },
)

print("\033[1mAnalytics of the model\033[0m\n")


def example_analytics():
    # Create and train a simple model
    markov = MarkovChain(n_gram=2, logger=logger)
    text = "the cat sat on the mat. the dog sat on the floor. the cat saw the dog."
    markov.train(text)

    # Log the analytics model training
    log_json(
        logger,
        "Analytics model trained",
        {"text": text, "n_gram": 2},
    )

    # Create analytics object
    analytics = MarkovChainAnalytics(markov, logger=logger)

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


print("\033[1mRunning example analytics...\033[0m")
example_analytics()

# Add final log entry
log_json(
    logger,
    "Test run completed",
    {"timestamp": datetime.datetime.now().isoformat()},
)

print(f"\033[1mTest run completed.\033[0m")
