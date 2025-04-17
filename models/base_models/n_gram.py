# Import necessary libraries
import random  # For random sampling

# For creating nested dictionaries with default values
from collections import defaultdict


class NGramModel:
    """
    A simple implementation of an N-gram language model for text generation.
    The N-gram model learns word transitions based on sequences of (n-1) words
    and predicts the next word based on the learned probabilities.
    """

    def __init__(self, n=3):
        """
        Initializes the N-gram model.

        Args:
            n (int): The size of the N-gram (default is 3, i.e., trigram).

        Attributes:
            n (int): The size of the N-gram.
            ngram_counts (defaultdict): A nested dictionary to store N-gram transition counts.
                Example: ngram_counts[("the", "cat")]["sat"] = 2
            total_counts (defaultdict): A dictionary to store the total count of transitions
                for each (n-1)-gram.
                Example: total_counts[("the", "cat")] = 5
        """
        self.n = n
        # Transition counts for N-grams
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        # Total transition counts for each (n-1)-gram
        self.total_counts = defaultdict(int)

    def train(self, text):
        """
        Trains the N-gram model on a given text by learning word transitions.

        Args:
            text (str): The input text used to train the N-gram model.

        Steps:
        1. Splits the input text into words.
        2. Iterates through consecutive N-grams in the text.
        3. Updates the transition counts and total counts for each (n-1)-gram.

        Example:
            Input: "the cat sat on the mat"
            Updates:
                - ngram_counts[("the", "cat")]["sat"] += 1
                - ngram_counts[("cat", "sat")]["on"] += 1
                - total_counts[("the", "cat")] += 1
                - total_counts[("cat", "sat")] += 1
        """
        words = text.split()  # Split the text into words
        for i in range(len(words) - self.n + 1):  # Iterate through consecutive N-grams
            # Take the last (n-1) words as the N-gram
            ngram = tuple(words[i: i + self.n - 1])
            next_word = words[i + self.n - 1]  # The next word to predict
            # Increment transition count
            self.ngram_counts[ngram][next_word] += 1
            # Increment total count for the N-gram
            self.total_counts[ngram] += 1

    def predict(self, words):
        """
        Predicts the next word based on the given (n-1) words using learned probabilities.

        Args:
            words (list of str): The last (n-1) words for which the next word is predicted.

        Returns:
            str or None: The predicted next word, or None if the (n-1) words are not in the model.

        Steps:
        1. Converts the input words into a tuple of the last (n-1) words.
        2. Checks if the (n-1) words exist in the N-gram model.
        3. Retrieves the possible next words and their counts.
        4. Calculates the probabilities for each next word.
        5. Samples a next word based on the calculated probabilities.

        Example:
            If ngram_counts[("the", "cat")] = {"sat": 2, "jumped": 1} and total_counts[("the", "cat")] = 3:
                - Probabilities: {"sat": 2/3, "jumped": 1/3}
                - Randomly selects "sat" or "jumped" based on these probabilities.
        """
        words = tuple(words[-(self.n - 1):])  # Take the last (n-1) words
        if words not in self.ngram_counts:
            # No prediction available if the (n-1) words are not in the model
            return None

        # Get possible next words and their counts
        next_words = self.ngram_counts[words]
        # Get the total count for the (n-1) words
        total = self.total_counts[words]
        probabilities = {
            word: count / total for word, count in next_words.items()
        }  # Calculate probabilities

        # Randomly select the next word based on the probabilities
        return random.choices(
            list(probabilities.keys()), weights=probabilities.values()
        )[0]


# -------------------------------
# Example Usage
# -------------------------------


# Define a sample text for training
text = "the cat sat on the mat the cat jumped over the mat"

# Create an instance of the NGramModel class
ngram_model = NGramModel(n=3)  # Trigram model

# Train the N-gram model on the sample text
ngram_model.train(text)

# Predict the next word for the given (n-1) words
# The output is likely to be "sat" or "jumped" based on the learned probabilities.
print(ngram_model.predict(["the", "cat"]))  # Example output: "sat" or "jumped"
print(ngram_model.predict(["over", "the"]))  # Example output: "mat"
