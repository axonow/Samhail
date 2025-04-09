# Import necessary libraries
import random  # For random sampling

# For creating nested dictionaries with default values
from collections import defaultdict

# -------------------------------
# Markov Chain Class Definition
# -------------------------------


class MarkovChain:
    """
    A simple implementation of a Markov Chain for text generation.
    The Markov Chain learns word transitions from a given text and predicts the next word
    based on the learned probabilities.
    """

    def __init__(self):
        """
        Initializes the Markov Chain with:
        - `transitions`: A nested dictionary to store word transition counts.
        - `total_counts`: A dictionary to store the total count of transitions for each word.
        """
        self.transitions = defaultdict(
            lambda: defaultdict(int)
        )  # Transition counts between words
        # Total transition counts for each word
        self.total_counts = defaultdict(int)

    def train(self, text):
        """
        Trains the Markov Chain on a given text by learning word transitions.

        Args:
            text (str): The input text used to train the Markov Chain.

        Steps:
        1. Splits the input text into words.
        2. Iterates through consecutive word pairs in the text.
        3. Updates the transition counts and total counts for each word.
        Note: The last word in the text does not have a next word, so it is ignored.

        Example:
            Input: "the cat sat on the mat"
            Updates:
                - transitions["the"]["cat"] += 1
                - transitions["cat"]["sat"] += 1
                - total_counts["the"] += 1
                - total_counts["cat"] += 1
        """
        words = text.split()  # Split the text into words
        for i in range(len(words) - 1):  # Iterate through all valid transitions
            current_word = words[i]
            next_word = words[i + 1]

            # Update transitions
            if current_word not in self.transitions:
                self.transitions[current_word] = {}
            if next_word not in self.transitions[current_word]:
                self.transitions[current_word][next_word] = 0
            self.transitions[current_word][next_word] += 1

            # Update total counts
            if current_word not in self.total_counts:
                self.total_counts[current_word] = 0
            self.total_counts[current_word] += 1

    def predict(self, current_word):
        """
        Predicts the next word based on the current word using learned probabilities.

        Args:
            current_word (str): The current word for which the next word is predicted.

        Returns:
            str or None: The predicted next word, or None if the current word is not in the model.

        Steps:
        1. Checks if the current word exists in the transitions dictionary.
        2. Retrieves the possible next words and their counts.
        3. Calculates the probabilities for each next word.
        4. Samples a next word based on the calculated probabilities.

        Example:
            If transitions["cat"] = {"sat": 2, "jumped": 1} and total_counts["cat"] = 3:
                - Probabilities: {"sat": 2/3, "jumped": 1/3}
                - Randomly selects "sat" or "jumped" based on these probabilities.
        """
        if current_word not in self.transitions:
            return None  # No prediction available if the word is not in the model

        # Get possible next words and their counts
        next_words = self.transitions[current_word]
        # Get the total count for the current word
        total = self.total_counts[current_word]
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

# Create an instance of the MarkovChain class
mc = MarkovChain()

# Train the Markov Chain on the sample text
mc.train(text)

# Predict the next word for the word "cat"
# The output is likely to be "sat" or "jumped" based on the learned probabilities.
print(mc.predict("cat"))
# Predict the next word for the word "the"
# The output is likely to be "cat" based on the learned probabilities.
print(mc.predict("the"))
