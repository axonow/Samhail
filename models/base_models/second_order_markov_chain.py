# Import necessary libraries
import random  # For random sampling
from collections import defaultdict  # For creating nested dictionaries with default values

class SecondOrderMarkovChain:
    """
    A simple implementation of a second-order Markov Chain for text generation.
    The second-order Markov Chain predicts the next word based on the two previous words
    by learning word transitions from a given text.
    """

    def __init__(self):
        """
        Initializes the SecondOrderMarkovChain with:
        - `transitions`: A nested dictionary to store second-order word transition counts.
            Example: transitions["the"]["cat"]["sat"] = 2
        - `total_counts`: A nested dictionary to store the total count of transitions
          for each pair of consecutive words.
            Example: total_counts["the"]["cat"] = 5
        """
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # Transition counts
        self.total_counts = defaultdict(lambda: defaultdict(int))  # Total transition counts for word pairs

    def train(self, text):
        """
        Trains the second-order Markov Chain on a given text by learning word transitions.

        Args:
            text (str): The input text used to train the Markov Chain.

        Steps:
        1. Splits the input text into words.
        2. Iterates through consecutive triplets of words in the text.
        3. Updates the transition counts and total counts for each pair of consecutive words.

        Example:
            Input: "the cat sat on the mat"
            Updates:
                - transitions["the"]["cat"]["sat"] += 1
                - transitions["cat"]["sat"]["on"] += 1
                - total_counts["the"]["cat"] += 1
                - total_counts["cat"]["sat"] += 1
        """
        words = text.split()  # Split the text into words
        for i in range(len(words) - 2):  # Iterate through consecutive triplets of words
            prev1, prev2, next_word = words[i], words[i+1], words[i+2]  # Extract the triplet
            self.transitions[prev1][prev2][next_word] += 1  # Increment transition count
            self.total_counts[prev1][prev2] += 1  # Increment total count for the word pair

    def predict(self, word1, word2):
        """
        Predicts the next word based on the two previous words using learned probabilities.

        Args:
            word1 (str): The first word of the pair.
            word2 (str): The second word of the pair.

        Returns:
            str or None: The predicted next word, or None if the word pair is not in the model.

        Steps:
        1. Checks if the word pair exists in the transitions dictionary.
        2. Retrieves the possible next words and their counts.
        3. Calculates the probabilities for each next word.
        4. Samples a next word based on the calculated probabilities.

        Example:
            If transitions["the"]["cat"] = {"sat": 2, "jumped": 1} and total_counts["the"]["cat"] = 3:
                - Probabilities: {"sat": 2/3, "jumped": 1/3}
                - Randomly selects "sat" or "jumped" based on these probabilities.
        """
        if word1 not in self.transitions or word2 not in self.transitions[word1]:
            return None  # No prediction available if the word pair is not in the model

        next_words = self.transitions[word1][word2]  # Get possible next words and their counts
        total = self.total_counts[word1][word2]  # Get the total count for the word pair
        probabilities = {word: count / total for word, count in next_words.items()}  # Calculate probabilities

        # Randomly select the next word based on the probabilities
        return random.choices(list(probabilities.keys()), weights=probabilities.values())[0]

# -------------------------------
# Example Usage
# -------------------------------

# Define a sample text for training
text = "the cat sat on the mat the cat jumped over the mat"

# Create an instance of the SecondOrderMarkovChain class
mc = SecondOrderMarkovChain()

# Train the Markov Chain on the sample text
mc.train(text)

# Predict the next word for the given pair of words
# The output is likely to be "sat" or "jumped" based on the learned probabilities.
print(mc.predict("the", "cat"))  # Example output: "sat" or "jumped"
