from collections import defaultdict
import random


class MarkovChain:
    """
    A simple implementation of a Markov Chain for text generation.
    """

    def __init__(self):
        """
        Initializes the Markov Chain with:
        - `transitions`: A nested dictionary to store word transition counts.
        - `total_counts`: A dictionary to store the total count of transitions for each word.
        """
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)

    def train(self, text):
        """
        Trains the Markov Chain on a given text by learning word transitions.

        Args:
            text (str): The input text used to train the Markov Chain.
        """
        words = text.split()
        if len(words) < 2:
            # If there are fewer than 2 words, no transitions can be learned
            return

        # Clear previous counts to ensure consistency
        self.transitions.clear()
        self.total_counts.clear()

        # Count transitions and current word occurrences simultaneously
        # Skip the last word as it has no outgoing transition
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            # Update transitions
            self.transitions[current_word][next_word] += 1

            # Directly update total counts for each occurrence of a word
            # This way we only count words that have outgoing transitions
            self.total_counts[current_word] += 1

    def predict(self, current_word):
        """
        Predicts the next word based on the current word using learned probabilities.

        Args:
            current_word (str): The current word for which the next word is predicted.

        Returns:
            str or None: The predicted next word, or None if the current word is not in the model.
        """
        if current_word not in self.transitions:
            return None

        next_words = self.transitions[current_word]
        total = self.total_counts[current_word]
        probabilities = {word: count / total for word,
                         count in next_words.items()}

        return random.choices(
            list(probabilities.keys()), weights=probabilities.values()
        )[0]
