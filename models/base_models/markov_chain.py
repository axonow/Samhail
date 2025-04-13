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
    
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
    
            # Update transitions
            self.transitions[current_word][next_word] += 1
    
            # Update total counts
            self.total_counts[current_word] += 1
    
        # Ensure the last word is not included in total_counts
        if words:
            last_word = words[-1]
            if last_word in self.total_counts and not self.transitions[last_word]:
                self.total_counts.pop(last_word, None)

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
        probabilities = {word: count / total for word, count in next_words.items()}

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

