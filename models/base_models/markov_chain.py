from collections import defaultdict
import random


class MarkovChain:
    """
    A simple implementation of a Markov Chain for text generation.
    """

    def __init__(self, n_gram=1):
        """
        Initializes the Markov Chain with:
        - `transitions`: A nested dictionary to store word transition counts.
        - `total_counts`: A dictionary to store the total count of transitions for each word.
        - `n_gram`: The number of words to consider as a state.

        Args:
            n_gram (int): Number of words to consider as a state (default: 1)
        """
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.n_gram = n_gram

    def train(self, text):
        """
        Trains the Markov Chain on a given text by learning word transitions.

        Args:
            text (str): The input text used to train the Markov Chain.
        """
        words = text.split()
        if len(words) < self.n_gram + 1:
            # If there aren't enough words, no transitions can be learned
            return

        # Clear previous counts to ensure consistency
        self.transitions.clear()
        self.total_counts.clear()

        # Count transitions for n-grams
        for i in range(len(words) - self.n_gram):
            # Create n-gram state
            if self.n_gram == 1:
                current_state = words[i]
                next_word = words[i + 1]
            else:
                current_state = tuple(words[i:i + self.n_gram])
                next_word = words[i + self.n_gram]

            # Update transitions
            self.transitions[current_state][next_word] += 1

            # Directly update total counts
            self.total_counts[current_state] += 1

    def predict(self, current_state):
        """
        Predicts the next word based on the current state using learned probabilities.

        Args:
            current_state: The current state (word or tuple of words) for prediction.

        Returns:
            str or None: The predicted next word, or None if the current state is not in the model.
        """
        if current_state not in self.transitions:
            return None

        next_words = self.transitions[current_state]
        total = self.total_counts[current_state]
        probabilities = {word: count / total for word, count in next_words.items()}

        return random.choices(
            list(probabilities.keys()), weights=probabilities.values()
        )[0]
    
    def generate_text(self, start=None, max_length=100):
        """
        Generates text starting from a given state.

        Args:
            start: Starting state (word or tuple). If None, a random state is chosen.
            max_length (int): Maximum number of words to generate.

        Returns:
            str: Generated text.
        """
        if not self.transitions:
            return "Model not trained"

        # Select a random starting state if not provided
        if start is None:
            start = random.choice(list(self.transitions.keys()))
        
        # Convert string to tuple for n-grams if necessary
        if isinstance(start, str) and self.n_gram > 1:
            words = start.split()
            if len(words) >= self.n_gram:
                start = tuple(words[:self.n_gram])
            else:
                # Not enough words provided, use random start
                start = random.choice(list(self.transitions.keys()))
        
        # Validate that start is in transitions
        if start not in self.transitions:
            start = random.choice(list(self.transitions.keys()))
        
        # Generate text
        current_state = start
        text = []
        
        # Convert tuple to list of words for output
        if isinstance(current_state, tuple):
            text.extend(list(current_state))
        else:
            text.append(current_state)
        
        # Generate remaining words
        for _ in range(max_length - (self.n_gram if isinstance(current_state, tuple) else 1)):
            next_word = self.predict(current_state)
            if next_word is None:
                break
            
            text.append(next_word)
            
            # Update current state for n-grams
            if self.n_gram > 1:
                current_state = tuple(list(current_state)[1:] + [next_word])
            else:
                current_state = next_word
        
        return " ".join(text)

# Usage Example for text generation
# -------------------------------

# Example usage
text = "It was a sunny day. The cat sat on the mat. The dog barked."
markov_chain = MarkovChain(n_gram=2)
markov_chain.train(text)
generated_text = markov_chain.generate_text(start="It was", max_length=50)
print("Generated Text: \n")
print(generated_text)

# Usage example for predicting the next word
# -------------------------------
# Example usage
text = "It was a sunny day. The cat sat on the mat. The dog barked."
markov_chain = MarkovChain(n_gram=1)
markov_chain.train(text)

# Predict next word
current_state = "cat"
predicted_word = markov_chain.predict(current_state)
print(f"Predicted next word after '{current_state}': {predicted_word}")