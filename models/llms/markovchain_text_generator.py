
"""
MarkovChain Text Generator

This module implements a simple Markov Chain-based text generator. It allows training on a given text corpus
and generating new text sequences based on the learned Markov Chain model.

Classes:
    - MarkovChain: Represents the Markov Chain model for text generation.

Methods:
    - __init__: Initializes the Markov Chain with an empty graph.
    - _tokenize: Tokenizes input text by removing punctuation, numbers, and splitting into words.
    - train: Trains the Markov Chain by building a graph of word transitions from the input text.
    - generate: Generates a sequence of text based on the trained Markov Chain and a given prompt.

Usage:
    1. Create an instance of the `MarkovChain` class.
    2. Train the model using the `train` method with a text corpus.
    3. Generate new text using the `generate` method with a prompt and desired length.

Example:
    >>> mc = MarkovChain()
    >>> mc.train("This is a simple example. This example is simple.")
    >>> print(mc.generate("This", length=5))
    This is simple example is

Dependencies:
    - random: Used for randomly selecting the next word during text generation.
    - string.punctuation: Used for removing punctuation during tokenization.
    - collections.defaultdict: Used for storing the Markov Chain graph as a dictionary of lists.

Notes:
    - The `_tokenize` method removes punctuation and numbers, converts newlines to spaces, and splits text into words.
    - The `train` method builds a graph where each word points to a list of possible next words.
    - The `generate` method uses the graph to create a sequence of words, starting from the last word in the prompt.

Limitations:
    - The model assumes that the input text is well-formed and does not handle edge cases like empty input gracefully.
    - The generated text may not always be coherent, as it relies purely on word transitions without considering grammar or context.
"""

# The `random` module is used to randomly select the next word during text generation.
# Specifically, the `random.choice` method is used to pick a word from the list of possible next words.
import random

# The `punctuation` constant from the `string` module provides a list of all punctuation characters.
# It is used in the `_tokenize` method to remove punctuation from the input text during tokenization.
from string import punctuation

# The `defaultdict` class from the `collections` module is used to create the Markov Chain graph.
# It initializes the graph as a dictionary where each key (a word) maps to a list of possible next words.
# This simplifies the process of appending new words to the graph without needing to check for key existence.import random
from collections import defaultdict

class MarkovChain:
    def __init__(self):
        """
        Initializes the MarkovChain instance.

        This constructor sets up the Markov Chain graph as a `defaultdict` of lists. 
        Each key in the graph represents a word, and the corresponding value is a list 
        of words that can follow it based on the training data.

        Attributes:
            graph (defaultdict): A dictionary where each key is a word and the value 
                                 is a list of possible next words.
        """
        self.graph = defaultdict(list)

    def _tokenize(self, text):
        """
        Tokenizes the input text by removing punctuation, numbers, and splitting it into words.

        This method processes the input text to prepare it for training or generation. It removes
        all punctuation and numeric characters, replaces newlines with spaces, and splits the text
        into individual words.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of words (tokens) extracted from the input text.

        Notes:
            - Punctuation and numbers are removed using `str.maketrans` and `str.translate`.
            - Newlines are replaced with spaces to ensure consistent tokenization.
            - The resulting text is split into words using the `split` method.
        """
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )
    
    def train(self, text):
        """
        Trains the Markov Chain model by building a graph of word transitions from the input text.

        This method processes the input text, tokenizes it into words, and constructs a graph where 
        each word points to a list of possible next words based on the sequence in the input text.

        Args:
            text (str): The input text used to train the Markov Chain model.

        Notes:
            - The `_tokenize` method is used to preprocess the input text by removing punctuation, 
            numbers, and splitting it into words.
            - The graph is constructed as a `defaultdict` where each key is a word, and the value 
            is a list of words that can follow it in the input text.
            - The method iterates through the tokenized words and appends the next word in the sequence 
            to the list of possible transitions for the current word.

        Example:
            >>> mc = MarkovChain()
            >>> mc.train("This is a simple example. This example is simple.")
            >>> print(mc.graph)
            defaultdict(<class 'list'>, {'This': ['is'], 'is': ['a', 'simple.'], 'a': ['simple'], 
                                        'simple': ['example.'], 'example.': ['This'], 'example': ['is']})
        """
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            if (len(tokens) - 1) == i:
                break
            self.graph[token].append(tokens[i + 1])
               

    def generate(self, prompt, length=10):
        """
        Generates a sequence of text based on the trained Markov Chain and a given prompt.

        This method uses the Markov Chain graph to generate a sequence of words. It starts with the 
        last word in the given prompt and iteratively selects the next word based on the possible 
        transitions in the graph. The process continues until the desired sequence length is reached.

        Args:
            prompt (str): The initial text to start the generation. The last word of the prompt is 
                        used as the starting point for the generation.
            length (int): The number of words to generate in the sequence (default is 10).

        Returns:
            str: A string containing the generated sequence of text.

        Notes:
            - The `_tokenize` method is used to extract the last word from the prompt.
            - The `random.choice` method is used to randomly select the next word from the list of 
            possible transitions for the current word.
            - If no transitions are available for the current word, the generation process skips to 
            the next iteration without adding a new word.
            - The generated text is appended to the initial prompt to form the final output.

        Example:
            >>> mc = MarkovChain()
            >>> mc.train("This is a simple example. This example is simple.")
            >>> print(mc.generate("This", length=5))
            This is a simple example
        """
        # Get the last token from the prompt
        current = self._tokenize(prompt)[-1]
        # Initialize the output with the prompt
        output = prompt
        for i in range(length):
            # Look up the options in the graph dictionary
            options = self.graph.get(current, [])
            if not options:
                continue
            # Use random.choice to pick the next word
            current = random.choice(options)
            # Add the selected word to the output
            output += f" {current}"
        
        return output
    

# Usage example
"""
The following code will read the csv_datasets/markov_chain_impression_dataset.csv file.
It will train the Markov chain model based on the collection of the CSV rows
"""

def read_csv(file_path):
    """
    Reads the content of a CSV file and returns it as a string.

    This function takes a file path as input, opens the file in read mode, and reads its content.
    It is designed to work with text-based CSV files.

    Args:
        file_path (str): The relative or absolute path to the CSV file.

    Returns:
        str: The content of the file as a single string.

    Notes:
        - Ensure that the file exists at the specified path to avoid a `FileNotFoundError`.
        - The file is opened in read mode ('r'), so it must be a readable text file.

    Example:
        >>> content = read_csv("../../../csv_datasets/markov_chain_impression_dataset.csv")
        >>> print(content)
        "Row 1 content\nRow 2 content\n..."
    """
    print("Input file path", file_path)
    with open(file_path, 'r') as file:
        return file.read()
    
text = read_csv("/Users/apple/Documents/Projects/Samhail/models/llms/markovchain_text_generator.py")

chain = MarkovChain()
chain.train(text)
sample_prompt = input("Enter a prompt: ")
print(chain.generate(sample_prompt))

result = chain.generate(sample_prompt)