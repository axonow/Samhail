"""
MarkovChain Text Generator

This script implements a Markov Chain-based text generator. It allows training on a given text corpus
from CSV files and generating new text sequences based on the learned Markov Chain model.

Classes:
    - MarkovChain: Represents the Markov Chain model for text generation.

Methods:
    - __init__: Initializes the Markov Chain with an empty graph.
    - _tokenize: Tokenizes input text by removing punctuation, numbers, and splitting into words.
    - _train: Trains the Markov Chain by building a graph of word transitions from the input text.
    - _read_pd_csv: Reads a CSV file and converts the first column into a single string.
    - _generate: Generates a sequence of text based on the trained Markov Chain and a given prompt.
    - _train_model: Trains the Markov Chain model using text data from multiple CSV files.

Functions:
    - predict_next: A standalone function that trains the Markov Chain model and generates text based on user input.

Constants:
    - CSV_FILE_PATHS: A list of file paths to the CSV files used for training the model.

Usage:
    1. Create an instance of the `MarkovChain` class.
    2. Train the model using the `_train_model` method with the provided CSV files.
    3. Generate new text using the `_generate` method with a prompt and desired length.

Example:
    >>> model = MarkovChain()
    >>> trained_model = model._train_model()
    >>> prompt = "This is"
    >>> print(trained_model._generate(prompt, length=10))

Dependencies:
    - random: Used for randomly selecting the next word during text generation.
    - string.punctuation: Used for removing punctuation during tokenization.
    - collections.defaultdict: Used for storing the Markov Chain graph as a dictionary of lists.
    - pandas: Used for reading and processing CSV files.

Notes:
    - The `_tokenize` method removes punctuation and numbers, converts newlines to spaces, and splits text into words.
    - The `_train` method builds a graph where each word points to a list of possible next words.
    - The `_generate` method uses the graph to create a sequence of words, starting from the last word in the prompt.
    - The `_train_model` method combines text from multiple CSV files and trains the model.

Limitations:
    - The model assumes that the input text is well-formed and does not handle edge cases like empty input gracefully.
    - The generated text may not always be coherent, as it relies purely on word transitions without considering grammar or context.
    - The `_generate` method assumes that the graph has been trained before generating text.

"""

import os

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

# The `pandas` library is imported as `pd` to provide data manipulation and analysis tools.
# It is used in this script to read CSV files and handle data in a structured format (DataFrame).
# Specifically, `pandas.read_csv` is used to load CSV data, and `pandas.DataFrame` is used to create and manipulate tabular data.
import pandas as pd


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
        Tokenizes input text by removing punctuation, numeric characters, and splitting it into words.

        This method processes the input text to prepare it for further analysis or training. It removes
        all punctuation and numeric characters, leaving only alphabetic characters and spaces. The cleaned
        text is then split into individual words (tokens), and any empty strings resulting from the split
        operation are filtered out.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of tokens (words) extracted from the input text.

        Example:
            >>> markov_chain = MarkovChain()
            >>> markov_chain._tokenize("Hello, world! 123")
            ['Hello', 'world']

        Notes:
            - Punctuation and numeric characters are removed using a generator expression.
            - The `split()` method is used to split the cleaned text into words.
            - Empty strings are filtered out using a list comprehension.

        Limitations:
            - This method assumes that the input text is a single string.
            - It does not handle special cases like non-ASCII characters or text with mixed encodings.
        """
        # Remove punctuation and numeric characters
        text = "".join(char for char in text if char.isalpha()
                       or char.isspace())
        # Split into words and filter out empty strings
        tokens = [word for word in text.split() if word]
        return tokens

    def _train(self, text):
        """
        Trains the Markov Chain model by building a graph of word transitions from the input text.

        This method processes the input text to construct a graph where each word points to a list of possible
        next words based on the input text. The graph is stored as a `defaultdict` of lists.

        Args:
            text (str): The input text used to train the Markov Chain model.

        Returns:
            None

        Example:
            >>> markov_chain = MarkovChain()
            >>> markov_chain._train("Hello world. Hello again.")
            >>> print(markov_chain.graph)
            {'Hello': ['world', 'again'], 'world': ['Hello']}

        Notes:
            - The input text is preprocessed to ensure proper spacing between words and sentences.
            - The `_tokenize` method is used to split the text into tokens (words).
            - For each pair of consecutive tokens, the first token is added as a key in the graph, and the second
            token is appended to the list of possible next words for that key.

        Limitations:
            - This method assumes that the input text is well-formed and does not handle edge cases like empty input.
            - If the input text contains only one word, the graph will have that word as a key with an empty list as its value.
        """
        # Ensure proper spacing between words and sentences
        text = text.replace(".", ". ").replace(",", ", ").strip()
        tokens = self._tokenize(text)
        for i in range(len(tokens) - 1):
            self.graph[tokens[i]].append(tokens[i + 1])

    def _read_pd_csv(self, csv_file_path, header=None):
        """
        Reads a CSV file into a pandas DataFrame and converts the first column to a single string.

        This method reads the specified CSV file, extracts the first column, and concatenates its rows
        into a single string, with each row separated by a newline character.

        Args:
            csv_file_path (str): The path to the CSV file.
            header (int or None): Row number to use as the column names, or None if the CSV files have no headers.

        Returns:
            str: A string containing all rows of the first column, separated by newlines.

        Raises:
            Exception: If there is an error reading or processing the CSV file.

        Example:
            >>> markov_chain = MarkovChain()
            >>> text = markov_chain._read_pd_csv("example.csv")
            >>> print(text)
            "Hello world\nThis is a test"

        Notes:
            - The method uses `pandas.read_csv` to load the CSV file into a DataFrame.
            - The first column of the DataFrame is converted to a string, with rows joined by newline characters.
            - If the CSV file is empty or does not contain a valid first column, the method will return an empty string.

        Limitations:
            - The method assumes that the CSV file is well-formed and encoded in UTF-8.
            - If the CSV file contains multiple columns, only the first column is processed.
        """
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path, encoding="UTF-8", header=header)

            # Convert the first column to a string with rows separated by "\n"
            # First column is the comment, the second being sentiment
            first_column_as_string = "\n".join(df.iloc[:, 0].astype(str))
            return first_column_as_string
        except Exception as e:
            print(f"Error processing CSV file at {csv_file_path}: {e}")
            raise

    # Define the base directory as the root of the project
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Define constants for CSV file paths
    CSV_FILE_PATHS = [
        os.path.join(BASE_DIR, "csv_datasets",
                     "markov_chain_impression_dataset.csv"),
        os.path.join(BASE_DIR, "csv_datasets",
                     "reddit_social_media_comments.csv"),
        os.path.join(BASE_DIR, "csv_datasets",
                     "twitter_social_media_comments.csv"),
        os.path.join(BASE_DIR, "csv_datasets", "imdb_movie_reviews.csv"),
        os.path.join(BASE_DIR, "csv_datasets", "dcat_train_data.csv"),
    ]

    def _generate(self, prompt, length=10):
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

    def _train_model(self, csv_file_paths=CSV_FILE_PATHS, csv_header=None):
        """
        Trains the Markov Chain model using text data from multiple CSV files.

        This method reads text data from the specified CSV file paths, processes the first column
        of each file into a single string, and trains a Markov Chain model using the combined text.

        Args:
            csv_file_paths (list of str): A list of file paths to the CSV files containing the training data.
            csv_header (int or None): Row number to use as the column names, or None if the CSV files have no headers.

        Returns:
            MarkovChain: An instance of the MarkovChain class trained on the combined text data.

        Raises:
            ValueError: If no CSV file paths are provided.
            Exception: If any error occurs while reading or processing the CSV files.

        Notes:
            - The `_read_pd_csv` function is used to read and process each CSV file.
            - The first column of each CSV file is converted into a string, and all strings are concatenated.
            - The concatenated text is used to train the Markov Chain model.
        """
        if not csv_file_paths:
            raise ValueError("No CSV file paths provided.")
        else:
            text = ""
            for csv_file_path in csv_file_paths:
                text += self._read_pd_csv(csv_file_path, header=csv_header)
            self._train(text)
            return self


def predict_next(user_input="A cat"):
    """
    Trains the Markov Chain model and generates text based on user input.
    """
    model = MarkovChain()
    trained_model = model._train_model()
    if user_input is None:
        user_input = input("Enter a prompt: ")
    print(trained_model._generate(user_input, length=10))


# Predict next word based on user input
predict_next()
