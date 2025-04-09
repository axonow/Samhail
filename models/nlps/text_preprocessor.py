"""
Text Preprocessor Module

This module provides a comprehensive set of tools for preprocessing text data. It includes functions for
cleaning, normalizing, augmenting, and analyzing text, making it suitable for a wide range of natural
language processing (NLP) tasks.

### Features:
1. **Basic Text Cleaning**:
    - Lowercasing
    - Tokenization
    - Removing punctuation
    - Removing stopwords
    - Removing numbers
    - Removing special characters
    - Removing HTML tags
    - Removing URLs
    - Handling whitespace tokens
    - Handling dates and times

2. **Text Normalization**:
    - Handling contractions
    - Normalizing text for social media
    - Handling emojis and emoticons
    - Text encoding
    - Normalizing accents and special characters

3. **Text Augmentation**:
    - Synonym replacement for text augmentation
    - Handling abbreviations
    - Handling rare words

4. **Text Analysis**:
    - Stemming
    - Lemmatization
    - Parts of speech (POS) tagging
    - Named entity recognition (NER)
    - Dependency parsing
    - Text chunking
    - Language detection

5. **Vectorization**:
    - CountVectorizer
    - TfidfVectorizer

6. **Handling Missing Data**:
    - Replacing missing or null values with default values.

---

### Dependencies:
- `re`: For regular expressions.
- `string`: For handling punctuation.
- `spacy`: For advanced NLP tasks like dependency parsing and named entity recognition.
- `nltk`: For tokenization, stemming, lemmatization, POS tagging, and text chunking.
- `unicodedata`: For normalizing text by removing accents.
- `bs4 (BeautifulSoup)`: For removing HTML tags.
- `sklearn.feature_extraction.text`: For vectorization (CountVectorizer and TfidfVectorizer).
- `langdetect`: For detecting the language of the text.
- `emoji`: For handling emojis and emoticons.

---

### Class: `TextPreprocessor`
The `TextPreprocessor` class provides methods for preprocessing text data. It is initialized with a specified
language (default is English) for stopwords and lemmatization.

#### Methods:
1. **Initialization**:
    - `__init__(self, language='english')`: Initializes the preprocessor with the specified language.

2. **Basic Cleaning**:
    - `to_lowercase(self, text)`: Converts text to lowercase.
    - `tokenize(self, text)`: Tokenizes text into words.
    - `remove_punctuation(self, text)`: Removes punctuation from text.
    - `remove_stopwords(self, tokens)`: Removes stopwords from a list of tokens.
    - `remove_numbers(self, text)`: Removes numbers from text.
    - `remove_special_characters(self, text)`: Removes special characters from text.
    - `remove_html_tags(self, text)`: Removes HTML tags from text.
    - `handle_urls(self, text)`: Removes URLs from text.
    - `handle_whitespace(self, text)`: Removes extra whitespace from text.
    - `handle_dates_and_times(self, text)`: Removes dates and times from text.

3. **Text Normalization**:
    - `handle_contractions(self, text)`: Expands contractions in text.
    - `normalize(self, text)`: Normalizes text by removing accents and converting to ASCII.
    - `normalize_social_media_text(self, text)`: Normalizes text from social media by handling hashtags, mentions, and repeated characters.
    - `handle_emojis(self, text)`: Converts emojis to their textual representation.

4. **Text Augmentation**:
    - `text_augmentation(self, text)`: Performs text augmentation by randomly replacing words with synonyms.
    - `handle_abbreviations(self, text)`: Expands common abbreviations in text.
    - `handle_rare_words(self, tokens, threshold=1)`: Handles rare words by replacing them with a placeholder or removing them.

5. **Text Analysis**:
    - `stem(self, tokens)`: Applies stemming to a list of tokens.
    - `lemmatize(self, tokens)`: Applies lemmatization to a list of tokens.
    - `pos_tagging(self, tokens)`: Performs parts of speech tagging on a list of tokens.
    - `named_entity_recognition(self, text)`: Performs named entity recognition on text.
    - `dependency_parsing(self, text)`: Performs dependency parsing on the input text.
    - `text_chunking(self, text)`: Performs text chunking to identify noun phrases and verb phrases.
    - `detect_language(self, text)`: Detects the language of the text.

6. **Vectorization**:
    - `vectorize(self, corpus, method='tfidf')`: Vectorizes a corpus using CountVectorizer or TfidfVectorizer.

7. **Handling Missing Data**:
    - `handle_missing_data(self, text)`: Handles missing data by replacing None or NaN with an empty string.

8. **Utility Functions**:
    - `encode_text(self, text, encoding='utf-8')`: Encodes text to the specified encoding.
    - `get_synonyms(self, word)`: Retrieves synonyms for a given word using WordNet.

---

### Example Usage:

```python
from text_preprocessor import TextPreprocessor

# Initialize the preprocessor
preprocessor = TextPreprocessor()

# Sample text
text = "OMG!!! I soooo love this!!! #amazing @user Visit https://example.com on 12/12/2022 at 10:00 AM 😊"

# 1. Lowercasing
lowercased_text = preprocessor.to_lowercase(text)
print("Lowercased Text:", lowercased_text)

# 2. Tokenization
tokens = preprocessor.tokenize(lowercased_text)
print("Tokens:", tokens)

# 3. Removing Punctuation
no_punctuation_text = preprocessor.remove_punctuation(lowercased_text)
print("Text without Punctuation:", no_punctuation_text)

# 4. Removing Stopwords
filtered_tokens = preprocessor.remove_stopwords(tokens)
print("Tokens without Stopwords:", filtered_tokens)

# 5. Stemming
stemmed_tokens = preprocessor.stem(filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)

# 6. Lemmatization
lemmatized_tokens = preprocessor.lemmatize(filtered_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)

# 7. Removing Numbers
no_numbers_text = preprocessor.remove_numbers(lowercased_text)
print("Text without Numbers:", no_numbers_text)

# 8. Handling Contractions
expanded_text = preprocessor.handle_contractions("I can't believe it's not butter!")
print("Expanded Text:", expanded_text)

# 9. Removing Special Characters
cleaned_text = preprocessor.remove_special_characters(lowercased_text)
print("Text without Special Characters:", cleaned_text)

# 10. Parts of Speech Tagging
pos_tags = preprocessor.pos_tagging(filtered_tokens)
print("POS Tags:", pos_tags)

# 11. Named Entity Recognition
entities = preprocessor.named_entity_recognition(lowercased_text)
print("Named Entities:", entities)

# 12. Vectorization
corpus = ["This is a sample text.", "Text preprocessing is important."]
vectorized_corpus = preprocessor.vectorize(corpus, method='tfidf')
print("Vectorized Corpus (TF-IDF):", vectorized_corpus.toarray())

# 13. Handling Missing Data
handled_text = preprocessor.handle_missing_data(None)
print("Handled Missing Data:", handled_text)

# 14. Normalization
normalized_text = preprocessor.normalize("Café")
print("Normalized Text:", normalized_text)

# 15. Handling Emojis
emoji_text = preprocessor.handle_emojis("I love Python! 😊")
print("Text with Emojis Handled:", emoji_text)

# 16. Removing HTML Tags
html_text = preprocessor.remove_html_tags("<p>This is a paragraph.</p>")
print("Text without HTML Tags:", html_text)

# 17. Handling URLs
no_urls_text = preprocessor.handle_urls("Visit https://example.com for more info.")
print("Text without URLs:", no_urls_text)

# 18. Sentence Segmentation
sentences = preprocessor.sentence_segmentation("This is the first sentence. Here is another one.")
print("Segmented Sentences:", sentences)

# 19. Handling Abbreviations
expanded_abbreviations = preprocessor.handle_abbreviations("u r amazing btw!")
print("Expanded Abbreviations:", expanded_abbreviations)

# 20. Language Detection
language = preprocessor.detect_language("Bonjour tout le monde!")
print("Detected Language:", language)

# 21. Text Encoding
encoded_text = preprocessor.encode_text("This is a test.")
print("Encoded Text:", encoded_text)

# 22. Handling Whitespace
cleaned_whitespace_text = preprocessor.handle_whitespace("   This   is   a   test.   ")
print("Text with Whitespace Handled:", cleaned_whitespace_text)

# 23. Handling Dates and Times
no_dates_text = preprocessor.handle_dates_and_times("The event is on 12/12/2022 at 10:00 AM.")
print("Text without Dates and Times:", no_dates_text)

# 24. Text Augmentation
augmented_text = preprocessor.text_augmentation("The weather is good today.")
print("Augmented Text:", augmented_text)

# 25. Handling Negations
negation_text = preprocessor.handle_negations("I am not happy with this product.")
print("Negation Handled:", negation_text)

# 26. Dependency Parsing
dependencies = preprocessor.dependency_parsing("The quick brown fox jumps over the lazy dog.")
print("Dependency Parsing:", dependencies)

# 27. Handling Rare Words
tokens_with_rare_words = preprocessor.tokenize("This is a rare word example with rare rare words.")
handled_rare_words = preprocessor.handle_rare_words(tokens_with_rare_words, threshold=2)
print("Tokens with Rare Words Handled:", handled_rare_words)

# 28. Text Chunking
chunks = preprocessor.text_chunking("The quick brown fox jumps over the lazy dog.")
print("Text Chunks:", chunks)

# 29. Synonyms
synonyms = preprocessor.get_synonyms("happy")
print("Synonyms for 'happy':", synonyms)

# 30. Social Media Text Normalization
social_media_text = preprocessor.normalize_social_media_text("OMG!!! I soooo love this!!! #amazing @user")
print("Normalized Social Media Text:", social_media_text)
"""

import re
import string
import random
import spacy
import nltk
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from langdetect import detect
from emoji import demojize

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("omw-1.4")  # For WordNet synonyms


class TextPreprocessor:
    def __init__(self, language="english"):
        """
        Initializes the TextPreprocessor with the specified language for stopwords and lemmatization.

        Args:
            language (str): The language to use for stopwords and lemmatization (default is 'english').
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm")

    def to_lowercase(self, text):
        """Converts text to lowercase."""
        return text.lower()

    def tokenize(self, text):
        """Tokenizes text into words."""
        return word_tokenize(text)

    def remove_punctuation(self, text):
        """Removes punctuation from text."""
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_stopwords(self, tokens):
        """Removes stopwords from a list of tokens."""
        return [word for word in tokens if word not in self.stop_words]

    def stem(self, tokens):
        """Applies stemming to a list of tokens."""
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize(self, tokens):
        """Applies lemmatization to a list of tokens."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def remove_numbers(self, text):
        """Removes numbers from text."""
        return re.sub(r"\d+", "", text)

    def handle_contractions(self, text):
        """Expands contractions in text."""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am",
        }
        pattern = re.compile(r"\b(" + "|".join(contractions.keys()) + r")\b")
        return pattern.sub(lambda x: contractions[x.group()], text)

    def remove_special_characters(self, text):
        """Removes special characters from text."""
        return re.sub(r"[^a-zA-Z\s]", "", text)

    def pos_tagging(self, tokens):
        """Performs parts of speech tagging on a list of tokens."""
        return nltk.pos_tag(tokens)

    def named_entity_recognition(self, text):
        """Performs named entity recognition on text."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def vectorize(self, corpus, method="tfidf"):
        """
        Vectorizes a corpus using CountVectorizer or TfidfVectorizer.

        Args:
            corpus (list of str): The text corpus to vectorize.
            method (str): The vectorization method ('count' or 'tfidf').

        Returns:
            sparse matrix: The vectorized representation of the corpus.
        """
        if method == "count":
            vectorizer = CountVectorizer()
        elif method == "tfidf":
            vectorizer = TfidfVectorizer()
        else:
            raise ValueError("Invalid method. Choose 'count' or 'tfidf'.")
        return vectorizer.fit_transform(corpus)

    def handle_missing_data(self, text):
        """Handles missing data by replacing None or NaN with an empty string."""
        return text if text else ""

    def normalize(self, text):
        """Normalizes text by removing accents and converting to ASCII."""
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )

    def handle_emojis(self, text):
        """Converts emojis to their textual representation."""
        return demojize(text)

    def remove_html_tags(self, text):
        """Removes HTML tags from text."""
        return BeautifulSoup(text, "html.parser").get_text()

    def handle_urls(self, text):
        """Removes URLs from text."""
        return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    def sentence_segmentation(self, text):
        """Segments text into sentences."""
        return sent_tokenize(text)

    def handle_abbreviations(self, text):
        """Expands common abbreviations in text."""
        abbreviations = {
            "u": "you",
            "r": "are",
            "btw": "by the way",
            "idk": "I do not know",
            "imo": "in my opinion",
            "omg": "oh my god",
            "lol": "laughing out loud",
        }
        pattern = re.compile(r"\b(" + "|".join(abbreviations.keys()) + r")\b")
        return pattern.sub(lambda x: abbreviations[x.group()], text)

    def detect_language(self, text):
        """
        Detects the language of the text.

        Args:
            text (str): The input text.

        Returns:
            str: The detected language code (e.g., 'en', 'fr').

        Raises:
            ValueError: If the text is empty or language detection fails.
        """
        if not text.strip():
            raise ValueError("Input text is empty.")
        try:
            return detect(text)
        except Exception:
            return "unknown"  # Fallback for cases where detection fails

    def encode_text(self, text, encoding="utf-8"):
        """Encodes text to the specified encoding."""
        return text.encode(encoding)

    def handle_whitespace(self, text):
        """Removes extra whitespace from text."""
        return " ".join(text.split())

    def handle_dates_and_times(self, text):
        """Removes dates and times from text."""
        return re.sub(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b",
            "",
            text,
        )

    def text_augmentation(self, text):
        """
        Performs text augmentation by randomly replacing words with synonyms.

        Args:
            text (str): The input text to augment.

        Returns:
            str: The augmented text.
        """
        tokens = self.tokenize(text)
        augmented_tokens = []
        for token in tokens:
            synonyms = self.get_synonyms(token)
            if synonyms:
                augmented_tokens.append(random.choice(synonyms))
            else:
                augmented_tokens.append(token)
        return " ".join(augmented_tokens)

    def handle_negations(self, text):
        """
        Handles negations by converting phrases like "not good" to "bad".

        Args:
            text (str): The input text.

        Returns:
            str: The text with negations handled.
        """
        negations = {
            "not good": "bad",
            "not happy": "unhappy",
            "not bad": "good",
            "not like": "dislike",
            "not love": "hate",
        }
        pattern = re.compile(r"\b(" + "|".join(negations.keys()) + r")\b")
        return pattern.sub(lambda x: negations[x.group()], text)

    def dependency_parsing(self, text):
        """
        Performs dependency parsing on the input text.

        Args:
            text (str): The input text.

        Returns:
            list: A list of tuples representing dependency relations (word, dependency, head).
        """
        doc = self.nlp(text)
        return [(token.text, token.dep_, token.head.text) for token in doc]

    def handle_rare_words(self, tokens, threshold=1):
        """
        Handles rare words by replacing them with a placeholder or removing them.

        Args:
            tokens (list): A list of tokens.
            threshold (int): The minimum frequency for a word to be considered common.

        Returns:
            list: A list of tokens with rare words handled.
        """
        word_freq = nltk.FreqDist(tokens)
        return [token if word_freq[token] > threshold else "<RARE>" for token in tokens]

    def text_chunking(self, text):
        """
        Performs text chunking to identify noun phrases and verb phrases.

        Args:
            text (str): The input text.

        Returns:
            list: A list of chunks (noun phrases, verb phrases, etc.).
        """
        tokens = self.tokenize(text)
        pos_tags = self.pos_tagging(tokens)
        grammar = r"""
            NP: {<DT>?<JJ>*<NN>}   # Noun phrase
            VP: {<VB.*><NP|PP>*}   # Verb phrase
        """
        chunk_parser = nltk.RegexpParser(grammar)
        tree = chunk_parser.parse(pos_tags)
        return [
            " ".join(leaf[0] for leaf in subtree.leaves())
            for subtree in tree.subtrees()
            if subtree.label() in ["NP", "VP"]
        ]

    def get_synonyms(self, word):
        """
        Retrieves synonyms for a given word using WordNet.

        Args:
            word (str): The input word.

        Returns:
            list: A list of synonyms for the word.
        """
        from nltk.corpus import wordnet

        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return list(set(synonyms))

    def normalize_social_media_text(self, text):
        """
        Normalizes text from social media by handling hashtags, mentions, and repeated characters.

        Args:
            text (str): The input text.

        Returns:
            str: The normalized text.
        """
        # Remove mentions
        text = re.sub(r"@\w+", "", text)
        # Remove hashtags
        text = re.sub(r"#\w+", "", text)
        # Replace repeated characters (e.g., "soooo" -> "so")
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        return text.strip()
