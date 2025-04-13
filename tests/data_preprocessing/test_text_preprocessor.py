import pytest
from data_preprocessing.text_preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    """Fixture to initialize the TextPreprocessor."""
    return TextPreprocessor()


def test_to_lowercase(preprocessor):
    text = "HELLO WORLD"
    assert preprocessor.to_lowercase(text) == "hello world"


def test_tokenize(preprocessor):
    text = "This is a test."
    assert preprocessor.tokenize(text) == ["This", "is", "a", "test", "."]


def test_remove_punctuation(preprocessor):
    text = "Hello, world!"
    assert preprocessor.remove_punctuation(text) == "Hello world"


def test_remove_stopwords(preprocessor):
    tokens = ["this", "is", "a", "test"]
    assert preprocessor.remove_stopwords(tokens) == ["test"]


def test_stem(preprocessor):
    tokens = ["running", "jumps", "easily"]
    assert preprocessor.stem(tokens) == ["run", "jump", "easili"]


def test_lemmatize(preprocessor):
    tokens = ["running", "jumps", "easily"]
    assert preprocessor.lemmatize(tokens) == ["running", "jump", "easily"]


def test_remove_numbers(preprocessor):
    text = "There are 123 apples."
    assert preprocessor.remove_numbers(text) == "There are  apples."


def test_handle_contractions(preprocessor):
    text = "I can't believe it's true."
    assert preprocessor.handle_contractions(text) == "I cannot believe it is true."


def test_remove_special_characters(preprocessor):
    text = "Hello @world! #amazing"
    assert preprocessor.remove_special_characters(text) == "Hello world amazing"


def test_pos_tagging(preprocessor):
    tokens = ["This", "is", "a", "test"]
    assert isinstance(preprocessor.pos_tagging(tokens), list)


def test_named_entity_recognition(preprocessor):
    text = "Barack Obama was the 44th President of the United States."
    result = preprocessor.named_entity_recognition(text)
    assert ("Barack Obama", "PERSON") in result


def test_vectorize(preprocessor):
    corpus = ["This is a test.", "Another test."]
    result = preprocessor.vectorize(corpus, method="tfidf")
    assert result.shape[0] == 2

    # Empty corpus
    corpus = []
    with pytest.raises(ValueError):  # Vectorizers might raise an error for empty input
        preprocessor.vectorize(corpus, method="tfidf")

    # Single document
    corpus = ["This is a single document."]
    result = preprocessor.vectorize(corpus, method="tfidf")
    assert result.shape[0] == 1


def test_handle_missing_data(preprocessor):
    text = None
    assert preprocessor.handle_missing_data(text) == ""


def test_normalize(preprocessor):
    text = "Café"
    assert preprocessor.normalize(text) == "Cafe"


def test_handle_emojis(preprocessor):
    text = "I love Python! 😊"
    assert (
        preprocessor.handle_emojis(text)
        == "I love Python! :smiling_face_with_smiling_eyes:"
    )


def test_remove_html_tags(preprocessor):
    text = "<p>This is a paragraph.</p>"
    assert preprocessor.remove_html_tags(text) == "This is a paragraph."


def test_handle_urls(preprocessor):
    text = "Visit https://example.com for more info."
    assert preprocessor.handle_urls(text) == "Visit  for more info."


def test_sentence_segmentation(preprocessor):
    text = "This is the first sentence. Here is another one."
    assert preprocessor.sentence_segmentation(text) == [
        "This is the first sentence.",
        "Here is another one.",
    ]


def test_handle_abbreviations(preprocessor):
    text = "u r amazing btw!"
    assert preprocessor.handle_abbreviations(text) == "you are amazing by the way!"


def test_detect_language(preprocessor):
    text = "Bonjour tout le monde!"
    assert preprocessor.detect_language(text) == "fr"

    # Empty text
    text = ""
    # Language detection might raise an exception for empty input
    with pytest.raises(Exception):
        preprocessor.detect_language(text)

    # # Ambiguous text
    # text = "Hello Bonjour"
    # assert preprocessor.detect_language(text) in ["en", "fr"]


def test_encode_text(preprocessor):
    text = "This is a test."
    assert preprocessor.encode_text(text) == b"This is a test."


def test_handle_whitespace(preprocessor):
    text = "   This   is   a   test.   "
    assert preprocessor.handle_whitespace(text) == "This is a test."


def test_handle_dates_and_times(preprocessor):
    text = "The event is on 12/12/2022 at 10:00 AM."
    assert preprocessor.handle_dates_and_times(text) == "The event is on  at ."


def test_text_augmentation(preprocessor):
    text = "The weather is good."
    assert isinstance(preprocessor.text_augmentation(text), str)

    # Empty text
    text = ""
    assert preprocessor.text_augmentation(text) == ""

    # No synonyms available
    text = "qwerty"
    assert isinstance(preprocessor.text_augmentation(text), str)


def test_handle_negations(preprocessor):
    text = "I am not happy."
    assert preprocessor.handle_negations(text) == "I am unhappy."


def test_dependency_parsing(preprocessor):
    text = "The quick brown fox jumps over the lazy dog."
    assert isinstance(preprocessor.dependency_parsing(text), list)

    # Empty text
    text = ""
    assert preprocessor.dependency_parsing(text) == []

    # Single word
    text = "Hello"
    result = preprocessor.dependency_parsing(text)
    assert len(result) == 1
    assert result[0][0] == "Hello"


def test_handle_rare_words(preprocessor):
    tokens = ["rare", "word", "example", "rare"]
    result = preprocessor.handle_rare_words(tokens, threshold=1)
    assert result == ["rare", "<RARE>", "<RARE>", "rare"]

    # All tokens are rare
    tokens = ["unique", "words", "here"]
    result = preprocessor.handle_rare_words(tokens, threshold=1)
    assert result == ["<RARE>", "<RARE>", "<RARE>"]

    # No tokens are rare
    tokens = ["common", "common", "common"]
    result = preprocessor.handle_rare_words(tokens, threshold=1)
    assert result == ["common", "common", "common"]

    # Threshold is 0
    tokens = ["rare", "word", "example", "rare"]
    result = preprocessor.handle_rare_words(tokens, threshold=0)
    assert result == ["rare", "word", "example", "rare"]


def test_text_chunking(preprocessor):
    text = "The quick brown fox jumps over the lazy dog."
    assert isinstance(preprocessor.text_chunking(text), list)


def test_get_synonyms(preprocessor):
    word = "happy"
    assert isinstance(preprocessor.get_synonyms(word), list)


def test_normalize_social_media_text(preprocessor):
    text = "OMG!!! I soooo love this!!! #amazing @user"
    assert preprocessor.normalize_social_media_text(text) == "OMG! I so love this!"

    # No hashtags or mentions
    text = "I love this!"
    assert preprocessor.normalize_social_media_text(text) == "I love this!"

    # Excessive repeated characters
    text = "Soooooo happy!!!"
    assert preprocessor.normalize_social_media_text(text) == "So happy!"
