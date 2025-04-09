import pytest
import pandas as pd
from models.nlps.markovchain_text_generator import MarkovChain


@pytest.fixture
def markov_chain():
    """Fixture to initialize the MarkovChain instance."""
    return MarkovChain()


def test_init(markov_chain):
    assert isinstance(markov_chain.graph, dict)
    assert len(markov_chain.graph) == 0


def test_tokenize(markov_chain):
    text = "Hello, world! 123"
    tokens = markov_chain._tokenize(text)
    assert tokens == ["Hello", "world"]

    # Empty string
    text = ""
    tokens = markov_chain._tokenize(text)
    assert tokens == []

    # String with only punctuation
    text = "!!!"
    tokens = markov_chain._tokenize(text)
    assert tokens == []

    # String with special characters
    text = "Hello @world!"
    tokens = markov_chain._tokenize(text)
    assert tokens == ["Hello", "world"]


def test_train(markov_chain):
    text = "Hello world. Hello again."
    markov_chain._train(text)
    assert "Hello" in markov_chain.graph
    assert markov_chain.graph["Hello"] == ["world", "again"]
    assert markov_chain.graph["world"] == ["Hello"]

    # # Empty input
    # text = ""
    # markov_chain._train(text)
    # assert len(markov_chain.graph) == 0

    # # Input with one word
    # text = "Hello"
    # markov_chain._train(text)
    # assert len(markov_chain.graph) == 1
    # assert markov_chain.graph["Hello"] == []


def test_read_pd_csv(mocker, markov_chain):
    # Mock pandas.read_csv to simulate reading a CSV file
    mock_csv_data = {"column1": ["Hello world", "This is a test"]}
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame(mock_csv_data))

    result = markov_chain._read_pd_csv("dummy_path.csv")
    assert result == "Hello world\nThis is a test"

    # Empty CSV file
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame(columns=["column1"]))
    result = markov_chain._read_pd_csv("dummy_path.csv")
    assert result == ""

    # CSV file with multiple columns
    mock_csv_data = {"column1": ["Hello world"], "column2": ["Extra data"]}
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame(mock_csv_data))
    result = markov_chain._read_pd_csv("dummy_path.csv")
    assert result == "Hello world"


def test_generate(markov_chain):
    # Train the Markov Chain with sample text
    text = "Hello world. Hello again."
    markov_chain._train(text)

    # Generate text with a valid prompt
    prompt = "Hello"
    generated_text = markov_chain._generate(prompt, length=5)
    assert generated_text.startswith("Hello")
    # Ensure at least one word is generated
    assert len(generated_text.split()) >= 1

    # Generate text with an invalid prompt
    prompt = "Invalid"
    generated_text = markov_chain._generate(prompt, length=5)
    assert generated_text == "Invalid"  # No transitions available


def test_train_model(mocker, markov_chain):
    # Mock the _read_pd_csv method to return sample text
    mocker.patch.object(
        markov_chain, "_read_pd_csv", return_value="Hello world. Hello again."
    )

    # Train the model with mock CSV paths
    trained_model = markov_chain._train_model(csv_file_paths=["dummy_path.csv"])
    assert "Hello" in trained_model.graph
    assert trained_model.graph["Hello"] == ["world", "again"]

    # # Mock the _read_pd_csv method to return different text for each file
    # mocker.patch.object(markov_chain, "_read_pd_csv", side_effect=["Hello world.", "Hello again."])

    # # Train the model with multiple CSV paths
    # trained_model = markov_chain._train_model(csv_file_paths=["file1.csv", "file2.csv"])
    # assert "Hello" in trained_model.graph
    # assert trained_model.graph["Hello"] == ["world", "again"]


def test_predict_next(mocker):
    # Mock the input function to simulate user input
    # Simulate two inputs: "Hello" and an empty string
    mocker.patch("builtins.input", side_effect=["Hello", ""])

    # Mock the MarkovChain methods
    mock_chain = MarkovChain()
    mocker.patch.object(mock_chain, "_train_model", return_value=mock_chain)
    mocker.patch.object(mock_chain, "_generate", return_value="Hello world again")

    # Mock the MarkovChain class to return the mocked instance
    mocker.patch(
        "models.nlps.markovchain_text_generator.MarkovChain", return_value=mock_chain
    )

    # Import and call the predict_next function
    from models.nlps.markovchain_text_generator import predict_next

    predict_next()  # Simulate input "Hello"
    predict_next()  # Simulate empty input
