from collections import defaultdict
import random
import os
import json
import yaml
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
import logging
import sys

# Import the TextPreprocessor class
try:
    from data_preprocessing.text_preprocessor import TextPreprocessor
    TEXT_PREPROCESSOR_AVAILABLE = True
    logger.info("TextPreprocessor successfully imported")
except ImportError as e:
    TEXT_PREPROCESSOR_AVAILABLE = False
    print(f"Import error: {e}")  # Print directly to see the exact import error
    # The path might be incorrect or the module not installed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class MarkovChain:
    """
    A hybrid implementation of a Markov Chain for text generation.
    
    This class implements an n-gram Markov model that can use either in-memory
    storage or PostgreSQL database storage based on data size and configuration.
    """

    def __init__(self, n_gram=1, memory_threshold=10000, db_config=None, environment="development"):
        """
        Initializes the Markov Chain with flexible storage options.
        
        Args:
            n_gram (int): Number of words to consider as a state (default: 1)
            memory_threshold (int): Maximum number of states before switching to DB
            db_config (dict, optional): PostgreSQL configuration dictionary
            environment (str): Which environment to use ('development' or 'test')
        
        Attributes:
            transitions: Dictionary for in-memory transition storage
            total_counts: Dictionary for in-memory total counts
            n_gram: The size of word sequences to use as states
            using_db: Flag indicating if DB storage is active
            conn_pool: Connection pool for PostgreSQL (if used)
            environment: Current environment setting ('development' or 'test')
        """
        self.n_gram = n_gram
        self.memory_threshold = memory_threshold
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.using_db = False
        self.conn_pool = None
        self.environment = environment

        # Initialize text preprocessor if available
        self.preprocessor = None
        if TEXT_PREPROCESSOR_AVAILABLE:
            try:
                self.preprocessor = TextPreprocessor()
                logger.info("TextPreprocessor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TextPreprocessor: {e}")
        
        # Load database configuration if not provided
        if db_config is None:
            db_config = self._load_db_config()
            
        # Initialize database connection if config is available
        if db_config:
            try:
                self.conn_pool = pool.SimpleConnectionPool(
                    1, 10,  # min and max connections
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    dbname=db_config.get('dbname', 'markov_chain'),
                    user=db_config.get('user', 'postgres'),
                    password=db_config.get('password', ''),
                )
                # Test connection
                conn = self._get_connection()
                self._setup_database(conn)
                self._return_connection(conn)
                logger.info(f"Successfully connected to PostgreSQL database for {self.environment} environment")
            except Exception as e:
                logger.warning(f"Failed to connect to database: {e}")
                logger.warning("Falling back to in-memory storage only")
                self.conn_pool = None

    def _load_db_config(self):
        """
        Load database configuration from file.
        
        Returns:
            dict: Database configuration or None if not found
        """
        # Base config paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_dir = os.path.join(project_root, "configs")
        
        # Try to get environment-specific configuration first
        env_config_paths = [
            os.path.join(config_dir, f"database_{self.environment}.yaml"),
        ]
        
        # Fallback to default configuration
        default_config_paths = [
            os.path.join(config_dir, "database.yaml"),
        ]
        
        # First check environment-specific configs
        for config_path in env_config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        logger.info(f"Loaded database config from {config_path}")
                        return config
                except Exception as e:
                    logger.warning(f"Error loading database config from {config_path}: {e}")
        
        # Then check default configs if environment-specific ones not found
        for config_path in default_config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        logger.info(f"Loaded database config from {config_path}")
                        return config
                except Exception as e:
                    logger.warning(f"Error loading database config from {config_path}: {e}")
        
        logger.warning("No database configuration found")
        return None

    def _get_connection(self):
        """Get a connection from the pool"""
        if self.conn_pool:
            return self.conn_pool.getconn()
        return None

    def _return_connection(self, conn):
        """Return a connection to the pool"""
        if self.conn_pool:
            self.conn_pool.putconn(conn)

    def _setup_database(self, conn):
        """Set up the necessary database tables and indexes"""
        if not conn:
            return
            
        # Add environment suffix to table names to separate test and dev data
        table_prefix = f"markov_{self.environment}"
        
        try:
            # First check if the database schema exists
            with conn.cursor() as cur:
                # Check if the transitions table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (f"{table_prefix}_transitions",))
                
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    logger.info(f"Creating database schema for {self.environment} environment")
                else:
                    logger.info(f"Database schema for {self.environment} environment already exists")
                    
                # Create transitions table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_prefix}_transitions (
                        state TEXT,
                        next_word TEXT,
                        count INTEGER,
                        n_gram INTEGER,
                        PRIMARY KEY (state, next_word, n_gram)
                    )
                """)
                
                # Create indexes
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_prefix}_transitions_state_ngram 
                    ON {table_prefix}_transitions(state, n_gram)
                """)
                
                # Create total counts table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_prefix}_total_counts (
                        state TEXT,
                        count INTEGER,
                        n_gram INTEGER,
                        PRIMARY KEY (state, n_gram)
                    )
                """)
                
            conn.commit()
            logger.info(f"Database setup complete for {self.environment} environment")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error setting up database: {e}")
            raise

    def train(self, text, clear_previous=True, preprocess=True):
        """
        Trains the Markov Chain on a given text by learning word transitions.
        
        Args:
            text (str): The input text used to train the Markov Chain.
            clear_previous (bool): Whether to clear previous training data
            preprocess (bool): Whether to preprocess the text before training
        """
        # Apply preprocessing if requested
        if preprocess:
            text = self._preprocess_text(text)
            
        words = text.split()
        if len(words) < self.n_gram + 1:
            # If there aren't enough words, no transitions can be learned
            logger.warning("Text too short for training with current n-gram setting")
            return

        # Estimate the potential size of the model
        unique_words_estimate = len(set(words))
        estimated_transitions = unique_words_estimate ** 2
        
        # Determine storage strategy
        use_db = (estimated_transitions > self.memory_threshold) and self.conn_pool is not None
        
        if use_db:
            self._train_using_db(words, clear_previous)
        else:
            self._train_using_memory(words, clear_previous)

    def _train_using_memory(self, words, clear_previous):
        """Train the model using in-memory storage"""
        if clear_previous:
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
            
        self.using_db = False
        logger.info(f"Trained model in memory with {len(self.transitions)} states")

    def _train_using_db(self, words, clear_previous):
        """Train the model using PostgreSQL storage"""
        conn = self._get_connection()
        if not conn:
            logger.warning("Failed to get database connection, falling back to memory")
            self._train_using_memory(words, clear_previous)
            return
        
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
            
        try:
            # Clear previous data if requested
            if clear_previous:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        DELETE FROM {table_prefix}_transitions WHERE n_gram = %s
                    """, (self.n_gram,))
                    cur.execute(f"""
                        DELETE FROM {table_prefix}_total_counts WHERE n_gram = %s
                    """, (self.n_gram,))
                conn.commit()
            
            # Process transitions in batches
            batch_size = 5000
            transitions_batch = []
            
            for i in range(len(words) - self.n_gram):
                if self.n_gram == 1:
                    current_state = words[i]
                    next_word = words[i + 1]
                else:
                    current_state = ' '.join(words[i:i + self.n_gram])
                    next_word = words[i + self.n_gram]
                    
                # Add to batch
                transitions_batch.append((current_state, next_word, 1, self.n_gram))
                
                # Process batch when it reaches the threshold
                if len(transitions_batch) >= batch_size:
                    self._insert_transitions_batch(conn, transitions_batch)
                    transitions_batch = []
            
            # Insert any remaining transitions
            if transitions_batch:
                self._insert_transitions_batch(conn, transitions_batch)
                
            # Update total counts
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {table_prefix}_total_counts (state, count, n_gram)
                    SELECT state, SUM(count), n_gram 
                    FROM {table_prefix}_transitions 
                    WHERE n_gram = %s
                    GROUP BY state, n_gram
                    ON CONFLICT (state, n_gram) 
                    DO UPDATE SET count = EXCLUDED.count
                """, (self.n_gram,))
                
            conn.commit()
            self.using_db = True
            logger.info(f"Successfully trained model in {self.environment} database")
            
            # Clear in-memory structures to save RAM
            self.transitions.clear()
            self.total_counts.clear()
            
        except Exception as e:
            logger.error(f"Database error during training: {e}")
            conn.rollback()
            logger.warning("Falling back to in-memory training")
            self._train_using_memory(words, clear_previous)
        finally:
            self._return_connection(conn)

    def _insert_transitions_batch(self, conn, batch):
        """Insert a batch of transitions into the database"""
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
        
        with conn.cursor() as cur:
            execute_values(cur, f"""
                INSERT INTO {table_prefix}_transitions (state, next_word, count, n_gram) 
                VALUES %s
                ON CONFLICT (state, next_word, n_gram) 
                DO UPDATE SET count = {table_prefix}_transitions.count + EXCLUDED.count
            """, batch)

    def _preprocess_text(self, text):
        """
        Perform comprehensive text normalization using TextPreprocessor.
        
        This method applies a sequence of normalization steps to improve text quality
        for both training and generation:
        
        1. Converts to lowercase
        2. Expands contractions
        3. Normalizes for accents/special characters
        4. Normalizes social media text (hashtags, mentions)
        5. Handles emojis
        6. Removes URLs 
        7. Removes HTML tags
        8. Handles whitespace
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Normalized text ready for processing
        """
        if self.preprocessor is None:
            logger.warning("TextPreprocessor not available, skipping normalization")
            return text
        try:
            preprocessor = TextPreprocessor()
            
            # Apply comprehensive normalization pipeline
            text = preprocessor.to_lowercase(text)
            text = preprocessor.handle_contractions(text)
            text = preprocessor.normalize(text)
            text = preprocessor.normalize_social_media_text(text)
            text = preprocessor.handle_emojis(text)
            text = preprocessor.handle_urls(text)
            text = preprocessor.remove_html_tags(text)
            text = preprocessor.handle_whitespace(text)
            
            logger.info("Text normalization completed")
            return text

        except Exception as e:
            logger.error(f"Error during text normalization: {e}")
            return text


    def predict(self, current_state, preprocess=True):
        """
        Predicts the next word based on the current state using learned probabilities.
        
        Args:
            current_state: The current state (word or tuple of words) for prediction.
            preprocess (bool): Whether to preprocess the input state
        
        Returns:
            str or None: The predicted next word, or None if the current state is not in the model.
        """
        # Handle string input with preprocessing if requested
        if isinstance(current_state, str):
            if preprocess:
                current_state = self._preprocess_text(current_state)
                
            # Handle n-gram conversion for string input
            if self.n_gram > 1:
                words = current_state.split()
                if len(words) >= self.n_gram:
                    current_state = tuple(words[:self.n_gram])
                else:
                    return None
        
        # Choose the appropriate prediction method based on storage
        if self.using_db:
            return self._predict_from_db(current_state)
        else:
            return self._predict_from_memory(current_state)

    def _predict_from_memory(self, current_state):
        """Predict next word using in-memory storage"""
        if current_state not in self.transitions:
            return None

        next_words = self.transitions[current_state]
        total = self.total_counts[current_state]
        probabilities = {word: count / total for word, count in next_words.items()}

        return random.choices(
            list(probabilities.keys()), weights=probabilities.values()
        )[0]

    def _predict_from_db(self, current_state):
        """Predict next word using database storage"""
        conn = self._get_connection()
        if not conn:
            logger.warning("Failed to get database connection for prediction")
            return None
        
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
            
        try:
            # Convert tuple to string for database query
            if isinstance(current_state, tuple):
                db_state = ' '.join(current_state)
            else:
                db_state = str(current_state)
                
            with conn.cursor() as cur:
                # Get the total count for this state
                cur.execute(f"""
                    SELECT count FROM {table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """, (db_state, self.n_gram))
                
                result = cur.fetchone()
                if not result:
                    return None
                    
                total_count = result[0]
                
                # Get all transitions with their counts
                cur.execute(f"""
                    SELECT next_word, count 
                    FROM {table_prefix}_transitions
                    WHERE state = %s AND n_gram = %s
                """, (db_state, self.n_gram))
                
                transitions = cur.fetchall()
                if not transitions:
                    return None
                    
                # Calculate probabilities
                next_words = [word for word, _ in transitions]
                probabilities = [count / total_count for _, count in transitions]
                
                # Choose next word based on probabilities
                return random.choices(next_words, weights=probabilities)[0]
                
        except Exception as e:
            logger.error(f"Database error during prediction: {e}")
            return None
        finally:
            self._return_connection(conn)

    def generate_text(self, start=None, max_length=100, preprocess=True):
        """
        Generates text starting from a given state.
        
        Args:
            start: Starting state (word or tuple). If None, a random state is chosen.
            max_length (int): Maximum number of words to generate.
            preprocess (bool): Whether to preprocess the starting state
        
        Returns:
            str: Generated text.
        """
        # Determine if we have any transitions to work with
        if self.using_db:
            has_transitions = self._check_db_has_transitions()
        else:
            has_transitions = bool(self.transitions)
            
        if not has_transitions:
            return "Model not trained"

        # Preprocess the starting state if needed
        if preprocess and start and isinstance(start, str):
            start = self._preprocess_text(start)
            
        # Get a valid starting state
        current_state = self._get_valid_start_state(start)
        if current_state is None:
            return "Could not find valid starting state"
        
        # Initialize text generation
        text = []
        
        # Convert tuple to list of words for output
        if isinstance(current_state, tuple):
            text.extend(list(current_state))
        else:
            text.append(current_state)
        
        # Calculate how many more words to generate
        remaining = max_length - (self.n_gram if isinstance(current_state, tuple) else 1)
        
        # Generate remaining words
        for _ in range(remaining):
            next_word = self.predict(current_state, preprocess=False)  # Already preprocessed
            if next_word is None:
                break
            
            text.append(next_word)
            
            # Update current state for n-grams
            if self.n_gram > 1:
                if isinstance(current_state, tuple):
                    current_state = tuple(list(current_state)[1:] + [next_word])
                else:
                    # Handle case where current_state might be a string
                    words = current_state.split() if isinstance(current_state, str) else [current_state]
                    words = words + [next_word]
                    current_state = tuple(words[-self.n_gram:])
            else:
                current_state = next_word
        
        return " ".join(text)

    def _check_db_has_transitions(self):
        """Check if the database has any transitions for this n-gram"""
        conn = self._get_connection()
        if not conn:
            return False
            
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
            
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {table_prefix}_transitions WHERE n_gram = %s LIMIT 1
                    )
                """, (self.n_gram,))
                result = cur.fetchone()
                return result[0] if result else False
        except Exception as e:
            logger.error(f"Error checking transitions: {e}")
            return False
        finally:
            self._return_connection(conn)

    def _get_valid_start_state(self, start):
        """Get a valid starting state, either the provided one or a random one"""
        # If start is None, choose randomly
        if start is None:
            if self.using_db:
                return self._get_random_db_state()
            else:
                return random.choice(list(self.transitions.keys())) if self.transitions else None
        
        # Process the provided start state
        if isinstance(start, str) and self.n_gram > 1:
            words = start.split()
            if len(words) >= self.n_gram:
                start = tuple(words[:self.n_gram])
            else:
                # Not enough words, get a random state
                if self.using_db:
                    return self._get_random_db_state()
                else:
                    return random.choice(list(self.transitions.keys())) if self.transitions else None
        
        # Validate that start exists in model
        if self.using_db:
            if not self._check_state_exists_in_db(start):
                return self._get_random_db_state()
        else:
            if start not in self.transitions:
                return random.choice(list(self.transitions.keys())) if self.transitions else None
                
        return start

    def _get_random_db_state(self):
        """Get a random state from the database"""
        conn = self._get_connection()
        if not conn:
            return None
            
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
            
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT state FROM {table_prefix}_transitions
                    WHERE n_gram = %s
                    GROUP BY state
                    ORDER BY RANDOM()
                    LIMIT 1
                """, (self.n_gram,))
                result = cur.fetchone()
                
                if not result:
                    return None
                    
                state = result[0]
                
                # Convert to tuple if n_gram > 1
                if self.n_gram > 1:
                    return tuple(state.split())
                return state
                
        except Exception as e:
            logger.error(f"Error getting random state: {e}")
            return None
        finally:
            self._return_connection(conn)

    def _check_state_exists_in_db(self, state):
        """Check if the state exists in the database"""
        conn = self._get_connection()
        if not conn:
            return False
            
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"
            
        try:
            # Convert tuple to string for query
            if isinstance(state, tuple):
                db_state = ' '.join(state)
            else:
                db_state = str(state)
                
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {table_prefix}_transitions
                        WHERE state = %s AND n_gram = %s
                        LIMIT 1
                    )
                """, (db_state, self.n_gram))
                result = cur.fetchone()
                return result[0] if result else False
        except Exception as e:
            logger.error(f"Error checking state existence: {e}")
            return False
        finally:
            self._return_connection(conn)

    def __del__(self):
        """Cleanup method to close database connections"""
        if self.conn_pool:
            try:
                self.conn_pool.closeall()
            except:
                pass

# Example usage for text generation
markov_chain = MarkovChain(n_gram=2, memory_threshold=10000, environment="test")
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain.train(text)
generated_text = markov_chain.generate_text(start="It was", max_length=50)
print(generated_text)
print("\n")

# Example usage for predicting next word
markov_chain = MarkovChain(n_gram=1, memory_threshold=10000, environment="test")
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain.train(text)
predicted_word = markov_chain.predict("striking")
print(predicted_word)
print("\n")

# Example usage for generating text using PostgreSQL using test environment
markov_chain_test = MarkovChain(n_gram=2, memory_threshold=10000, environment="test")
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain_test.train(text)
generated_text_test = markov_chain_test.generate_text(start="It was", max_length=50)
print(generated_text_test)
print("\n")

# Example usage for predicting next word using PostgreSQL using test environment
markov_chain_test = MarkovChain(n_gram=2, memory_threshold=10000, environment="test")
text = "It was a bright cold day in April, and the clocks were striking thirteen."
markov_chain_test.train(text)
predicted_word_test = markov_chain_test.predict("It was")
print(predicted_word_test)
print("\n")

# Example usage with preprocessing
markov_chain = MarkovChain(n_gram=2, memory_threshold=10000, environment="test")

# Raw text with various issues that preprocessing will handle
raw_text = """It was a bright cold day in April, and the clocks were striking thirteen. 
Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, 
slipped quickly through the glass doors of Victory Mansions, though not quickly 
enough to prevent a swirl of gritty dust from entering along with him.
http://example.com/test?page=1 
<b>HTML tags</b> should be removed!
Don't forget about contractions :) 😊"""

# Train with preprocessing
markov_chain.train(raw_text, preprocess=True)

# Generate text
generated_text = markov_chain.generate_text(start="It was", max_length=50)
print("Generated text with preprocessed training:")
print(generated_text)
print("\n")

# Compare with specific normalization
normalized_start = markov_chain._preprocess_text("It's cold in April, don't you think? 🥶")
print("Normalized input:", normalized_start)
generated_normalized = markov_chain.generate_text(start=normalized_start, max_length=30)
print("Generated from normalized input:")
print(generated_normalized)



