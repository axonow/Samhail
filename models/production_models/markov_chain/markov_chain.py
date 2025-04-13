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
import pickle
import datetime

# Import the enhanced JSON logger
try:
    from json_logger import get_logger, log_json
except ImportError:
    # Fallback if json_logger is not found (for direct imports from other directories)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    try:
        from json_logger import get_logger, log_json
    except ImportError:
        print(
            "\033[1mWarning: json_logger module not found, using basic logging\033[0m"
        )
        get_logger = None
        log_json = None

# Setup default logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Set up the JSON logger
# Define the log file path for consistent logging across modules
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
log_file_path = os.path.join(log_dir, "test_run.log")

# Configure the JSON logger if available
if get_logger:
    logger = get_logger("markov_chain", log_file=log_file_path)
else:
    logger = logging.getLogger("markov_chain")

# Import the TextPreprocessor class
try:
    # Add project root to Python path
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    sys.path.insert(0, project_root)

    from data_preprocessing.text_preprocessor import TextPreprocessor

    TEXT_PREPROCESSOR_AVAILABLE = True
    logger.info("TextPreprocessor successfully imported")
except ImportError as e:
    TEXT_PREPROCESSOR_AVAILABLE = False
    print(f"\033[1mImport error: {e}\033[0m")  # Print in bold format
    # The path might be incorrect or the module not installed


class MarkovChain:
    """
    A hybrid implementation of a Markov Chain for text generation.

    This class implements an n-gram Markov model that can use either in-memory
    storage or PostgreSQL database storage based on data size and configuration.
    """

    def __init__(
        self,
        n_gram=1,
        memory_threshold=10000,
        db_config=None,
        environment="development",
    ):
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

        # Log initialization details
        init_data = {
            "n_gram": n_gram,
            "memory_threshold": memory_threshold,
            "environment": environment,
            "preprocessor_available": TEXT_PREPROCESSOR_AVAILABLE,
        }
        if log_json:
            log_json(logger, "MarkovChain initialized", init_data)

        # Load database configuration if not provided
        if db_config is None:
            db_config = self._load_db_config()

        # Initialize database connection if config is available
        if db_config:
            try:
                self.conn_pool = pool.SimpleConnectionPool(
                    1,
                    10,  # min and max connections
                    host=db_config.get("host", "localhost"),
                    port=db_config.get("port", 5432),
                    dbname=db_config.get("dbname", "markov_chain"),
                    user=db_config.get("user", "postgres"),
                    password=db_config.get("password", ""),
                )
                # Test connection
                conn = self._get_connection()
                self._setup_database(conn)
                self._return_connection(conn)
                logger.info(
                    f"Successfully connected to PostgreSQL database for {self.environment} environment"
                )

                # Log successful database connection
                if log_json:
                    db_info = {
                        "host": db_config.get("host", "localhost"),
                        "port": db_config.get("port", 5432),
                        "dbname": db_config.get("dbname", "markov_chain"),
                        "environment": self.environment,
                    }
                    log_json(logger, "Database connection established", db_info)
            except Exception as e:
                logger.warning(f"Failed to connect to database: {e}")
                logger.warning("Falling back to in-memory storage only")
                self.conn_pool = None

                # Log database connection failure
                if log_json:
                    log_json(
                        logger,
                        "Database connection failed, using in-memory storage",
                        {"error": str(e), "environment": self.environment},
                    )

    def _load_db_config(self):
        """
        Load database configuration from file.

        Returns:
            dict: Database configuration or None if not found
        """
        # Base config paths
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
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
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        logger.info(
                            f"Loaded database config from {config_path}")

                        # Log config loading
                        if log_json:
                            log_json(
                                logger,
                                "Database config loaded",
                                {
                                    "config_path": config_path,
                                    "environment": self.environment,
                                },
                            )
                        return config
                except Exception as e:
                    logger.warning(
                        f"Error loading database config from {config_path}: {e}"
                    )

        # Then check default configs if environment-specific ones not found
        for config_path in default_config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        logger.info(
                            f"Loaded database config from {config_path}")

                        # Log config loading
                        if log_json:
                            log_json(
                                logger,
                                "Database config loaded (default)",
                                {
                                    "config_path": config_path,
                                    "environment": self.environment,
                                },
                            )
                        return config
                except Exception as e:
                    logger.warning(
                        f"Error loading database config from {config_path}: {e}"
                    )

        logger.warning("No database configuration found")

        # Log no config found
        if log_json:
            log_json(
                logger,
                "No database configuration found",
                {"environment": self.environment},
            )
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
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """,
                    (f"{table_prefix}_transitions",),
                )

                table_exists = cur.fetchone()[0]

                if not table_exists:
                    logger.info(
                        f"Creating database schema for {self.environment} environment"
                    )
                else:
                    logger.info(
                        f"Database schema for {self.environment} environment already exists"
                    )

                # Create transitions table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_prefix}_transitions (
                        state TEXT,
                        next_word TEXT,
                        count INTEGER,
                        n_gram INTEGER,
                        PRIMARY KEY (state, next_word, n_gram)
                    )
                """
                )

                # Create indexes
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_prefix}_transitions_state_ngram 
                    ON {table_prefix}_transitions(state, n_gram)
                """
                )

                # Create total counts table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_prefix}_total_counts (
                        state TEXT,
                        count INTEGER,
                        n_gram INTEGER,
                        PRIMARY KEY (state, n_gram)
                    )
                """
                )

            conn.commit()
            logger.info(
                f"Database setup complete for {self.environment} environment")

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
            logger.warning(
                "Text too short for training with current n-gram setting")

            # Log the warning
            if log_json:
                log_json(
                    logger,
                    "Text too short for training",
                    {
                        "n_gram": self.n_gram,
                        "text_length": len(words),
                        "min_required": self.n_gram + 1,
                    },
                )
            return

        # Estimate the potential size of the model
        unique_words_estimate = len(set(words))
        estimated_transitions = unique_words_estimate**2

        # Determine storage strategy
        use_db = (
            estimated_transitions > self.memory_threshold
        ) and self.conn_pool is not None

        # Log training parameters
        if log_json:
            log_json(
                logger,
                "Training started",
                {
                    "text_sample": text[:100] + ("..." if len(text) > 100 else ""),
                    "word_count": len(words),
                    "unique_words": unique_words_estimate,
                    "estimated_transitions": estimated_transitions,
                    "storage": "database" if use_db else "memory",
                    "n_gram": self.n_gram,
                    "clear_previous": clear_previous,
                    "preprocess": preprocess,
                },
            )

        if use_db:
            self._train_using_db(words, clear_previous)
        else:
            self._train_using_memory(words, clear_previous)

        # Log training completion
        if log_json:
            log_json(
                logger,
                "Training completed",
                {
                    "storage": "database" if self.using_db else "memory",
                    "n_gram": self.n_gram,
                    "word_count": len(words),
                },
            )

    def train_parallel(self, texts, clear_previous=True, preprocess=True, n_jobs=-1):
        """
        Train model on multiple texts using parallel processing

        Args:
            texts (list): List of text strings to train on
            clear_previous (bool): Whether to clear previous training
            preprocess (bool): Whether to preprocess texts
            n_jobs (int): Number of parallel jobs (-1 for all cores)
        """
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()

        # Clear if requested
        if clear_previous:
            if self.using_db:
                self._clear_db_tables()
            else:
                self.transitions.clear()
                self.total_counts.clear()

        # Helper function for parallel processing
        def process_text(text):
            if preprocess:
                text = self._preprocess_text(text)
            words = text.split()

            # Extract transitions but don't update model yet
            local_transitions = defaultdict(lambda: defaultdict(int))
            for i in range(len(words) - self.n_gram):
                if self.n_gram == 1:
                    state = words[i]
                else:
                    state = tuple(words[i: i + self.n_gram])
                next_word = words[i + self.n_gram]
                local_transitions[state][next_word] += 1

            return local_transitions

        # Process texts in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_text, texts))

        # Combine all results
        for local_transitions in results:
            for state, next_words in local_transitions.items():
                for next_word, count in next_words.items():
                    if self.using_db:
                        self._increment_db_transition(state, next_word, count)
                    else:
                        self.transitions[state][next_word] += count
                        self.total_counts[state] += count

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
                current_state = tuple(words[i: i + self.n_gram])
                next_word = words[i + self.n_gram]

            # Update transitions
            self.transitions[current_state][next_word] += 1

            # Directly update total counts
            self.total_counts[current_state] += 1

        self.using_db = False
        logger.info(
            f"Trained model in memory with {len(self.transitions)} states")

        # Log memory training details
        if log_json:
            log_json(
                logger,
                "Memory training completed",
                {
                    "states_count": len(self.transitions),
                    "transitions_count": sum(
                        len(next_words) for next_words in self.transitions.values()
                    ),
                    "n_gram": self.n_gram,
                },
            )

    def _train_using_db(self, words, clear_previous):
        """Train the model using PostgreSQL storage"""
        conn = self._get_connection()
        if not conn:
            logger.warning(
                "Failed to get database connection, falling back to memory")

            # Log the fallback
            if log_json:
                log_json(
                    logger,
                    "Database training fallback to memory",
                    {
                        "reason": "Failed to get database connection",
                        "n_gram": self.n_gram,
                    },
                )

            self._train_using_memory(words, clear_previous)
            return

        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"

        try:
            # Clear previous data if requested
            if clear_previous:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM {table_prefix}_transitions WHERE n_gram = %s
                    """,
                        (self.n_gram,),
                    )
                    cur.execute(
                        f"""
                        DELETE FROM {table_prefix}_total_counts WHERE n_gram = %s
                    """,
                        (self.n_gram,),
                    )
                conn.commit()

                # Log clearing of previous data
                if log_json:
                    log_json(
                        logger,
                        "Cleared previous database training data",
                        {"environment": self.environment, "n_gram": self.n_gram},
                    )

            # Process transitions in batches
            batch_size = 5000
            transitions_batch = []
            total_transitions = 0

            for i in range(len(words) - self.n_gram):
                if self.n_gram == 1:
                    current_state = words[i]
                    next_word = words[i + 1]
                else:
                    current_state = " ".join(words[i: i + self.n_gram])
                    next_word = words[i + self.n_gram]

                # Add to batch
                transitions_batch.append(
                    (current_state, next_word, 1, self.n_gram))

                # Process batch when it reaches the threshold
                if len(transitions_batch) >= batch_size:
                    self._insert_transitions_batch(conn, transitions_batch)
                    total_transitions += len(transitions_batch)
                    transitions_batch = []

                    # Log batch processing
                    if log_json and total_transitions % (batch_size * 5) == 0:
                        log_json(
                            logger,
                            "Database training progress",
                            {
                                "transitions_processed": total_transitions,
                                "environment": self.environment,
                                "n_gram": self.n_gram,
                            },
                        )

            # Insert any remaining transitions
            if transitions_batch:
                self._insert_transitions_batch(conn, transitions_batch)
                total_transitions += len(transitions_batch)

            # Update total counts
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {table_prefix}_total_counts (state, count, n_gram)
                    SELECT state, SUM(count), n_gram 
                    FROM {table_prefix}_transitions 
                    WHERE n_gram = %s
                    GROUP BY state, n_gram
                    ON CONFLICT (state, n_gram) 
                    DO UPDATE SET count = EXCLUDED.count
                """,
                    (self.n_gram,),
                )

            conn.commit()
            self.using_db = True
            logger.info(
                f"Successfully trained model in {self.environment} database")

            # Log database training completion
            if log_json:
                log_json(
                    logger,
                    "Database training completed",
                    {
                        "environment": self.environment,
                        "n_gram": self.n_gram,
                        "total_transitions": total_transitions,
                    },
                )

            # Clear in-memory structures to save RAM
            self.transitions.clear()
            self.total_counts.clear()

        except Exception as e:
            logger.error(f"Database error during training: {e}")
            conn.rollback()
            logger.warning("Falling back to in-memory training")

            # Log database error
            if log_json:
                log_json(
                    logger,
                    "Database training error",
                    {
                        "error": str(e),
                        "environment": self.environment,
                        "n_gram": self.n_gram,
                        "fallback": "memory",
                    },
                )

            self._train_using_memory(words, clear_previous)
        finally:
            self._return_connection(conn)

    def _insert_transitions_batch(self, conn, batch):
        """Insert a batch of transitions into the database"""
        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"

        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {table_prefix}_transitions (state, next_word, count, n_gram) 
                VALUES %s
                ON CONFLICT (state, next_word, n_gram) 
                DO UPDATE SET count = {table_prefix}_transitions.count + EXCLUDED.count
            """,
                batch,
            )

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
            logger.warning(
                "TextPreprocessor not available, skipping normalization")

            # Log skipping normalization
            if log_json:
                log_json(
                    logger,
                    "Text normalization skipped - preprocessor not available",
                    {"text_sample": text[:50] +
                        ("..." if len(text) > 50 else "")},
                )

            return text
        try:
            # Log normalization start
            if log_json:
                log_json(
                    logger,
                    "Text normalization started",
                    {
                        "original_text_sample": text[:50]
                        + ("..." if len(text) > 50 else ""),
                        "text_length": len(text),
                    },
                )

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

            # Log normalization result
            if log_json:
                log_json(
                    logger,
                    "Text normalization completed",
                    {
                        "normalized_text_sample": text[:50]
                        + ("..." if len(text) > 50 else ""),
                        "text_length": len(text),
                    },
                )

            return text

        except Exception as e:
            logger.error(f"Error during text normalization: {e}")

            # Log normalization error
            if log_json:
                log_json(
                    logger,
                    "Text normalization error",
                    {"error": str(e), "error_type": type(e).__name__},
                )

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
        # Log prediction attempt
        if log_json:
            log_json(
                logger,
                "Prediction attempt",
                {
                    "current_state": (
                        str(current_state)
                        if not isinstance(current_state, tuple)
                        else " ".join(current_state)
                    ),
                    "n_gram": self.n_gram,
                    "storage": "database" if self.using_db else "memory",
                    "preprocess": preprocess,
                },
            )

        # Handle string input with preprocessing if requested
        if isinstance(current_state, str):
            if preprocess:
                current_state = self._preprocess_text(current_state)

            # Handle n-gram conversion for string input
            if self.n_gram > 1:
                words = current_state.split()
                if len(words) >= self.n_gram:
                    current_state = tuple(words[-self.n_gram:])
                else:
                    # Log insufficient words
                    if log_json:
                        log_json(
                            logger,
                            "Prediction failed - insufficient words",
                            {
                                "words_provided": len(words),
                                "words_required": self.n_gram,
                            },
                        )
                    return None

        # Choose the appropriate prediction method based on storage
        if self.using_db:
            next_word = self._predict_from_db(current_state)
        else:
            next_word = self._predict_from_memory(current_state)

        # Log prediction result
        if log_json:
            log_json(
                logger,
                "Prediction result",
                {
                    "current_state": (
                        str(current_state)
                        if not isinstance(current_state, tuple)
                        else " ".join(current_state)
                    ),
                    "next_word": next_word if next_word is not None else "None",
                    "success": next_word is not None,
                    "storage": "database" if self.using_db else "memory",
                },
            )

        return next_word

    def _predict_from_memory(self, current_state):
        """Predict next word using in-memory storage"""
        if current_state not in self.transitions:
            return None

        next_words = self.transitions[current_state]
        total = self.total_counts[current_state]
        probabilities = {word: count / total for word,
                         count in next_words.items()}

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
                db_state = " ".join(current_state)
            else:
                db_state = str(current_state)

            with conn.cursor() as cur:
                # Get the total count for this state
                cur.execute(
                    f"""
                    SELECT count FROM {table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, self.n_gram),
                )

                result = cur.fetchone()
                if not result:
                    return None

                total_count = result[0]

                # Get all transitions with their counts
                cur.execute(
                    f"""
                    SELECT next_word, count 
                    FROM {table_prefix}_transitions
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, self.n_gram),
                )

                transitions = cur.fetchall()
                if not transitions:
                    return None

                # Calculate probabilities
                next_words = [word for word, _ in transitions]
                probabilities = [
                    count / total_count for _, count in transitions]

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
        # Log generation start
        if log_json:
            log_json(
                logger,
                "Text generation started",
                {
                    "start": str(start) if start else "random",
                    "max_length": max_length,
                    "n_gram": self.n_gram,
                    "storage": "database" if self.using_db else "memory",
                    "preprocess": preprocess,
                },
            )

        # Determine if we have any transitions to work with
        if self.using_db:
            has_transitions = self._check_db_has_transitions()
        else:
            has_transitions = bool(self.transitions)

        if not has_transitions:
            # Log no transitions available
            if log_json:
                log_json(
                    logger,
                    "Text generation failed - model not trained",
                    {"storage": "database" if self.using_db else "memory"},
                )
            return "Model not trained"

        # Preprocess the starting state if needed
        if preprocess and start and isinstance(start, str):
            start = self._preprocess_text(start)

        # Get a valid starting state
        current_state = self._get_valid_start_state(start)
        if current_state is None:
            # Log no valid starting state
            if log_json:
                log_json(
                    logger,
                    "Text generation failed - no valid starting state",
                    {"requested_start": str(start) if start else "random"},
                )
            return "Could not find valid starting state"

        # Initialize text generation
        text = []

        # Convert tuple to list of words for output
        if isinstance(current_state, tuple):
            text.extend(list(current_state))
        else:
            text.append(current_state)

        # Calculate how many more words to generate
        remaining = max_length - (
            self.n_gram if isinstance(current_state, tuple) else 1
        )

        # Generate remaining words
        for i in range(remaining):
            next_word = self.predict(
                current_state, preprocess=False
            )  # Already preprocessed
            if next_word is None:
                # Log early termination
                if log_json:
                    log_json(
                        logger,
                        "Text generation ended early - no prediction available",
                        {
                            "words_generated": len(text),
                            "last_state": (
                                str(current_state)
                                if not isinstance(current_state, tuple)
                                else " ".join(current_state)
                            ),
                        },
                    )
                break

            text.append(next_word)

            # Update current state for n-grams
            if self.n_gram > 1:
                if isinstance(current_state, tuple):
                    current_state = tuple(
                        list(current_state)[1:] + [next_word])
                else:
                    # Handle case where current_state might be a string
                    words = (
                        current_state.split()
                        if isinstance(current_state, str)
                        else [current_state]
                    )
                    words = words + [next_word]
                    current_state = tuple(words[-self.n_gram:])
            else:
                current_state = next_word

            # Periodically log progress for long sequences
            if log_json and i > 0 and i % 50 == 0:
                log_json(
                    logger,
                    "Text generation progress",
                    {
                        "words_generated": len(text),
                        "percentage_complete": f"{i/remaining*100:.1f}%",
                    },
                )

        generated_text = " ".join(text)

        # Log generation completion
        if log_json:
            log_json(
                logger,
                f"Text generation completed ({self.n_gram}-gram, {'PostgreSQL' if self.using_db else 'memory'})",
                {
                    "text": generated_text,
                    "n_gram": self.n_gram,
                    "start": str(start) if start else "random",
                    "words_generated": len(text),
                    "storage": "PostgreSQL" if self.using_db else "memory",
                },
            )

        return generated_text

    def _check_db_has_transitions(self):
        """Check if the database has any transitions for this n-gram"""
        conn = self._get_connection()
        if not conn:
            return False

        # Add environment suffix to table names
        table_prefix = f"markov_{self.environment}"

        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {table_prefix}_transitions WHERE n_gram = %s LIMIT 1
                    )
                """,
                    (self.n_gram,),
                )
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
                return (
                    random.choice(list(self.transitions.keys()))
                    if self.transitions
                    else None
                )

        # Process the provided start state
        if isinstance(start, str) and self.n_gram > 1:
            words = start.split()
            if len(words) >= self.n_gram:
                start = tuple(words[: self.n_gram])
            else:
                # Not enough words, get a random state
                if self.using_db:
                    return self._get_random_db_state()
                else:
                    return (
                        random.choice(list(self.transitions.keys()))
                        if self.transitions
                        else None
                    )

        # Validate that start exists in model
        if self.using_db:
            if not self._check_state_exists_in_db(start):
                return self._get_random_db_state()
        else:
            if start not in self.transitions:
                return (
                    random.choice(list(self.transitions.keys()))
                    if self.transitions
                    else None
                )

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
                cur.execute(
                    f"""
                    SELECT state FROM {table_prefix}_transitions
                    WHERE n_gram = %s
                    GROUP BY state
                    ORDER BY RANDOM()
                    LIMIT 1
                """,
                    (self.n_gram,),
                )
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
                db_state = " ".join(state)
            else:
                db_state = str(state)

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {table_prefix}_transitions
                        WHERE state = %s AND n_gram = %s
                        LIMIT 1
                    )
                """,
                    (db_state, self.n_gram),
                )
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

    def save_model(self, filepath):
        """Save model to a file"""
        model_data = {
            "n_gram": self.n_gram,
            "environment": self.environment,
            "memory_threshold": self.memory_threshold,
        }

        if not self.using_db:
            model_data["transitions"] = dict(self.transitions)
            model_data["total_counts"] = dict(self.total_counts)

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")
        return True

    @classmethod
    def load_model(cls, filepath, db_config=None):
        """Load model from a file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Create new model with same parameters
        model = cls(
            n_gram=model_data["n_gram"],
            memory_threshold=model_data["memory_threshold"],
            db_config=db_config,
            environment=model_data["environment"],
        )

        # Load in-memory data if present
        if "transitions" in model_data:
            model.transitions = defaultdict(lambda: defaultdict(int))
            for state, next_words in model_data["transitions"].items():
                for next_word, count in next_words.items():
                    model.transitions[state][next_word] = count

            model.total_counts = defaultdict(int)
            for state, count in model_data["total_counts"].items():
                model.total_counts[state] = count

            model.using_db = False

        logger.info(f"Model loaded from {filepath}")
        return model
