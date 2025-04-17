#!/usr/bin/env python3
"""
MarkovChainPostgreSqlAdapter - PostgreSQL Database Adapter for Markov Chain Models

This module provides a dedicated PostgreSQL adapter for Markov Chain models,
abstracting database operations away from the core model logic. It handles:
- Database configuration loading
- Connection pooling
- Table creation and setup
- CRUD operations for Markov Chain transitions and states

By centralizing database operations, this adapter improves code organization and maintainability
while allowing the Markov Chain model to focus on its core functionality.
"""

import os
import sys
import yaml
import logging
from psycopg2 import pool
from psycopg2.extras import execute_values


class MarkovChainPostgreSqlAdapter:
    """
    PostgreSQL adapter for Markov Chain models.
    
    This adapter encapsulates all database operations needed by the Markov Chain model,
    providing a clean interface for database interactions.
    """

    def __init__(self, environment="development", logger=None, db_config=None):
        """
        Initialize the PostgreSQL adapter.

        Args:
            environment (str): Environment setting ('development' or 'test')
            logger: Logger instance for logging database operations
            db_config (dict, optional): PostgreSQL configuration dictionary
        """
        self.environment = environment
        self.logger = logger
        self.conn_pool = None
        self.table_prefix = f"markov_{self.environment}"


        self.is_available = False
        
        # Get database configuration
        self.db_config = db_config or self.load_db_config()
        
        # Initialize connection pool if configuration is available
        if self.db_config:
            self.is_available = self._initialize_connection_pool()
        else:
            if self.logger:
                self.logger.warning("No database configuration available, adapter will not be usable", extra={
                    "metrics": {"environment": self.environment}
                })
    
    def load_db_config(self):
        """
        Load database configuration from YAML files.

        Returns:
            dict: Database configuration or None if not found
        """
        # Determine project root for config paths
        # Try to find project root by looking for configs directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Go up directory levels until we find the configs directory or reach root
        while project_root != os.path.dirname(project_root):  # Stop at filesystem root
            if os.path.exists(os.path.join(project_root, "configs")):
                break
            project_root = os.path.dirname(project_root)
        
        config_dir = os.path.join(project_root, "configs")
        
        # Try to get environment-specific configuration first
        env_config_path = os.path.join(config_dir, f"database_{self.environment}.yaml")
        
        # Fallback to default configuration
        default_config_path = os.path.join(config_dir, "database.yaml")
        
        # First check environment-specific config
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    
                    if self.logger:
                        self.logger.info("Database config loaded", extra={
                            "metrics": {
                                "config_path": env_config_path,
                                "environment": self.environment,
                            }
                        })
                    return config
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Error loading database config from {env_config_path}: {e}"
                    )
                
        # Then check default config
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    
                    if self.logger:
                        self.logger.info("Database config loaded (default)", extra={
                            "metrics": {
                                "config_path": default_config_path,
                                "environment": self.environment,
                            }
                        })
                    return config
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Error loading database config from {default_config_path}: {e}"
                    )
                    
        if self.logger:
            self.logger.warning("No database configuration found", extra={
                "metrics": {"environment": self.environment}
            })
        return None
    
    def _initialize_connection_pool(self):
        """
        Initialize the database connection pool based on configuration.
        
        Returns:
            bool: True if connection pool was successfully initialized, False otherwise
        """
        try:
            if not self.db_config:
                if self.logger:
                    self.logger.warning("Cannot initialize connection pool - no config provided")
                return False
                
            # Validate required configuration parameters
            required_params = ['host', 'dbname', 'user']
            for param in required_params:
                if param not in self.db_config:
                    if self.logger:
                        self.logger.warning(f"Missing required database parameter: {param}")
                    return False
            
            self.conn_pool = pool.SimpleConnectionPool(
                1,  # min connections
                10,  # max connections
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                dbname=self.db_config.get("dbname", "markov_chain"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", ""),
            )
            
            # Test connection and set up database
            conn = self.get_connection()
            if conn:
                self.setup_database(conn)
                self.return_connection(conn)
                
                if self.logger:
                    self.logger.info("Database connection established", extra={
                        "metrics": {
                            "host": self.db_config.get("host", "localhost"),
                            "port": self.db_config.get("port", 5432),
                            "dbname": self.db_config.get("dbname", "markov_chain"),
                            "environment": self.environment,
                        }
                    })
                return True
            else:
                if self.logger:
                    self.logger.warning("Failed to get connection from pool")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.warning("Database connection failed", extra={
                    "metrics": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "fallback": "in-memory storage"
                    }
                })
            return False
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            connection: Database connection or None if pool not available
        """
        if self.conn_pool:
            try:
                return self.conn_pool.getconn()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error getting connection from pool: {e}")
                return None
        return None
    
    def return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: The connection to return to the pool
        """
        if self.conn_pool and conn:
            try:
                self.conn_pool.putconn(conn)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error returning connection to pool: {e}")
    
    def is_usable(self):
        """
        Check if this adapter is usable (properly configured and connected).
        
        Returns:
            bool: True if the adapter can be used, False otherwise
        """
        return self.is_available and self.conn_pool is not None
    
    def setup_database(self, conn=None):
        """
        Set up the necessary database tables and indexes.
        
        Args:
            conn: Database connection (optional, will get one if not provided)
            
        Returns:
            bool: True if setup was successful, False otherwise
        """
        close_conn = False
        if not conn:
            conn = self.get_connection()
            close_conn = True
            
        if not conn:
            return False

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
                    (f"{self.table_prefix}_transitions",),
                )

                table_exists = cur.fetchone()[0]

                if not table_exists:
                    if self.logger:
                        self.logger.info(
                            f"Creating database schema for {self.environment} environment"
                        )
                else:
                    if self.logger:
                        self.logger.info(
                            f"Database schema for {self.environment} environment already exists"
                        )

                # Create transitions table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_transitions (
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
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_transitions_state_ngram 
                    ON {self.table_prefix}_transitions(state, n_gram)
                """
                )

                # Create total counts table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_prefix}_total_counts (
                        state TEXT,
                        count INTEGER,
                        n_gram INTEGER,
                        PRIMARY KEY (state, n_gram)
                    )
                """
                )

            conn.commit()
            if self.logger:
                self.logger.info(
                    f"Database setup complete for {self.environment} environment")
            
            return True

        except Exception as e:
            conn.rollback()
            if self.logger:
                self.logger.error(f"Error setting up database: {e}")
            return False
            
        finally:
            if close_conn:
                self.return_connection(conn)
    
    def insert_transitions_batch(self, batch, n_gram):
        """
        Insert a batch of transitions into the database.
        
        Args:
            batch (list): List of (state, next_word, count) tuples to insert
            n_gram (int): The n-gram size for these transitions
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if not conn:
            if self.logger:
                self.logger.warning("Failed to get connection for batch insert")
            return False
            
        try:
            # Add n_gram to each transition tuple
            augmented_batch = []
            for transition in batch:
                if len(transition) == 3:  # (state, next_word, count)
                    augmented_batch.append(transition + (n_gram,))
                else:  # Assume it already has n_gram
                    augmented_batch.append(transition)
                    
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_prefix}_transitions (state, next_word, count, n_gram) 
                    VALUES %s
                    ON CONFLICT (state, next_word, n_gram) 
                    DO UPDATE SET count = {self.table_prefix}_transitions.count + EXCLUDED.count
                """,
                    augmented_batch,
                )
                
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            if self.logger:
                self.logger.error(f"Error inserting transitions batch: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def update_total_counts(self, n_gram):
        """
        Update total counts table based on transitions.
        
        Args:
            n_gram (int): The n-gram size to update counts for
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if not conn:
            return False
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_prefix}_total_counts (state, count, n_gram)
                    SELECT state, SUM(count), n_gram 
                    FROM {self.table_prefix}_transitions 
                    WHERE n_gram = %s
                    GROUP BY state, n_gram
                    ON CONFLICT (state, n_gram) 
                    DO UPDATE SET count = EXCLUDED.count
                """,
                    (n_gram,),
                )
                
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            if self.logger:
                self.logger.error(f"Error updating total counts: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def increment_transition(self, state, next_word, count, n_gram):
        """
        Increment a single transition count in the database.
        
        Args:
            state (str): The state (word or space-separated n-gram)
            next_word (str): The next word in the transition
            count (int): The count to increment by
            n_gram (int): The n-gram size
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.insert_transitions_batch([(state, next_word, count)], n_gram)
    
    def clear_tables(self, n_gram):
        """
        Clear all data for a specific n-gram from the tables.
        
        Args:
            n_gram (int): The n-gram size to clear data for
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if not conn:
            return False
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self.table_prefix}_transitions 
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                cur.execute(
                    f"""
                    DELETE FROM {self.table_prefix}_total_counts 
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                
            conn.commit()
            
            if self.logger:
                self.logger.info(f"Cleared tables for n_gram {n_gram}", extra={
                    "metrics": {"environment": self.environment, "n_gram": n_gram}
                })
                
            return True
            
        except Exception as e:
            conn.rollback()
            if self.logger:
                self.logger.error(f"Error clearing tables: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def get_total_count(self, state, n_gram):
        """
        Get the total count for a state.
        
        Args:
            state (str): The state to get the count for
            n_gram (int): The n-gram size
            
        Returns:
            int: The total count or 0 if not found
        """
        conn = self.get_connection()
        if not conn:
            return 0
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT count FROM {self.table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """,
                    (state, n_gram),
                )
                
                result = cur.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting total count: {e}")
            return 0
            
        finally:
            self.return_connection(conn)
    
    def has_transitions(self, n_gram):
        """
        Check if there are any transitions for this n-gram.
        
        Args:
            n_gram (int): The n-gram size to check
            
        Returns:
            bool: True if transitions exist, False otherwise
        """
        conn = self.get_connection()
        if not conn:
            return False
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT EXISTS(
                        SELECT 1 FROM {self.table_prefix}_transitions 
                        WHERE n_gram = %s 
                        LIMIT 1
                    )
                """,
                    (n_gram,),
                )
                
                result = cur.fetchone()
                return result[0] if result else False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking transitions: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def get_random_state(self, n_gram):
        """
        Get a random state from the database.
        
        Args:
            n_gram (int): The n-gram size
            
        Returns:
            str or tuple: A random state or None if no states found
        """
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT state FROM {self.table_prefix}_transitions
                    WHERE n_gram = %s
                    GROUP BY state
                    ORDER BY RANDOM()
                    LIMIT 1
                """,
                    (n_gram,),
                )
                
                result = cur.fetchone()
                if not result:
                    return None
                    
                state = result[0]
                
                # Return tuple for multi-word states
                if n_gram > 1:
                    return tuple(state.split())
                return state
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting random state: {e}")
            return None
            
        finally:
            self.return_connection(conn)
    
    def check_state_exists(self, state, n_gram):
        """
        Check if a state exists in the database.
        
        Args:
            state: The state to check (string or tuple)
            n_gram (int): The n-gram size
            
        Returns:
            bool: True if the state exists, False otherwise
        """
        conn = self.get_connection()
        if not conn:
            return False
            
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
                        SELECT 1 FROM {self.table_prefix}_transitions
                        WHERE state = %s AND n_gram = %s
                        LIMIT 1
                    )
                """,
                    (db_state, n_gram),
                )
                
                result = cur.fetchone()
                return result[0] if result else False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking state existence: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def predict_next_word(self, current_state, n_gram):
        """
        Get probability distribution for next words from a state.
        
        Args:
            current_state: The current state (string or tuple)
            n_gram (int): The n-gram size
            
        Returns:
            dict: Dictionary mapping next words to probabilities
        """
        conn = self.get_connection()
        if not conn:
            return {}
            
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
                    SELECT count FROM {self.table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, n_gram),
                )
                
                result = cur.fetchone()
                if not result:
                    return {}
                    
                total_count = result[0]
                
                # Get all transitions with their counts
                cur.execute(
                    f"""
                    SELECT next_word, count 
                    FROM {self.table_prefix}_transitions
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, n_gram),
                )
                
                transitions = cur.fetchall()
                if not transitions:
                    return {}
                    
                # Calculate probabilities
                return {word: count/total_count for word, count in transitions}
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error predicting next word: {e}")
            return {}
            
        finally:
            self.return_connection(conn)
    
    def get_transition_probability(self, current_state, next_word, n_gram):
        """
        Get the probability of a specific transition.
        
        Args:
            current_state: The current state (string or tuple)
            next_word: The next word
            n_gram (int): The n-gram size
            
        Returns:
            float: The transition probability (0-1)
        """
        conn = self.get_connection()
        if not conn:
            return 0.0
            
        try:
            # Convert tuple to string for database query
            if isinstance(current_state, tuple):
                db_state = " ".join(current_state)
            else:
                db_state = str(current_state)
                
            with conn.cursor() as cur:
                # Get the count for this specific transition
                cur.execute(
                    f"""
                    SELECT count FROM {self.table_prefix}_transitions 
                    WHERE state = %s AND next_word = %s AND n_gram = %s
                """,
                    (db_state, next_word, n_gram),
                )
                
                transition_result = cur.fetchone()
                if not transition_result:
                    return 0.0
                    
                transition_count = transition_result[0]
                
                # Get the total count for this state
                cur.execute(
                    f"""
                    SELECT count FROM {self.table_prefix}_total_counts 
                    WHERE state = %s AND n_gram = %s
                """,
                    (db_state, n_gram),
                )
                
                total_result = cur.fetchone()
                if not total_result:
                    return 0.0
                    
                total_count = total_result[0]
                
                # Calculate probability
                return transition_count / total_count
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating transition probability: {e}")
            return 0.0
            
        finally:
            self.return_connection(conn)
    
    def get_model_statistics(self, n_gram):
        """
        Get comprehensive statistics about the model.
        
        Args:
            n_gram (int): The n-gram size
            
        Returns:
            dict: Dictionary of statistics
        """
        conn = self.get_connection()
        if not conn:
            return {"error": "No database connection available"}
            
        try:
            stats = {
                "storage_type": "database",
                "environment": self.environment,
                "n_gram": n_gram,
            }
            
            with conn.cursor() as cur:
                # Get total number of transitions
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.table_prefix}_transitions 
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                stats["transitions_count"] = cur.fetchone()[0]
                
                # Get unique states count
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.table_prefix}_total_counts
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                stats["states_count"] = cur.fetchone()[0]
                
                # Get top 5 most common state transitions
                cur.execute(
                    f"""
                    SELECT state, next_word, count 
                    FROM {self.table_prefix}_transitions
                    WHERE n_gram = %s
                    ORDER BY count DESC
                    LIMIT 5
                """,
                    (n_gram,),
                )
                
                top_transitions = cur.fetchall()
                stats["top_transitions"] = [
                    {"state": state, "next_word": next_word, "count": count}
                    for state, next_word, count in top_transitions
                ]
                
                # Get vocabulary size (unique words)
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT next_word) 
                    FROM {self.table_prefix}_transitions
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                stats["vocabulary_size"] = cur.fetchone()[0]
                
                # Get average transitions per state
                cur.execute(
                    f"""
                    SELECT AVG(transition_count) FROM (
                        SELECT state, COUNT(*) as transition_count
                        FROM {self.table_prefix}_transitions
                        WHERE n_gram = %s
                        GROUP BY state
                    ) as state_counts
                """,
                    (n_gram,),
                )
                stats["avg_transitions_per_state"] = cur.fetchone()[0]
                
                # Get database size
                cur.execute(
                    f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{self.table_prefix}_transitions'))
                """
                )
                stats["transitions_table_size"] = cur.fetchone()[0]
                
                # Get state distribution metrics
                cur.execute(
                    f"""
                    SELECT 
                        MIN(count) as min_count,
                        MAX(count) as max_count,
                        AVG(count) as avg_count,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY count) as median_count
                    FROM {self.table_prefix}_total_counts
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                
                dist_metrics = cur.fetchone()
                stats["state_count_distribution"] = {
                    "min": dist_metrics[0],
                    "max": dist_metrics[1],
                    "avg": dist_metrics[2],
                    "median": dist_metrics[3],
                }
                
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting model statistics: {e}")
            return {"error": str(e)}
            
        finally:
            self.return_connection(conn)
    
    def find_high_probability_sequences(self, n_gram, length=3, top_n=10):
        """
        Find high probability sequences in the database.
        
        Args:
            n_gram (int): The n-gram size
            length (int): The length of sequences to find
            top_n (int): Number of top sequences to return
            
        Returns:
            list: List of (sequence, probability) tuples
        """
        conn = self.get_connection()
        if not conn:
            return []
            
        try:
            sequences_found = []
            
            with conn.cursor() as cur:
                # Get sample of states to use as starting points
                cur.execute(
                    f"""
                    SELECT state 
                    FROM {self.table_prefix}_total_counts
                    WHERE n_gram = %s
                    ORDER BY count DESC
                    LIMIT 100
                """,
                    (n_gram,),
                )
                
                start_states = [row[0] for row in cur.fetchall()]
                
                # For each starting state, find sequences by following transitions
                for db_state in start_states:
                    # Convert string state to appropriate format
                    if n_gram > 1:
                        start_state = tuple(db_state.split())
                    else:
                        start_state = db_state
                        
                    # Try to build sequences starting from this state
                    current_state = start_state
                    
                    for _ in range(min(top_n, 10)):  # Sample sequences from each start state
                        if n_gram == 1:
                            sequence = [current_state]
                        else:
                            sequence = list(current_state)
                            
                        current_probability = 1.0
                        
                        # Build a sequence of required length
                        for _ in range(length - len(sequence)):
                            # Get possible next words and their probabilities
                            if n_gram > 1:
                                db_current_state = " ".join(current_state)
                            else:
                                db_current_state = current_state
                                
                            cur.execute(
                                f"""
                                SELECT next_word, count 
                                FROM {self.table_prefix}_transitions
                                WHERE state = %s AND n_gram = %s
                                ORDER BY count DESC
                                LIMIT 10
                            """,
                                (db_current_state, n_gram),
                            )
                            
                            next_words = cur.fetchall()
                            if not next_words:
                                break
                                
                            # Get total count for current state
                            cur.execute(
                                f"""
                                SELECT count 
                                FROM {self.table_prefix}_total_counts
                                WHERE state = %s AND n_gram = %s
                            """,
                                (db_current_state, n_gram),
                            )
                            
                            total_result = cur.fetchone()
                            if not total_result:
                                break
                                
                            total = total_result[0]
                            
                            # Choose next word based on probabilities
                            next_word_items = next_words
                            
                            # For simplicity, select the highest probability next word
                            # In a full implementation, you would select randomly
                            next_word, count = next_word_items[0]
                            transition_prob = count / total
                            
                            sequence.append(next_word)
                            current_probability *= transition_prob
                            
                            # Update current state by shifting words for n-gram
                            if n_gram > 1:
                                if len(sequence) >= n_gram:
                                    current_state = tuple(sequence[-(n_gram):])
                            else:
                                current_state = next_word
                                
                        # If we successfully built a sequence of required length
                        if len(sequence) >= length:
                            sequences_found.append((" ".join(sequence), current_probability))
                            
            # Sort by probability (descending) and take top_n
            return sorted(sequences_found, key=lambda x: x[1], reverse=True)[:top_n]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finding high probability sequences: {e}")
            return []
            
        finally:
            self.return_connection(conn)
            
    def extract_vocabulary(self, n_gram):
        """
        Extract vocabulary (all unique words) from the database.
        
        Args:
            n_gram (int): The n-gram size
            
        Returns:
            set: Set of unique words
        """
        conn = self.get_connection()
        if not conn:
            return set()
            
        try:
            vocab = set()
            
            with conn.cursor() as cur:
                # First get all next_words as they're definitely single words
                cur.execute(
                    f"""
                    SELECT DISTINCT next_word 
                    FROM {self.table_prefix}_transitions
                    WHERE n_gram = %s
                """,
                    (n_gram,),
                )
                
                for (word,) in cur.fetchall():
                    vocab.add(word)
                    
                # Then get state words (for n_gram=1 these are single words)
                if n_gram == 1:
                    cur.execute(
                        f"""
                        SELECT DISTINCT state 
                        FROM {self.table_prefix}_transitions
                        WHERE n_gram = %s
                    """,
                        (n_gram,),
                    )
                    
                    for (word,) in cur.fetchall():
                        vocab.add(word)
                else:
                    # For n>1, split multi-word states into individual words
                    cur.execute(
                        f"""
                        SELECT DISTINCT state 
                        FROM {self.table_prefix}_transitions
                        WHERE n_gram = %s
                    """,
                        (n_gram,),
                    )
                    
                    for (state,) in cur.fetchall():
                        for word in state.split():
                            vocab.add(word)
                            
            return vocab
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting vocabulary: {e}")
            return set()
            
        finally:
            self.return_connection(conn)
    
    def close_connections(self):
        """Close all connections in the pool."""
        if self.conn_pool:
            try:
                self.conn_pool.closeall()
                if self.logger:
                    self.logger.info("Database connections closed")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error closing database connections: {e}")