#!/usr/bin/env python3
"""
ExportMarkovChain - Training and exporting Markov Chain models to ONNX format

This module defines a class that handles the complete lifecycle of Markov Chain models:
1. Training on large JSON text datasets (specifically Wikipedia text data)
2. Optimizing memory usage and database connections for large datasets
3. Exporting trained models to ONNX format for cross-platform compatibility

The class is designed to be used as a standalone utility for creating pre-trained
Markov Chain models that can be deployed in various environments.

Example usage:
    python train_and_export.py

Dependencies:
    - MarkovChain class from the markov_chain module
    - JSON logger from the json_logger module
    - Database configuration from configs/database.yaml
"""

from utils.loggers.json_logger import get_logger, log_json
from models.production_models.markov_chain.markov_chain import MarkovChain
import os
import json
import glob
import shutil
import logging
import multiprocessing
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

# Import MarkovChain class

# Import JSON logger


class ExportMarkovChain:
    """
    Class for training and exporting MarkovChain models to ONNX format.

    This class handles the entire lifecycle of training a MarkovChain model on a large
    text dataset (Wikipedia JSON dump) and exporting it to ONNX format. It's optimized
    for handling large datasets (8GB+) with database storage and parallel processing.

    Attributes:
        data_dir (str): Path to directory containing JSON text data files
        n_gram (int): The n-gram size for the Markov Chain model
        memory_threshold (int): Threshold for keeping transitions in memory vs. database
        db_config (dict): Database configuration for PostgreSQL
        environment (str): Environment setting ('development' or 'test')
        log_path (str): Path to the log file
        logger (Logger): JSON logger instance
    """

    def __init__(
        self,
        data_dir="data/text_data/wikipedia_text_json_dump",
        n_gram=2,
        memory_threshold=1000000,  # Set high threshold for memory optimization with 8GB dataset
        environment="development",
        log_path="logs/train_and_export.log"
    ):
        """
        Initialize the ExportMarkovChain class.

        Args:
            data_dir (str): Path to directory containing JSON text data files
            n_gram (int): The n-gram size for the Markov Chain model
            memory_threshold (int): Threshold for memory vs. database storage
            environment (str): Environment setting ('development' or 'test')
            log_path (str): Path to the log file
        """
        # Base attributes
        self.data_dir = os.path.join(project_root, data_dir)
        self.n_gram = n_gram
        self.memory_threshold = memory_threshold
        self.environment = environment
        self.db_config = self._load_db_config()

        # Ensure log directory exists
        self.log_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), log_path))
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Clear previous log file if it exists
        if os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("")  # Clear the file

        # Configure the logger
        if get_logger:
            self.logger = get_logger(
                "export_markov_chain", log_file=self.log_path)
        else:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                filename=self.log_path,
                filemode='a'
            )
            self.logger = logging.getLogger("export_markov_chain")

        # Log initialization
        self.logger.info("ExportMarkovChain initialized")
        if log_json:
            log_json(
                self.logger,
                "ExportMarkovChain initialized",
                {
                    "data_dir": self.data_dir,
                    "n_gram": self.n_gram,
                    "memory_threshold": self.memory_threshold,
                    "environment": self.environment,
                    "db_config_host": self.db_config.get("host") if self.db_config else None,
                    "db_config_dbname": self.db_config.get("dbname") if self.db_config else None,
                    "log_path": self.log_path,
                    "timestamp": datetime.now().isoformat()
                }
            )

    def _setup_logging(self):
        """Set up JSON logging with file output."""
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Clear previous log file if it exists
        if os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("")  # Clear the file

        # Configure the logger
        if get_logger:
            self.logger = get_logger(
                "export_markov_chain", log_file=self.log_path)
        else:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                filename=self.log_path,
                filemode='a'
            )
            self.logger = logging.getLogger("export_markov_chain")

    def _load_db_config(self):
        """
        Load database configuration from file.

        Returns:
            dict: Database configuration or None if not found
        """
        # Base config paths
        config_dir = os.path.join(project_root, "configs")

        # Try to get environment-specific configuration first
        env_config_path = os.path.join(
            config_dir, f"database_{self.environment}.yaml")

        # Fallback to default configuration
        default_config_path = os.path.join(config_dir, "database.yaml")

        # First check environment-specific config
        if os.path.exists(env_config_path):
            try:
                with open(env_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    print(f"Loaded database config from {env_config_path}")
                    return config
            except Exception as e:
                print(
                    f"Error loading database config from {env_config_path}: {e}")

        # Then check default config
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, "r") as f:
                    config = yaml.safe_load(f)
                    print(f"Loaded database config from {default_config_path}")
                    return config
            except Exception as e:
                print(
                    f"Error loading database config from {default_config_path}: {e}")

        print("No database configuration found")
        return None

    def check_data_directory(self):
        """
        Check if the data directory exists and contains JSON files.

        Returns:
            bool: True if directory exists and contains JSON files, False otherwise
        """
        if not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory not found: {self.data_dir}")
            if log_json:
                log_json(
                    self.logger,
                    "Data directory not found",
                    {"data_dir": self.data_dir, "error": "Directory does not exist"}
                )
            return False

        # Check if there are JSON files in the directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        if not json_files:
            self.logger.error(
                f"No JSON files found in data directory: {self.data_dir}")
            if log_json:
                log_json(
                    self.logger,
                    "No JSON files found",
                    {"data_dir": self.data_dir, "error": "No JSON files in directory"}
                )
            return False

        self.logger.info(
            f"Found {len(json_files)} JSON files in {self.data_dir}")
        if log_json:
            log_json(
                self.logger,
                "JSON files found",
                {"data_dir": self.data_dir, "file_count": len(json_files)}
            )
        return True

    def load_json_texts(self):
        """
        Load and extract text content from all JSON files in the data directory.

        This method handles large datasets by processing files one at a time.

        Returns:
            list: List of text strings extracted from JSON files
        """
        self.logger.info(f"Loading texts from JSON files in {self.data_dir}")

        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        texts = []

        file_count = len(json_files)
        processed_count = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract text - assuming JSON structure has a 'text' field
                # Adjust this based on your actual JSON structure
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                elif isinstance(data, dict) and 'text' in data:
                    texts.append(data['text'])

                processed_count += 1

                # Log progress periodically with bold formatting
                if processed_count % 10 == 0 or processed_count == file_count:
                    progress_message = f"Processed {processed_count}/{file_count} JSON files - {(processed_count/file_count)*100:.1f}%"
                    self.logger.info(progress_message)
                    # Print bold text to terminal
                    print(f"\033[1m{progress_message}\033[0m")

                    if log_json:
                        log_json(
                            self.logger,
                            "JSON loading progress",
                            {
                                "processed": processed_count,
                                "total": file_count,
                                "percentage": f"{(processed_count/file_count)*100:.1f}%"
                            }
                        )

            except Exception as e:
                self.logger.error(
                    f"Error processing JSON file {json_file}: {e}")
                if log_json:
                    log_json(
                        self.logger,
                        "JSON processing error",
                        {"file": json_file, "error": str(
                            e), "error_type": type(e).__name__}
                    )

        completion_message = f"Loaded {len(texts)} text entries from {processed_count} JSON files"
        self.logger.info(completion_message)
        # Print bold completion message
        print(f"\033[1m{completion_message}\033[0m")

        if log_json:
            log_json(
                self.logger,
                "JSON loading completed",
                {"text_count": len(texts), "file_count": processed_count}
            )

        return texts

    def train_model(self, texts):
        """
        Train the MarkovChain model on the provided texts.

        Args:
            texts (list): List of text strings to train on

        Returns:
            MarkovChain: Trained model
        """
        self.logger.info("Initializing MarkovChain model for training")

        # Optimize DB connection pool for large dataset
        if self.db_config:
            # Add connection pool optimization
            self.db_config['min_connections'] = 5
            # Higher value for parallel processing
            self.db_config['max_connections'] = 20

        # Initialize model with optimized parameters and pass the logger
        model = MarkovChain(
            n_gram=self.n_gram,
            memory_threshold=self.memory_threshold,
            db_config=self.db_config,
            environment=self.environment,
            logger=self.logger  # Pass the logger instance directly
        )

        # Calculate optimal number of jobs for parallel training
        cpu_count = multiprocessing.cpu_count()

        # Begin preprocessing notification
        preprocess_start_message = f"Starting text preprocessing with {cpu_count} CPUs"
        self.logger.info(preprocess_start_message)
        # Print bold preprocessing start message
        print(f"\033[1m{preprocess_start_message}\033[0m")

        # Log preprocessing details
        sample_text = texts[0][:100] + \
            "..." if texts and len(texts[0]) > 100 else "No text available"
        preprocess_details = (
            f"Text sample before preprocessing: '{sample_text}'\n"
            f"Total texts to preprocess: {len(texts)}\n"
            f"Average text length: {sum(len(t) for t in texts) // len(texts) if texts else 0} characters"
        )
        self.logger.info(preprocess_details)

        # Print preprocessing details in bold blue
        print(f"\033[1;34mPreprocessing details:\033[0m")
        for line in preprocess_details.split('\n'):
            print(f"\033[1;34m- {line}\033[0m")

        self.logger.info("Preprocessing started", extra={
            "metrics": {
                "text_count": len(texts),
                "sample_text": sample_text,
                "average_length": sum(len(t) for t in texts) // len(texts) if texts else 0,
                "cpu_count": cpu_count
            }
        })

        training_start_message = f"Starting parallel training with {cpu_count} CPUs"
        self.logger.info(training_start_message)
        # Print bold start message
        print(f"\033[1m{training_start_message}\033[0m")

        self.logger.info("Training started", extra={
            "metrics": {
                "method": "train_parallel",
                "n_gram": self.n_gram,
                "text_count": len(texts),
                "cpu_count": cpu_count,
                "environment": self.environment,
                "memory_threshold": self.memory_threshold
            }
        })

        # Set up a progress listener for the training process
        def training_progress_listener():
            """Display enhanced progress in the terminal while training is running"""
            import threading
            import time
            import psutil

            stop_event = threading.Event()
            last_processed = 0
            progress_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']

            def progress_thread():
                nonlocal last_processed
                i = 0

                # Initial delay before starting to show progress animation
                time.sleep(2)

                print(f"\033[1m⏳ Training in progress...\033[0m")

                while not stop_event.is_set():
                    # Get current memory usage
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)

                    # Show spinning animation and memory usage
                    char = progress_chars[i % len(progress_chars)]
                    print(
                        f"\r\033[1;33m{char} Training in progress... Memory usage: {memory_mb:.1f}MB\033[0m", end='', flush=True)

                    time.sleep(0.5)
                    i += 1

                print(
                    "\r\033[1;32mTraining completed!                                        \033[0m")

            thread = threading.Thread(target=progress_thread)
            thread.daemon = True
            thread.start()

            return stop_event

        # Start the training progress animation
        progress_stop = training_progress_listener()

        # Use parallel training with maximum CPU utilization
        start_time = datetime.now()

        try:
            # Call train_parallel with a progress callback
            training_results = model.train_parallel(
                texts=texts,
                clear_previous=True,  # Clear any previous training data
                preprocess=True,      # Apply preprocessing to texts
                n_jobs=cpu_count      # Use all available CPUs
            )

            # Process the returned training results
            if isinstance(training_results, dict):
                # Enhanced terminal output with detailed statistics
                print("\n\033[1;32mTraining Statistics:\033[0m")
                print(
                    f"\033[1;32m- Training time: {training_results.get('training_time', 0):.2f} seconds\033[0m")
                print(
                    f"\033[1;32m- Total texts processed: {training_results.get('total_texts', 0)}\033[0m")
                print(
                    f"\033[1;32m- States (unique contexts): {training_results.get('total_states', 0)}\033[0m")
                print(
                    f"\033[1;32m- Transitions: {training_results.get('total_transitions', 0)}\033[0m")
                print(
                    f"\033[1;32m- Processing speed: {training_results.get('transitions_per_second', 0):.1f} transitions/second\033[0m")

                # Log detailed statistics
                self.logger.info("Training results", extra={
                    "metrics": training_results
                })

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            print(f"\033[1;31mError during training: {e}\033[0m")
            self.logger.error("Training error", extra={
                "metrics": {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            })
        finally:
            # Stop the progress animation
            progress_stop.set()

        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()

        completion_message = f"Training completed in {training_time:.2f} seconds"
        self.logger.info(completion_message)
        # Print bold completion message
        print(f"\033[1m{completion_message}\033[0m")

        # Get storage information
        if model.using_db:
            storage_message = "Model stored in database (PostgreSQL)"
        else:
            storage_message = f"Model stored in memory with {len(model.transitions)} states"

        print(f"\033[1;36m{storage_message}\033[0m")

        self.logger.info("Training completed", extra={
            "metrics": {
                "training_time_seconds": training_time,
                "n_gram": self.n_gram,
                "text_count": len(texts),
                "environment": self.environment,
                "storage": "database" if model.using_db else "memory"
            }
        })

        return model

    def export_model(self, model):
        """
        Export the trained model to ONNX format.

        Args:
            model (MarkovChain): Trained MarkovChain model

        Returns:
            str: Path to the exported model file, or None if export failed
        """
        # Define the export filepath
        export_dir = os.path.dirname(os.path.abspath(__file__))
        export_filepath = os.path.join(
            export_dir, "model0.0.1-pretraining.onnx")

        export_start_message = f"Exporting model to ONNX format: {export_filepath}"
        self.logger.info(export_start_message)
        # Print bold export start message
        print(f"\033[1m{export_start_message}\033[0m")

        if log_json:
            log_json(
                self.logger,
                "ONNX export started",
                {"filepath": export_filepath, "n_gram": self.n_gram}
            )

        # Export to ONNX format
        start_time = datetime.now()
        success = model.export_to_onnx(export_filepath)
        export_time = (datetime.now() - start_time).total_seconds()

        if success:
            completion_message = f"Successfully exported model to {export_filepath} in {export_time:.2f} seconds"
            self.logger.info(completion_message)
            # Print bold success message
            print(f"\033[1m{completion_message}\033[0m")

            if log_json:
                log_json(
                    self.logger,
                    "ONNX export completed",
                    {
                        "filepath": export_filepath,
                        "export_time_seconds": export_time,
                        "file_size_bytes": os.path.getsize(export_filepath) if os.path.exists(export_filepath) else 0
                    }
                )
            return export_filepath
        else:
            error_message = "Failed to export model to ONNX format"
            self.logger.error(error_message)
            # Print bold error message
            print(f"\033[1m{error_message}\033[0m")

            if log_json:
                log_json(
                    self.logger,
                    "ONNX export failed",
                    {"filepath": export_filepath}
                )
            return None

    def run(self):
        """
        Run the complete model training and export process.

        This method orchestrates the entire workflow:
        1. Check data directory
        2. Load JSON text data
        3. Train the model
        4. Export the model to ONNX format

        Returns:
            bool: True if process completed successfully, False otherwise
        """
        start_message = "Starting ExportMarkovChain process"
        self.logger.info(start_message, extra={
            "metrics": {"timestamp": datetime.now().isoformat()}
        })
        # Print bold start message
        print(f"\033[1m{start_message}\033[0m")

        # Check data directory
        dir_check_message = f"Checking data directory: {self.data_dir}"
        print(f"\033[1m{dir_check_message}\033[0m")
        if not self.check_data_directory():
            error_message = f"Process aborted: Data directory check failed - {self.data_dir}"
            self.logger.error(error_message, extra={
                "metrics": {"reason": "Data directory check failed"}
            })
            # Print bold error message
            print(f"\033[1m{error_message}\033[0m")
            return False

        # Load texts from JSON files
        load_message = "Loading JSON text data..."
        print(f"\033[1m{load_message}\033[0m")
        texts = self.load_json_texts()
        if not texts:
            error_message = "Process aborted: No texts loaded from JSON files"
            self.logger.error(error_message, extra={
                "metrics": {"reason": "No texts loaded"}
            })
            # Print bold error message
            print(f"\033[1m{error_message}\033[0m")
            return False

        # Train the model
        model = self.train_model(texts)

        # Export the model to ONNX format
        export_path = self.export_model(model)

        if export_path and os.path.exists(export_path):
            success_message = "ExportMarkovChain process completed successfully"
            self.logger.info(success_message, extra={
                "metrics": {
                    "status": "success",
                    "export_path": export_path,
                    "model_size_mb": f"{os.path.getsize(export_path) / (1024 * 1024):.2f}",
                    "timestamp": datetime.now().isoformat()
                }
            })
            # Print bold success message with green color
            print(f"\033[1;32m{success_message}\033[0m")

            # Print model details
            model_size = os.path.getsize(
                export_path) / (1024 * 1024)  # Convert to MB
            model_details = f"Model saved to: {export_path} (Size: {model_size:.2f} MB)"
            print(f"\033[1m{model_details}\033[0m")

            return True
        else:
            error_message = "ExportMarkovChain process failed: Model export failed"
            self.logger.error(error_message, extra={
                "metrics": {
                    "status": "failed",
                    "reason": "Model export failed",
                    "timestamp": datetime.now().isoformat()
                }
            })
            # Print bold error message with red color
            print(f"\033[1;31m{error_message}\033[0m")

            return False


if __name__ == "__main__":
    """
    Execute the ExportMarkovChain process when script is run directly.
    """
    exporter = ExportMarkovChain()
    success = exporter.run()

    # Exit with appropriate status code
    sys.exit(0 if success else 1)
