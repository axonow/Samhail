#!/usr/bin/env python3
"""
Markov Chain Model Training and Export Script

This script trains a Markov chain model on various text datasets
and exports it for production use. It handles database storage,
model serialization, and system resource monitoring.
"""
import os
import sys
import time
import pickle
import yaml
import argparse
import concurrent.futures
from datetime import datetime

# Add project root to Python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from models.production_models.markov_chain.markov_chain import MarkovChain
from data_preprocessing.text_preprocessor import TextPreprocessor
from utils.loggers.json_logger import get_logger
from utils.system_monitoring import ResourceMonitor

class MarkovChainTrainer:
    """
    Handles training and exporting Markov Chain models for production use.

    This class manages the full pipeline of:
    1. Loading and preprocessing training data
    2. Training the Markov Chain model
    3. Exporting the model for production use
    4. Handling database storage and connections
    """

    def __init__(self, n_gram=2, environment="development", memory_threshold_mb=None,
                 memory_percentage=85, dataset_paths=None):
        """
        Initialize the trainer with specified parameters.

        Args:
            n_gram (int): The n-gram size for the Markov Chain model
            environment (str): Environment setting ('development', 'test', 'production')
            memory_threshold_mb (int, optional): Memory threshold in MB
            memory_percentage (int): Percentage of system memory to use if threshold not specified
            dataset_paths (list, optional): List of dataset file paths
        """
        self.n_gram = n_gram
        self.environment = environment

        # Set up logging to a file specific to this run
        self.log_dir = os.path.join(current_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Use a fixed log file path
        log_file = os.path.join(self.log_dir, "train_and_export.log")
        self.logger = get_logger(
            f"markov_train_{environment}", log_file=log_file)

        # Initialize resource monitor for tracking system resources
        self.resource_monitor = ResourceMonitor(
            logger=self.logger,
            memory_limit_mb=memory_threshold_mb,
            memory_limit_percentage=memory_percentage,
            monitoring_interval=5.0  # Log metrics every 5 seconds
        )

        # Log initialization with system info
        self.logger.info(f"MarkovChainTrainer initialized with n-gram={n_gram}", extra={
            "metrics": {
                "n_gram": n_gram,
                "environment": environment,
                "memory_threshold_mb": memory_threshold_mb,
                "memory_percentage": memory_percentage
            }
        })

        # Setup dataset paths
        if (dataset_paths):
            self.dataset_paths = dataset_paths
        else:
            self.dataset_paths = self._get_default_dataset_paths()

        # Load database configuration
        self.db_config = self._load_db_config()

        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()

        # Print initialization info
        print(
            f"\033[1m🚀 Initializing Markov Chain Trainer (n-gram={n_gram}, env={environment})\033[0m")
        print(
            f"\033[1m📊 Memory limit: {memory_threshold_mb if memory_threshold_mb else memory_percentage}% of available memory\033[0m")

    def _get_default_dataset_paths(self):
        """
        Get default dataset paths for training.

        Returns:
            list: Paths to default training datasets
        """
        # Default dataset paths
        datasets = [
            os.path.join(project_root, "csv_datasets", "dcat_train_data.csv"),
            os.path.join(project_root, "csv_datasets",
                         "reddit_social_media_comments.csv"),
            os.path.join(project_root, "csv_datasets",
                         "twitter_social_media_comments.csv"),
            os.path.join(project_root, "csv_datasets",
                         "markov_chain_impression_dataset.csv"),
        ]

        # Log available datasets
        self.logger.info("Default dataset paths loaded", extra={
            "metrics": {
                "dataset_count": len(datasets),
                "dataset_paths": datasets
            }
        })

        return datasets

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

    def _load_dataset(self, file_path):
        """
        Load and preprocess a dataset from file.

        Args:
            file_path (str): Path to the dataset file

        Returns:
            str: Preprocessed text content
        """
        start_time = time.time()

        self.logger.info(f"Loading dataset: {file_path}", extra={
            "metrics": {
                "file_path": file_path,
                "operation": "dataset_load"
            }
        })

        try:
            # Check file existence
            if not os.path.exists(file_path):
                self.logger.error(f"Dataset file not found: {file_path}")
                return None

            # Load based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            text_content = ""

            if file_ext == '.csv':
                import pandas as pd
                try:
                    # Try to load with different encoding options
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='latin-1')

                    # Get text from the first column that has 'text' in its name
                    text_columns = [col for col in df.columns if 'text' in col.lower()
                                    or 'content' in col.lower()
                                    or 'comment' in col.lower()]

                    if text_columns:
                        # Use the first matching column
                        text_content = " ".join(
                            df[text_columns[0]].astype(str).tolist())
                    else:
                        # Fall back to the first column
                        first_col = df.columns[0]
                        text_content = " ".join(
                            df[first_col].astype(str).tolist())

                except Exception as e:
                    self.logger.error(f"Error processing CSV file: {e}", extra={
                        "metrics": {
                            "error": str(e),
                            "file_path": file_path
                        }
                    })
                    return None

            elif file_ext == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text_content = f.read()

            else:
                self.logger.error(f"Unsupported file format: {file_ext}")
                return None

            # Preprocess the text
            preprocessed_text = self.preprocessor.preprocess(text_content)

            # Log completion
            execution_time = time.time() - start_time
            self.resource_monitor.log_progress(
                f"Dataset loaded and preprocessed: {file_path}",
                operation="dataset_load_complete",
                extra_metrics={
                    "file_path": file_path,
                    "original_size": len(text_content),
                    "preprocessed_size": len(preprocessed_text),
                    "execution_time": execution_time
                }
            )

            return preprocessed_text

        except Exception as e:
            self.logger.error(f"Error loading dataset {file_path}: {e}")
            return None

    def train_model(self):
        """
        Train the Markov Chain model on all datasets.

        Returns:
            MarkovChain: The trained model
        """
        start_time = time.time()
        self.resource_monitor.start()

        # Log training start
        self.resource_monitor.print_system_architecture()
        self.logger.info("Starting model training", extra={
            "metrics": {
                "n_gram": self.n_gram,
                "environment": self.environment,
                "dataset_count": len(self.dataset_paths)
            }
        })

        # Create model
        model = MarkovChain(
            n_gram=self.n_gram,
            logger=self.logger,
            memory_threshold=self.resource_monitor.memory_manager.memory_limit_mb,
            environment=self.environment
        )

        # Process each dataset
        for i, dataset_path in enumerate(self.dataset_paths):
            try:
                # Print progress
                print(
                    f"\033[1m\nProcessing dataset {i+1}/{len(self.dataset_paths)}: {os.path.basename(dataset_path)}\033[0m")

                # Load and preprocess dataset
                text_data = self._load_dataset(dataset_path)

                if text_data:
                    dataset_size_mb = len(text_data) / (1024 * 1024)
                    print(
                        f"\033[1m📄 Loaded {dataset_size_mb:.2f} MB of text\033[0m")

                    # Train model on this dataset
                    print(f"\033[1m🧠 Training model on dataset...\033[0m")
                    train_start = time.time()
                    # Already preprocessed above
                    model.train(text_data, preprocess=False)

                    # Log training metrics for this dataset
                    train_time = time.time() - train_start
                    self.resource_monitor.log_progress(
                        f"Trained on dataset: {dataset_path}",
                        progress_percent=(
                            (i+1) / len(self.dataset_paths)) * 100,
                        operation="dataset_training",
                        extra_metrics={
                            "dataset": os.path.basename(dataset_path),
                            "dataset_size_mb": dataset_size_mb,
                            "training_time": train_time
                        }
                    )

                # Monitor memory after each dataset and print resources
                self.resource_monitor.print_resource_usage(prefix="  ")
                print()  # Add newline after resource info

            except Exception as e:
                self.logger.error(
                    f"Error training on dataset {dataset_path}: {e}")
                print(f"\033[1;31mError training on dataset: {e}\033[0m")

        # Collect final training metrics
        training_time = time.time() - start_time
        transitions_count = sum(len(next_words) for next_words in model.transitions.values(
        )) if hasattr(model, 'transitions') else 0

        # Log training completion
        self.logger.info("Model training completed", extra={
            "metrics": {
                "training_time": training_time,
                "transitions_count": transitions_count,
                "states_count": len(model.transitions) if hasattr(model, 'transitions') else 0,
                "model_type": "markov_chain",
                "n_gram": self.n_gram
            }
        })

        # Stop resource monitoring
        self.resource_monitor.stop()

        print(
            f"\033[1m✅ Training completed in {training_time:.2f} seconds\033[0m")
        return model

    def export_model(self, model, export_path=None):
        """
        Export the trained model to disk.

        Args:
            model: The trained Markov Chain model
            export_path (str, optional): Path to save the model

        Returns:
            str: Path to the exported model file
        """
        start_time = time.time()

        if not export_path:
            # Create models directory if needed
            models_dir = os.path.join(
                project_root, "models", "production_models", "markov_chain", "exported_models")
            os.makedirs(models_dir, exist_ok=True)

            # Generate filename with timestamp and environment
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(
                models_dir, f"markov_chain_{self.n_gram}gram_{timestamp}_{self.environment}.pkl"
            )

        # Log export start
        self.logger.info("Exporting model to disk", extra={
            "metrics": {
                "export_path": export_path,
                "environment": self.environment,
                "model_type": "markov_chain",
                "n_gram": self.n_gram
            }
        })

        try:
            # Export the model using pickle
            with open(export_path, 'wb') as f:
                pickle.dump(model, f)

            # Get file size
            file_size = os.path.getsize(
                export_path) / (1024 * 1024)  # Size in MB

            # Log export completion
            export_time = time.time() - start_time
            self.logger.info("Model export completed", extra={
                "metrics": {
                    "export_path": export_path,
                    "file_size_mb": file_size,
                    "export_time": export_time
                }
            })

            print(
                f"\033[1m📦 Model exported to: {export_path} ({file_size:.2f} MB)\033[0m")
            return export_path

        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            print(f"\033[1;31mError exporting model: {e}\033[0m")
            return None

    def run_pipeline(self):
        """
        Run the full pipeline: train and export the model.

        Returns:
            tuple: (trained_model, export_path)
        """
        print("\033[1;32m" + "="*80 + "\033[0m")
        print("\033[1;32m🚀 Starting Markov Chain Training Pipeline\033[0m")
        print("\033[1;32m" + "="*80 + "\033[0m")

        # Train model
        model = self.train_model()

        # Export model if training was successful
        if model:
            export_path = self.export_model(model)
            return model, export_path
        else:
            self.logger.error(
                "Pipeline failed: model training was unsuccessful")
            print(
                "\033[1;31mPipeline failed: model training was unsuccessful\033[0m")
            return None, None


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train and export Markov Chain model")
    parser.add_argument("--n-gram", type=int, default=2,
                        help="N-gram size (default: 2)")
    parser.add_argument("--env", choices=["development", "test", "production"],
                        default="development", help="Environment (default: development)")
    parser.add_argument(
        "--memory", type=int, help="Memory threshold in MB (default: uses percentage)")
    parser.add_argument("--memory-pct", type=int, default=80,
                        help="Memory percentage to use if threshold not specified (default: 80)")
    parser.add_argument("--datasets", nargs="+", help="Paths to dataset files")

    args = parser.parse_args()

    # Create trainer
    trainer = MarkovChainTrainer(
        n_gram=args.n_gram,
        environment=args.env,
        memory_threshold_mb=args.memory,
        memory_percentage=args.memory_pct,
        dataset_paths=args.datasets
    )

    # Run pipeline
    trainer.run_pipeline()
