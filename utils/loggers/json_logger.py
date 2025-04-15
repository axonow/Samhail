import logging
import json
import os
from datetime import datetime
import inspect
import sys


# Configure JSON logging
class JsonLogger(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the log record."""

    def format(self, record):
        logobj = {}
        logobj["timestamp"] = datetime.utcnow().isoformat()
        logobj["name"] = record.name
        logobj["level"] = record.levelname
        logobj["module"] = record.module
        logobj["function"] = record.funcName
        logobj["message"] = record.getMessage()

        # Include exception info if available
        if record.exc_info:
            logobj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # Include any extra attributes
        if hasattr(record, "metrics"):
            logobj["metrics"] = record.metrics

        if hasattr(record, "model_id"):
            logobj["model_id"] = record.model_id

        if hasattr(record, "operation"):
            logobj["operation"] = record.operation

        return json.dumps(logobj)


def get_project_root():
    """
    Get the project root directory (Samhail directory)
    
    Returns:
        str: Path to project root
    """
    # Start with the current file's directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up until we find the project root (where Samhail is the directory name)
    while os.path.basename(current_path) != "Samhail" and current_path != os.path.dirname(current_path):
        current_path = os.path.dirname(current_path)
    
    # Return the project root path
    return current_path


def setup_log_file(log_file_path):
    """
    Creates or clears the log file.
    Ensures the directory exists and recreates the file.

    Args:
        log_file_path (str): Path to the log file

    Returns:
        str: Path to the created log file
    """
    # Get directory from file path
    log_dir = os.path.dirname(log_file_path)

    # Create directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create or overwrite the log file
    with open(log_file_path, "w") as f:
        f.write(f"# Log started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return log_file_path


def determine_log_path(log_file=None):
    """
    Determine the appropriate log file path based on caller module
    
    Args:
        log_file (str, optional): Explicit log file path
        
    Returns:
        str: The determined log file path
    """
    # If an explicit log file path is provided, use it
    if log_file:
        return os.path.abspath(log_file)
        
    # Get the caller's module file path
    frame = inspect.stack()[2]
    caller_file = frame.filename
    
    # Get relative path from project root
    project_root = get_project_root()
    if project_root in caller_file:
        relative_path = os.path.relpath(caller_file, project_root)
        # Remove file extension and get directory
        file_dir = os.path.dirname(relative_path)
        file_name = os.path.splitext(os.path.basename(relative_path))[0]
        
        # Create log path: logs/[module_dir]/[filename].log
        log_path = os.path.join(project_root, "logs", file_dir, f"{file_name}.log")
    else:
        # Default to logs/default.log if caller is outside project
        log_path = os.path.join(project_root, "logs", "default.log")
    
    return log_path


def get_logger(logger_name, log_file=None, clear_existing=True):
    """
    Create and configure a logger with JSON formatting.
    
    If log_file is not provided, generates a log path based on the calling module's location.
    For example, if called from models/production_models/markov_chain/train_and_export.py,
    the log file will be created at logs/models/production_models/markov_chain/train_and_export.log

    Args:
        logger_name (str): Name of the logger
        log_file (str, optional): Path to log file if file logging is desired
        clear_existing (bool): Whether to clear existing log file (default: True)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine the log file path
    log_path = determine_log_path(log_file)
    
    # Create logger with JSON formatter
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonLogger())
    logger.addHandler(console_handler)

    # Add file handler for persistent logs
    try:
        # Setup log file (create directory and clear file if requested)
        if clear_existing:
            setup_log_file(log_path)
        elif not os.path.exists(os.path.dirname(log_path)) and os.path.dirname(log_path):
            os.makedirs(os.path.dirname(log_path))

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(JsonLogger())
        logger.addHandler(file_handler)
        logger.info(f"JSON logging initialized to {log_path}")
    except (IOError, PermissionError) as e:
        # Log to console if file logging fails
        logger.error(f"Could not create log file {log_path}: {e}")

    return logger


def log_json(logger, message, data=None):
    """
    Directly log a JSON-formatted message with optional data.
    This is a helper function to consistently log structured data.

    Args:
        logger: The logger instance to use
        message (str): Log message
        data (dict, optional): Data to include in the log
    """
    extra = {}

    if data:
        extra["metrics"] = data

    logger.info(message, extra=extra)