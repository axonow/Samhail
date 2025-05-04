from datetime import datetime
import os
import logging
import json
import sys
import inspect

# Configure JSON logging


class JsonLogger(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the log record."""

    def format(self, record):
        """
        Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            str: JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'path': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }

        # Include extra data if available
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics

        # Include exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': str(record.exc_info[0].__name__),
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        return json.dumps(log_data)


def get_project_root():
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Path to project root directory
    """
    # Start with the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the project root (2 levels up from loggers)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    return project_root


def setup_log_file(log_file_path):
    """
    Set up a log file with proper directory structure.

    Args:
        log_file_path (str): Path to the log file

    Returns:
        str: Absolute path to the log file
    """
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory {log_dir}: {e}")
            # Fall back to a temporary directory
            log_file_path = os.path.join(
                '/tmp', os.path.basename(log_file_path))

    return log_file_path


def determine_log_path(log_file=None):
    """
    Determine the path for the log file.

    Args:
        log_file (str, optional): Specific log file path

    Returns:
        str: Path to use for logging
    """
    if log_file:
        # Use the specified log file
        return setup_log_file(log_file)

    # Default log file in project logs directory
    project_root = get_project_root()
    log_dir = os.path.join(project_root, 'logs')

    # Create a timestamp-based log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_file = os.path.join(log_dir, f"samhail_{timestamp}.log")

    return setup_log_file(default_log_file)


def get_logger(logger_name, log_file=None, clear_existing=True, console_json=True):
    """
    Get a configured logger instance with JSON formatting.

    Args:
        logger_name (str): Name for the logger
        log_file (str, optional): Path to the log file
        clear_existing (bool): Whether to clear existing handlers
        console_json (bool): Whether to use JSON formatting for console output

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get logger by name
    logger = logging.getLogger(logger_name)

    # Set level to DEBUG by default
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers if requested
    if clear_existing and logger.handlers:
        logger.handlers.clear()

    # If logger already has handlers, return it
    if logger.handlers:
        return logger

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Use JSON formatting for console if requested
    if console_json:
        console_handler.setFormatter(JsonLogger())
    else:
        # Use regular text formatting for console
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # Create file handler with JSON formatting if log file specified
    if log_file is not None:
        log_path = determine_log_path(log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonLogger())
        logger.addHandler(file_handler)

    return logger


def log_json(logger, message, data=None):
    """
    Log a message with optional JSON data.

    Args:
        logger (logging.Logger): Logger instance
        message (str): Log message
        data (dict, optional): Data to include in the log
    """
    if data is None:
        logger.info(message)
    else:
        logger.info(message, extra={"metrics": data})
