import logging
import json
import os
from datetime import datetime


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
        print(f"\033[1mCreated log directory at {log_dir}\033[0m")

    # Create or overwrite the log file
    with open(log_file_path, "w") as f:
        f.write(
            f"# Log started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\033[1mInitialized log file at {log_file_path}\033[0m")
    return log_file_path


def get_logger(logger_name, log_file=None, clear_existing=True):
    """
    Create and configure a logger with JSON formatting.

    Args:
        logger_name (str): Name of the logger
        log_file (str, optional): Path to log file if file logging is desired
        clear_existing (bool): Whether to clear existing log file (default: True)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger with JSON formatter
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Only add handlers if the logger doesn't have any
    if not logger.handlers:
        # Create console handler with JSON formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonLogger())
        logger.addHandler(console_handler)

        # Optional: Add file handler for persistent logs
        if log_file:
            try:
                # Setup log file (create directory and clear file if requested)
                if clear_existing:
                    setup_log_file(log_file)
                elif not os.path.exists(os.path.dirname(log_file)) and os.path.dirname(
                    log_file
                ):
                    os.makedirs(os.path.dirname(log_file))

                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(JsonLogger())
                logger.addHandler(file_handler)
                logger.info(f"JSON logging initialized to {log_file}")
            except (IOError, PermissionError) as e:
                # Log to console if file logging fails
                print(
                    f"\033[1mWarning: Could not create log file {log_file}: {e}\033[0m"
                )

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
