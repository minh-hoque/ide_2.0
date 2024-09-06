import logging
import colorlog
import streamlit as st


def setup_logger(name=__name__, level=logging.INFO):
    """
    Set up a colored logger with the specified name and level.

    Args:
        name (str): The name of the logger. Defaults to the module name.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = colorlog.getLogger(name)

    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = False

        # Create console handler
        console_handler = colorlog.StreamHandler()

        # Create formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Add formatter to console handler
        console_handler.setFormatter(formatter)

        # Add console handler to logger
        logger.addHandler(console_handler)

    # Set logging level
    logger.setLevel(level)

    return logger


def get_logger(name=__name__, level=logging.INFO):
    """
    Get or create a logger with the specified name and level.

    Args:
        name (str): The name of the logger. Defaults to the module name.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return setup_logger(name, level)


def setup_logging(logger_name: str):
    """Set up logging level selector in the sidebar for all pages."""
    logger = get_logger(logger_name)
    with st.sidebar:
        logging_level = st.selectbox(
            "Select Logging Level",
            ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            index=1,
        )
    logger.setLevel(logging_level)
    return logger, logging_level
