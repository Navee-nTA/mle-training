import logging


def get_logger(name, log_file=None, log_level="INFO", no_console_log=False):
    """
    Create and configure a logger.

    This function creates a logger with the specified name and log level. It
    can log to a file, the console, or both. The format for logging is:
    timestamp - logger name - log level - message.

    Parameters
    ----------
    name : str
        The name of the logger.
    log_file : str, optional
        The file path to log to. If None, file logging is disabled. Default is
        None.
    log_level : str, optional
        The log level for the logger. Default is "INFO". Acceptable values are
        "DEBUG", "INFO","WARNING", "ERROR", "CRITICAL".
    no_console_log : bool, optional
        If True, disables logging to the console. Default is False.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
