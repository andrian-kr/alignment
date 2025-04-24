import logging

LOGGER_CONFIG = {
    "general": {
        "level": logging.DEBUG,
        "formatter": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "llm": {
        "level": logging.INFO,
        "formatter": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}


def setup_logger(log_name: str):
    log_config = LOGGER_CONFIG[log_name]
    logger = logging.getLogger(log_name)
    logger.propagate = False  # Prevent propagation to root logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_config["formatter"]))
    logger.addHandler(console_handler)
    logger.setLevel(log_config["level"])
    return logger


general_logger: logging.Logger = setup_logger("general")
llm_logger: logging.Logger = setup_logger("llm")
