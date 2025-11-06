import sys
import os
import logging
from pythonjsonlogger import jsonlogger


def setup_logger(name: str):

    os.makedirs('./logs/', exist_ok=True)

    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'./logs/{name.lower()}.log')
    console_handler = logging.StreamHandler(stream=sys.stdout)
 
    formatter = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s")
    jsonFmt = jsonlogger.JsonFormatter(
        "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(process)d %(message)s",
        rename_fields={"levelname": "severity", "asctime": "timestamp"},
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(jsonFmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
