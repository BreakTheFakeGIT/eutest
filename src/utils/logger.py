import sys
import os
import logging
from pythonjsonlogger import jsonlogger


def diff_lists_set(list1: list, list2: list):
    """
    Returns the difference between two lists using sets:
    """
    set1, set2 = set(list1), set(list2)
    diff1 = set1 - set2   # Elements in list1 but not in list2
    diff2 = set2 - set1   # Elements in list2 but not in list1
    return list(diff1), list(diff2)

def set_datetime_local():
    """ Get current datetime in 'Europe/Warsaw' timezone formatted as YYYYMMDD_HHMMSS """
    import time
    import datetime
    import pytz
    tz_info = pytz.timezone('Europe/Warsaw')
    return datetime.datetime.now(tz=tz_info).strftime("%Y%m%d_%H%M%S")

def root_dir() -> str:
    """ Get the root directory of the project """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def base_path_with_timestamp(path: str, prefix: str) -> str:
    """ Create a base path with current timestamp """
    timestamp = set_datetime_local()
    base_path = os.path.join(path, f"{prefix}_{timestamp}")
    return base_path

def create_directories(dir_path: str):
    """ Create necessary directories for the project """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_env():
    """ Load environment variables from .env file """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root_dir() , '.env'))

def setup_logger(name: str):
    """ Set up a logger with both file and console handlers """
    os.makedirs('./logs/', exist_ok=True)

    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'./logs/{name.lower()}.log',mode='w', encoding='utf-8')
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
