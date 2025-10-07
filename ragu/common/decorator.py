import logging


def no_throw(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error occurred while executing {func.__name__}: {e}")
            return None
    return wrapper
