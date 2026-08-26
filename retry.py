"""
Retry utility with exponential backoff for transient API failures.

Usage:
    @retry(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
    def call_api():
        ...
"""

import logging
import random
import time
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_DELAY = 30.0

TRANSIENT_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay: float = DEFAULT_MAX_DELAY,
    exceptions: Tuple[Type[Exception], ...] = TRANSIENT_EXCEPTIONS,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """
    Decorator that retries a function on transient exceptions
    using exponential backoff with jitter.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt == max_attempts:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts,
                            func.__qualname__,
                            exc,
                        )
                        raise

                    delay = min(
                        base_delay * (backoff_factor ** (attempt - 1)),
                        max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.5)
                    sleep_time = delay + jitter

                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). "
                        "Retrying in %.1fs...",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                        sleep_time,
                    )

                    if on_retry:
                        on_retry(attempt, exc)

                    time.sleep(sleep_time)

            raise last_exception

        return wrapper

    return decorator
