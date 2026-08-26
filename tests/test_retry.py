from unittest.mock import MagicMock

import pytest

from retry import retry


class TestRetryDecorator:
    def test_succeeds_on_first_attempt(self):
        mock_fn = MagicMock(return_value="ok")

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            return mock_fn()

        result = fn()
        assert result == "ok"
        assert mock_fn.call_count == 1

    def test_retries_on_transient_error(self):
        mock_fn = MagicMock(side_effect=[ConnectionError("fail"), "ok"])

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            return mock_fn()

        result = fn()
        assert result == "ok"
        assert mock_fn.call_count == 2

    def test_raises_after_max_attempts(self):
        mock_fn = MagicMock(side_effect=ConnectionError("fail"))

        @retry(max_attempts=2, base_delay=0.01)
        def fn():
            return mock_fn()

        with pytest.raises(ConnectionError):
            fn()
        assert mock_fn.call_count == 2

    def test_does_not_retry_on_non_transient_error(self):
        mock_fn = MagicMock(side_effect=TypeError("permanent"))

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            return mock_fn()

        with pytest.raises(TypeError):
            fn()
        assert mock_fn.call_count == 1

    def test_on_retry_callback_called(self):
        mock_fn = MagicMock(side_effect=[TimeoutError("t"), "ok"])
        on_retry = MagicMock()

        @retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        def fn():
            return mock_fn()

        result = fn()
        assert result == "ok"
        assert on_retry.call_count == 1
        args = on_retry.call_args[0]
        assert args[0] == 1
        assert isinstance(args[1], TimeoutError)

    def test_preserves_function_name(self):
        @retry(max_attempts=1, base_delay=0.01)
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"

    def test_multiple_retries(self):
        mock_fn = MagicMock(
            side_effect=[ConnectionError("1"), ConnectionError("2"), "ok"]
        )

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            return mock_fn()

        result = fn()
        assert result == "ok"
        assert mock_fn.call_count == 3
