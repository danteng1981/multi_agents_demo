"""Circuit Breaker Pattern Implementation

This module implements the Circuit Breaker pattern to prevent cascading failures
in distributed systems by failing fast when a service is unavailable.
"""

import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Enumeration of circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Failure threshold exceeded, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit Breaker implementation for protecting function calls.
    
    The circuit breaker monitors function calls and tracks failures. When the
    failure threshold is exceeded, it opens the circuit and fails fast for
    subsequent calls. After a recovery timeout, it enters a half-open state
    to test if the service has recovered.
    
    Attributes:
        failure_threshold (int): Number of failures before opening the circuit
        recovery_timeout (float): Seconds to wait before attempting recovery
        expected_exception (Exception): Exception type that counts as failure
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaking
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count = 0
        
        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the protected function
        """
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker entering HALF_OPEN state")
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Last failure: "
                    f"{time.time() - self._last_failure_time:.1f}s ago"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery.
        
        Returns:
            True if recovery should be attempted, False otherwise
        """
        if self._last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution.
        
        Resets the circuit breaker to CLOSED state if in HALF_OPEN state,
        or resets failure count if in CLOSED state.
        """
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker recovery successful, closing circuit")
            self._reset()
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count += 1
    
    def _on_failure(self) -> None:
        """Handle failed function execution.
        
        Increments failure count and opens circuit if threshold is exceeded.
        If in HALF_OPEN state, immediately reopens the circuit.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker recovery failed, reopening circuit")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker opening after {self._failure_count} failures"
            )
            self._state = CircuitState.OPEN
        else:
            logger.warning(
                f"Circuit breaker failure {self._failure_count}/{self.failure_threshold}"
            )
    
    def _reset(self) -> None:
        """Reset the circuit breaker to initial CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker reset to CLOSED state")
    
    def get_state(self) -> dict:
        """Get the current state of the circuit breaker.
        
        Returns:
            Dictionary containing current circuit breaker status:
            - state: Current circuit state (CLOSED, OPEN, or HALF_OPEN)
            - failure_count: Number of consecutive failures
            - failure_threshold: Threshold for opening circuit
            - last_failure_time: Timestamp of last failure (if any)
            - recovery_timeout: Configured recovery timeout
            - time_until_retry: Seconds until retry attempt (if circuit is open)
        """
        status = {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.recovery_timeout,
            "success_count": self._success_count
        }
        
        if self._state == CircuitState.OPEN and self._last_failure_time:
            time_since_failure = time.time() - self._last_failure_time
            time_until_retry = max(0, self.recovery_timeout - time_since_failure)
            status["time_until_retry"] = round(time_until_retry, 2)
        
        return status
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker.
        
        Usage:
            @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
            def my_function():
                # function implementation
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def __repr__(self) -> str:
        """String representation of circuit breaker."""
        return (
            f"CircuitBreaker(state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )
