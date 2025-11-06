"""
Circuit breaker pattern for LLM API calls.
Implements automatic fallback and exponential backoff for retries.
"""

import time
import logging
from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 2           # Successes to close from half-open
    timeout: int = 60                    # Seconds before trying half-open
    timeout_multiplier: float = 2.0      # Exponential backoff multiplier


class CircuitBreaker:
    """
    Circuit breaker for monitoring and protecting LLM API calls.
    
    Automatically opens circuit after threshold failures,
    attempts recovery after timeout, and falls back to alternative providers.
    """
    
    def __init__(
        self,
        provider: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        timeout_multiplier: float = 2.0
    ):
        """
        Initialize circuit breaker.
        
        Args:
            provider: Provider name (e.g., "openai", "ollama")
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes to close from half-open
            timeout: Initial timeout in seconds before trying half-open
            timeout_multiplier: Multiplier for exponential backoff
        """
        self.provider = provider
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            timeout_multiplier=timeout_multiplier
        )
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        self.consecutive_timeouts = 0
    
    def _should_attempt(self) -> bool:
        """Check if we should attempt a request based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.next_attempt_time and datetime.now() >= self.next_attempt_time:
                # Transition to half-open
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker [{self.provider}] transitioning to HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN: allow attempts
        return True
    
    def _record_success(self):
        """Record a successful request."""
        self.failure_count = 0
        self.consecutive_timeouts = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                # Close the circuit
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker [{self.provider}] CLOSED after recovery")
        elif self.state == CircuitState.OPEN:
            # Shouldn't happen, but handle it
            self.state = CircuitState.CLOSED
            logger.warning(f"Circuit breaker [{self.provider}] unexpectedly CLOSED")
    
    def _record_failure(self):
        """Record a failed request."""
        self.last_failure_time = datetime.now()
        self.failure_count += 1
        self.success_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, reopen circuit
            self.state = CircuitState.OPEN
            self._calculate_next_attempt()
            logger.warning(f"Circuit breaker [{self.provider}] reopened after failure in HALF_OPEN")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                # Open the circuit
                self.state = CircuitState.OPEN
                self._calculate_next_attempt()
                logger.warning(
                    f"Circuit breaker [{self.provider}] OPENED after {self.failure_count} failures"
                )
    
    def _calculate_next_attempt(self):
        """Calculate next attempt time with exponential backoff."""
        timeout = self.config.timeout * (self.config.timeout_multiplier ** self.consecutive_timeouts)
        self.next_attempt_time = datetime.now() + timedelta(seconds=timeout)
        self.consecutive_timeouts += 1
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self._should_attempt():
            raise CircuitBreakerOpenError(
                f"Circuit breaker [{self.provider}] is OPEN. "
                f"Next attempt at {self.next_attempt_time}"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            logger.error(f"Circuit breaker [{self.provider}] recorded failure: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        if not self._should_attempt():
            raise CircuitBreakerOpenError(
                f"Circuit breaker [{self.provider}] is OPEN. "
                f"Next attempt at {self.next_attempt_time}"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
        return False  # Don't suppress exceptions
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "provider": self.provider,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None,
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breakers for different providers
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider: str, **kwargs) -> CircuitBreaker:
    """
    Get or create circuit breaker for a provider.
    
    Args:
        provider: Provider name
        **kwargs: Additional arguments for CircuitBreaker
        
    Returns:
        CircuitBreaker instance
    """
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker(provider, **kwargs)
    return _circuit_breakers[provider]

