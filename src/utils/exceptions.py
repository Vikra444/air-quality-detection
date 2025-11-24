"""
Custom exception classes for AirGuard system.
"""


class AirGuardException(Exception):
    """Base exception for AirGuard system."""
    pass


class APIError(AirGuardException):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        self.api_name = api_name
        self.status_code = status_code
        super().__init__(message)


class DataValidationError(AirGuardException):
    """Exception raised for data validation errors."""
    pass


class ModelError(AirGuardException):
    """Exception raised for ML model errors."""
    pass


class DatabaseError(AirGuardException):
    """Exception raised for database errors."""
    pass


class ConfigurationError(AirGuardException):
    """Exception raised for configuration errors."""
    pass

