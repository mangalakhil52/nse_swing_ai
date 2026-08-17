"""
Custom Domain Exceptions for nse_swing_ai.
"""


class NseSwingError(Exception):
    """Base exception for all domain errors."""
    pass


class DataIntegrityError(NseSwingError):
    """Raised when market data fails validation checks (e.g. missing bars, corrupt OHLC, split anomalies)."""
    pass


class StaleDataError(DataIntegrityError):
    """Raised when data timestamp exceeds acceptable staleness thresholds."""
    pass


class RiskVetoError(NseSwingError):
    """Raised when a candidate triggers a hard risk disqualifier."""
    pass


class ProviderUnavailableError(NseSwingError):
    """Raised when primary and secondary data providers fail or time out."""
    pass


class DisagreementError(NseSwingError):
    """Raised when independent specialist agents have unresolved contradictory conclusions."""
    pass


class ConfigurationError(NseSwingError):
    """Raised when system settings or scoring configurations are invalid."""
    pass


class DataUnavailableException(NseSwingError):
    """Raised when required investment data is missing or unavailable."""
    pass

