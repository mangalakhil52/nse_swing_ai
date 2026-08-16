"""
Core Domain Enumerations for nse_swing_ai.
Defines all strongly-typed enum categories across market regimes, conviction grades, agent states, and patterns.
"""

from enum import Enum


class MarketRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class TradingStance(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    SELECTIVE = "SELECTIVE"
    DEFENSIVE = "DEFENSIVE"
    NO_TRADE = "NO_TRADE"


class ConvictionGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    REJECT = "REJECT"


class ConfluenceState(str, Enum):
    VERY_HIGH = "VERY_HIGH_CONFLUENCE"
    HIGH = "HIGH_CONFLUENCE"
    MODERATE = "MODERATE_CONFLUENCE"
    LOW = "LOW_CONFLUENCE"
    CONFLICTED = "CONFLICTED"


class PatternType(str, Enum):
    VOLATILITY_CONTRACTION_PATTERN = "VOLATILITY_CONTRACTION_PATTERN"
    FLAT_BASE_BREAKOUT = "FLAT_BASE_BREAKOUT"
    CUP_AND_HANDLE = "CUP_AND_HANDLE"
    HIGH_TIGHT_FLAG = "HIGH_TIGHT_FLAG"
    EMA_PULLBACK_REVERSAL = "EMA_PULLBACK_REVERSAL"
    INSIDE_BAR_BREAKOUT = "INSIDE_BAR_BREAKOUT"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    UNSTRUCTURED_TREND = "UNSTRUCTURED_TREND"
    NO_PATTERN = "NO_PATTERN"


class AgentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    SKIPPED = "SKIPPED"


class SignalType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    REJECT = "REJECT"


class SourceTier(int, Enum):
    TIER_1 = 1  # Official Exchange (NSE/BSE), SEBI, RBI, Official Filings
    TIER_2 = 2  # Chartink, Screener, Reuters, ET, Mint, Business Standard
    TIER_3 = 3  # General Verified Financial Feeds / Multi-source RSS


class DataFreshness(str, Enum):
    LIVE = "LIVE"          # Within 15 minutes of market close
    RECENT = "RECENT"      # Within 24 hours
    DELAYED = "DELAYED"    # 1-5 trading sessions old
    STALE = "STALE"        # > 5 trading sessions old
    UNKNOWN = "UNKNOWN"    # Disqualification


class SentimentType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    ALREADY_PRICED = "ALREADY_PRICED"
    UNKNOWN = "UNKNOWN"


class CatalystType(str, Enum):
    EARNINGS_ANNOUNCEMENT = "EARNINGS_ANNOUNCEMENT"
    ORDER_WIN = "ORDER_WIN"
    CAPACITY_EXPANSION = "CAPACITY_EXPANSION"
    REGULATORY_CLEARANCE = "REGULATORY_CLEARANCE"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    FUNDRAISING_DEBT_REDUCTION = "FUNDRAISING_DEBT_REDUCTION"
    NO_CATALYST = "NO_CATALYST"
    NEGATIVE_CATALYST = "NEGATIVE_CATALYST"


class FundamentalGrade(str, Enum):
    EXCEPTIONAL = "EXCEPTIONAL"
    STRONG = "STRONG"
    GOOD = "GOOD"
    AVERAGE = "AVERAGE"
    WEAK = "WEAK"
    DANGEROUS = "DANGEROUS"


class ForensicVerdict(str, Enum):
    CLEAN = "NO_MATERIAL_RED_FLAG_FOUND"
    MINOR_CONCERN = "MINOR_CONCERN"
    RED_FLAG_REJECT = "RED_FLAG_REJECT"


class TradeStatus(str, Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    TARGET_1_HIT = "TARGET_1_HIT"
    TARGET_2_HIT = "TARGET_2_HIT"
    TARGET_3_HIT = "TARGET_3_HIT"
    STOPPED_OUT = "STOPPED_OUT"
    TIME_EXPIRED = "TIME_EXPIRED"
    CANCELLED = "CANCELLED"


class ExitReason(str, Enum):
    TARGET_1 = "TARGET_1_HIT"
    TARGET_2 = "TARGET_2_HIT"
    TARGET_3 = "TARGET_3_HIT"
    STOP_LOSS = "STOP_LOSS_HIT"
    TIME_STOP = "TIME_STOP_15_SESSIONS"
    TRAILING_STOP = "TRAILING_STOP_9_EMA"
    MANUAL_EXIT = "MANUAL_EXIT"
