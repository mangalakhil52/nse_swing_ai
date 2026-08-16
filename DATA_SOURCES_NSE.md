# Data Sources, Ingestion Architecture & Provider Abstraction
**Project**: NSE Swing Intelligence Engine (`nse_swing_ai`)  
**Data Integrity Principle**: Zero-Trust Ingestion, Timestamp Verification & Multi-Tier Fallback  
**Strict Policy**: NO Yahoo Finance data. Exclusively utilize NSE Official Website/Bhavcopy/APIs, BSE Official Feeds, and Chartink Custom Scanner APIs.

---

## 1. Data Source Hierarchy & Classification

To eliminate hallucinations and prevent stale analysis, all data ingested into the system is cataloged into strict reliability tiers.

```text
+-------------------------------------------------------------------------+
| TIER 1: AUTHORITATIVE & REGULATORY SOURCES (Primary Truth)              |
| - National Stock Exchange of India (NSE India Bhavcopy & Official API)  |
|   - CM Bhavcopy (EOD OHLCV, Trades, Traded Value)                       |
|   - Sec_Bhav (Delivery Quantity & Delivery Percentage)                  |
|   - NSE Live Market API (/api/quote-equity, /api/equity-stockIndices)   |
|   - NSE Corporate Announcements & Results Calendar                      |
| - Bombay Stock Exchange (BSE India Corporate Filings & Announcements)   |
| - Securities and Exchange Board of India (SEBI Circulars, ASM/GSM Lists)|
| - Reserve Bank of India (RBI Policy & Macro Data)                       |
| - Company Annual Reports, Investor Presentations & Quarterly Press Notes|
+------------------------------------+------------------------------------+
                                     |
+------------------------------------+------------------------------------+
| TIER 2: CHARTINK SCANNER & FINANCIAL INTELLIGENCE APIS                  |
| - Chartink Scanner API (chartink.com/screener/process custom queries)   |
| - Screener.in / Trendlyne structured fundamental databases              |
| - Reuters, Bloomberg India, Economic Times, Business Standard, Mint     |
+------------------------------------+------------------------------------+
                                     |
+------------------------------------+------------------------------------+
| TIER 3: SECONDARY VERIFIED FINANCIAL FEEDS & RSS                        |
| - Moneycontrol Verified Desks, BSE Announcement RSS Feeds               |
| - Google News Financial RSS Feeds (Strict Multi-Source Verification)    |
| * NOTE: Yahoo Finance is strictly banned due to data quality issues.    |
+-------------------------------------------------------------------------+
```

---

## 2. Provider Abstraction Interfaces

All data access is decoupled via Python Abstract Base Classes (ABC). Application components depend exclusively on abstract contracts.

### A. Market Data Provider (MarketDataProvider)
```python
class MarketDataProvider(ABC):
    @abstractmethod
    async def get_historical_ohlcv(
        self, symbol: str, timeframe: TimeFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Returns standardized OHLCV DataFrame with columns: 
        ['open', 'high', 'low', 'close', 'volume', 'delivery_volume', 'delivery_pct', 'turnover']"""
        pass

    @abstractmethod
    async def get_latest_quote(self, symbol: str) -> LiveQuote:
        """Returns real-time/delayed quote with bid, ask, last_price, vwap, circuit limits."""
        pass

    @abstractmethod
    async def get_market_breadth(self, index_symbol: str = "NIFTY 500") -> MarketBreadthData:
        """Returns advances, declines, unchanged, % above key SMAs."""
        pass
```

### B. Chartink Scanner Provider (ChartinkScannerProvider)
```python
class ChartinkScannerProvider(ABC):
    @abstractmethod
    async def run_scanner_query(self, scan_clause: str) -> list[str]:
        """Executes custom technical scan against Chartink query engine and returns matching symbols."""
        pass
```

### C. Fundamental Provider (FundamentalProvider)
```python
class FundamentalProvider(ABC):
    @abstractmethod
    async def get_quarterly_financials(self, symbol: str) -> QuarterlyFinancials:
        """Returns Sales, PAT, EBITDA, Margins, EPS for last 8 quarters."""
        pass

    @abstractmethod
    async def get_annual_ratios(self, symbol: str) -> AnnualRatios:
        """Returns ROE, ROCE, Debt/Equity, CFO, Working Capital Days for 5 years."""
        pass

    @abstractmethod
    async def get_shareholding_pattern(self, symbol: str) -> ShareholdingPattern:
        """Returns Promoter %, Pledging %, FII %, DII %, Public % across quarters."""
        pass
```

### D. News & Catalyst Provider (NewsProvider)
```python
class NewsProvider(ABC):
    @abstractmethod
    async def fetch_company_announcements(
        self, symbol: str, lookback_days: int = 14
    ) -> list[CorporateAnnouncement]:
        """Returns official exchange announcements (BSE/NSE filings)."""
        pass

    @abstractmethod
    async def fetch_news_feed(
        self, symbol: str, lookback_days: int = 7
    ) -> list[NewsArticle]:
        """Returns Tier 1/2 news articles with URL, source, author, timestamp, text."""
        pass
```

### E. Corporate Actions Provider (CorporateActionsProvider)
```python
class CorporateActionsProvider(ABC):
    @abstractmethod
    async def get_upcoming_events(self, symbol: str) -> list[CorporateEvent]:
        """Returns upcoming Board Meetings (Earnings), Ex-Dates (Dividend, Split, Bonus)."""
        pass

    @abstractmethod
    async def get_historical_adjustments(self, symbol: str) -> list[SplitBonusAdjustment]:
        """Returns split ratios and bonus factors for historical back-adjustment."""
        pass
```

---

## 3. Data Ingestion Architecture & Fallback Flow

```text
                       Fetch Request (e.g. OHLCV for TRENT)
                                        |
                                        v
                       +---------------------------------+
                       |    DATA INGESTION CONTROLLER    |
                       +----------------+----------------+
                                        |
                         Try Primary Provider (NSE Official Bhavcopy/API)
                                        |
                        +---------------+---------------+
                        | Success?                      |
                        +-----------------+-------------+
                        v (YES)           v (NO / Timeout)
           +----------------------+  +--------------------------+
           | Raw Primary Payload  |  | Try Secondary Provider   |
           +----------+-----------+  | (Chartink API / BSE EOD) |
                      |              +----------+---------------+
                      |                         |
                      |              +----------+-----------+
                      |              | Success?             |
                      |              +-----------+----------+
                      |              v (YES)     v (NO)
                      |  +----------------------+ +---------------+
                      |  | Raw Secondary Payload| | RAISE CRITICAL|
                      |  +-----------+----------+ | DATA MISSING  |
                      |              |            |  (ABORT/SKIP) |
                      |              |            +---------------+
                      v              v
           +------------------------------------+
           |      DATA VALIDATION PIPELINE      |
           |  - Missing bar detection           |
           |  - High >= Open/Close/Low check    |
           |  - Volume > 0 check                |
           |  - Abnormal outlier/spike filter   |
           |  - Timestamp freshness check       |
           +-----------------+------------------+
                             |
                             v
           +------------------------------------+
           |       PERSIST TO DATABASE          |
           |    (PostgreSQL / TimescaleDB)      |
           +------------------------------------+
```

---

## 4. Specific Data Feeds & Endpoints

| Data Category | Primary Source | Secondary Fallback | Refresh Cadence | Validation Rule |
|---|---|---|---|---|
| **Daily EOD OHLCV + Delivery** | NSE Bhavcopy & Sec_Bhav | BSE Official Bhavcopy / Chartink EOD | Daily at 15:45 IST | Zero volume reject, split continuity |
| **Market Indices (Nifty 50, Sectorals)** | NSE Index Bhavcopy (niftyindices.com) | NSE Live Index Feed | Daily at 15:45 IST | Close within High-Low range |
| **Realtime Screener & Scans** | Chartink Custom Scanner API (/screener/process) | NSE Sector & Index Scraper | Daily at 16:00 IST | Clause verification & cross-check |
| **Corporate Announcements** | BSE / NSE Corporate Feed API | Moneycontrol Verified Desks | 4x daily | Extracted timestamp <= 24h old |
| **Earnings Calendar** | NSE Board Meetings Feed | BSE Corporate Diary / Screener | Daily at 08:30 IST | Exact date verification |
| **Fundamentals & Ratios** | Screener.in API / Exchange Filings | Trendlyne / BSE Annual Filings | Weekly | CFO vs PAT sign & magnitude check |
| **Surveillance Lists (ASM/GSM)** | NSE Official Circulars | BSE Notice Feed | Daily at 08:00 IST | Strict binary flag (In/Out) |

---

## 5. Data Freshness & Anti-Staleness Rules

Every piece of data stored and passed to agents must carry:
1. `source_identifier`: e.g. `NSE_BHAVCOPY_20260816`
2. `retrieval_timestamp`: Exact ISO 8601 UTC/IST timestamp of HTTP response.
3. `data_effective_timestamp`: The market date/time the data represents.
4. `freshness_category`:
   - `LIVE`: Ingested within 15 minutes of market close.
   - `RECENT`: Ingested within 24 hours.
   - `DELAYED`: 1 to 5 trading sessions old.
   - `STALE`: > 5 trading sessions old (triggers warning/rejection for price; acceptable for quarterly financials up to 90 days).
   - `UNKNOWN`: Immediate disqualification.
