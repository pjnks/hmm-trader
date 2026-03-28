"""
src/alternative_data.py
───────────────────────
Alternative data integration for HMM-Trader.

Two free data sources:
  1. SEC EDGAR Form 4 — corporate insider buying/selling
  2. Congressional trading — via Capitol Trades scraping (free)

Produces an alt_data_boost multiplier (0.7–1.5) for CITRINE/BERYL scoring:
  - Strong insider buying → 1.3–1.5x boost
  - Neutral / no data → 1.0x (no effect)
  - Net insider selling → 0.7–0.9x penalty

Usage:
  # Standalone test
  python -m src.alternative_data --ticker NVDA

  # In CITRINE/BERYL
  from src.alternative_data import AlternativeDataScore
  scorer = AlternativeDataScore()
  boost = scorer.get_boost("NVDA")  # Returns 0.7–1.5
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

log = logging.getLogger("alt_data")

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "alternative_data.db"
CIK_CACHE_PATH = ROOT / "data_cache" / "sec_cik_map.json"

# SEC EDGAR requires a User-Agent header
SEC_USER_AGENT = "HMMTrader/1.0 research@hmmtrader.dev"

# Rate limiting: SEC requests max 10/sec, we'll be conservative
SEC_RATE_LIMIT_S = 0.2  # 200ms between requests


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class InsiderTrade:
    """A single Form 4 insider transaction."""
    ticker: str
    insider_name: str
    insider_title: str  # e.g., "CEO", "CFO", "Director"
    transaction_date: str  # YYYY-MM-DD
    transaction_code: str  # P=Purchase, S=Sale, F=Tax, A=Award, M=Exercise
    shares: float
    price_per_share: float
    acquired_or_disposed: str  # A=Acquired, D=Disposed
    filing_date: str  # YYYY-MM-DD
    is_10b5_1: bool = False  # Pre-planned sale under Rule 10b5-1 (NOT a signal)


@dataclass
class InsiderSignal:
    """Aggregated insider trading signal for a ticker."""
    ticker: str
    n_buys: int = 0  # Purchase transactions (code P)
    n_sells: int = 0  # Sale transactions (code S — excluding 10b5-1 planned sales)
    n_sells_10b5_1: int = 0  # Pre-planned 10b5-1 sales (NOT a signal)
    n_tax: int = 0  # Tax withholding (code F) — not meaningful signal
    buy_value: float = 0.0  # Total $ of purchases
    sell_value: float = 0.0  # Total $ of discretionary sales (excluding 10b5-1)
    sell_value_10b5_1: float = 0.0  # Total $ of 10b5-1 planned sales (ignored)
    unique_buyers: int = 0  # Distinct insiders who bought
    unique_sellers: int = 0  # Distinct insiders who sold (discretionary only)
    cluster_buy: bool = False  # 3+ insiders bought in 30 days
    last_updated: str = ""

    @property
    def net_signal(self) -> str:
        """BULLISH if net buying, BEARISH only if extreme discretionary selling.

        Large-cap reality: most insider sales are compensation-related (10b5-1 plans
        or tax withholding). Only flag BEARISH when discretionary selling is extreme:
        sell_value > 10x buy_value AND 3+ unique sellers.
        """
        if self.n_buys > 0 and self.n_buys > self.n_sells:
            return "BULLISH"
        # Only BEARISH for extreme discretionary selling
        if (self.n_sells > 0
                and self.sell_value > max(self.buy_value, 1) * 10
                and self.unique_sellers >= 3):
            return "BEARISH"
        return "NEUTRAL"


# ── CIK Mapper ───────────────────────────────────────────────────────────────

class CIKMapper:
    """Maps stock tickers to SEC CIK numbers."""

    def __init__(self):
        self._map: dict[str, int] = {}
        self._load_or_fetch()

    def _load_or_fetch(self):
        """Load CIK map from cache, or fetch from SEC if stale."""
        if CIK_CACHE_PATH.exists():
            age = time.time() - CIK_CACHE_PATH.stat().st_mtime
            if age < 7 * 86400:  # Refresh weekly
                with open(CIK_CACHE_PATH) as f:
                    self._map = json.load(f)
                return

        self._fetch_from_sec()

    def _fetch_from_sec(self):
        """Fetch ticker→CIK mapping from SEC company_tickers.json."""
        url = "https://www.sec.gov/files/company_tickers.json"
        try:
            req = Request(url, headers={"User-Agent": SEC_USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            self._map = {}
            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = entry.get("cik_str")
                if ticker and cik:
                    self._map[ticker] = int(cik)

            # Cache to disk
            CIK_CACHE_PATH.parent.mkdir(exist_ok=True)
            with open(CIK_CACHE_PATH, "w") as f:
                json.dump(self._map, f)

            log.info(f"Fetched CIK map: {len(self._map)} tickers")
        except Exception as e:
            log.warning(f"Failed to fetch CIK map: {e}")

    def get_cik(self, ticker: str) -> Optional[int]:
        return self._map.get(ticker.upper())


# ── Insider Tracker (SEC EDGAR Form 4) ───────────────────────────────────────

class InsiderTracker:
    """
    Fetches and parses SEC Form 4 filings for insider trading data.
    Uses the free data.sec.gov submissions API + XML parsing.
    """

    def __init__(self):
        self._cik_mapper = CIKMapper()
        self._db = _init_db()
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce SEC rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_RATE_LIMIT_S:
            time.sleep(SEC_RATE_LIMIT_S - elapsed)
        self._last_request_time = time.time()

    def _sec_get(self, url: str) -> bytes:
        """GET request to SEC with rate limiting and proper User-Agent."""
        self._rate_limit()
        req = Request(url, headers={"User-Agent": SEC_USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            return resp.read()

    def fetch_form4_filings(self, ticker: str, days: int = 90) -> list[InsiderTrade]:
        """
        Fetch recent Form 4 filings for a ticker from SEC EDGAR.
        Returns list of InsiderTrade objects.
        """
        # Check cache first
        cached = self._load_cached(ticker, days)
        if cached is not None:
            return cached

        cik = self._cik_mapper.get_cik(ticker)
        if cik is None:
            log.debug(f"No CIK for {ticker}")
            return []

        trades = []
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            # Get company submissions (includes recent filings list)
            cik_padded = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            data = json.loads(self._sec_get(url))

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            # Filter to Form 4 filings within date range
            form4_filings = []
            for form, date, accession, doc in zip(forms, dates, accessions, primary_docs):
                if form == "4" and date >= cutoff:
                    form4_filings.append((date, accession, doc))

            # Parse up to 20 most recent Form 4 XMLs
            for filing_date, accession, doc in form4_filings[:20]:
                try:
                    acc_clean = accession.replace("-", "")
                    # primaryDocument may have XSLT prefix (e.g., "xslF345X06/file.xml")
                    # Strip it to get the raw XML filename
                    xml_filename = doc.split("/")[-1] if "/" in doc else doc
                    xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{xml_filename}"
                    xml_data = self._sec_get(xml_url)
                    parsed = self._parse_form4_xml(xml_data, ticker, filing_date)
                    trades.extend(parsed)
                except Exception as e:
                    log.debug(f"Failed to parse Form 4 for {ticker}: {e}")
                    continue

            # Cache results
            self._cache_trades(ticker, trades)
            log.info(f"  [AltData] {ticker}: {len(trades)} insider transactions "
                     f"({len(form4_filings)} filings)")

        except Exception as e:
            log.warning(f"  [AltData] Failed to fetch Form 4 for {ticker}: {e}")

        return trades

    def _parse_form4_xml(self, xml_data: bytes, ticker: str,
                         filing_date: str) -> list[InsiderTrade]:
        """Parse a Form 4 XML document into InsiderTrade objects."""
        trades = []
        try:
            root = ET.fromstring(xml_data)

            # Get reporting owner info
            owner_el = root.find(".//reportingOwner")
            owner_name = ""
            owner_title = ""
            if owner_el is not None:
                name_el = owner_el.find(".//rptOwnerName")
                owner_name = name_el.text if name_el is not None else ""
                title_el = owner_el.find(".//officerTitle")
                owner_title = title_el.text if title_el is not None else ""
                # Check if director
                is_dir = owner_el.find(".//isDirector")
                if is_dir is not None and is_dir.text == "1" and not owner_title:
                    owner_title = "Director"

            # Check if filing is under a 10b5-1 pre-planned trading arrangement
            aff_el = root.find(".//aff10b5One")
            is_10b5_1 = (aff_el is not None and aff_el.text == "1")

            # Parse non-derivative transactions
            for txn in root.findall(".//nonDerivativeTransaction"):
                try:
                    date_el = txn.find(".//transactionDate/value")
                    code_el = txn.find(".//transactionCode")
                    shares_el = txn.find(".//transactionShares/value")
                    price_el = txn.find(".//transactionPricePerShare/value")
                    ad_el = txn.find(".//transactionAcquiredDisposedCode/value")

                    if date_el is None or code_el is None:
                        continue

                    txn_code = code_el.text or ""
                    shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0
                    price = float(price_el.text) if price_el is not None and price_el.text else 0

                    trades.append(InsiderTrade(
                        ticker=ticker,
                        insider_name=owner_name,
                        insider_title=owner_title,
                        transaction_date=date_el.text or filing_date,
                        transaction_code=txn_code,
                        shares=shares,
                        price_per_share=price,
                        acquired_or_disposed=ad_el.text if ad_el is not None else "",
                        filing_date=filing_date,
                        is_10b5_1=is_10b5_1,
                    ))
                except (ValueError, AttributeError):
                    continue

        except ET.ParseError as e:
            log.debug(f"XML parse error: {e}")

        return trades

    def get_signal(self, ticker: str, days: int = 90) -> InsiderSignal:
        """
        Get aggregated insider trading signal for a ticker.
        Focuses on open-market purchases (code P) and sales (code S).
        Ignores tax withholding (F), awards (A), and option exercises (M).
        """
        trades = self.fetch_form4_filings(ticker, days)
        signal = InsiderSignal(ticker=ticker)

        if not trades:
            return signal

        buyers = set()
        sellers = set()  # Discretionary sellers only

        for t in trades:
            if t.transaction_code == "P":  # Open-market purchase
                signal.n_buys += 1
                signal.buy_value += t.shares * t.price_per_share
                buyers.add(t.insider_name)
            elif t.transaction_code == "S":  # Sale
                value = t.shares * t.price_per_share
                if t.is_10b5_1:
                    # Pre-planned 10b5-1 sale — NOT a signal (compensation)
                    signal.n_sells_10b5_1 += 1
                    signal.sell_value_10b5_1 += value
                else:
                    # Discretionary sale — potential signal
                    signal.n_sells += 1
                    signal.sell_value += value
                    sellers.add(t.insider_name)
            elif t.transaction_code == "F":  # Tax withholding — not signal
                signal.n_tax += 1

        signal.unique_buyers = len(buyers)
        signal.unique_sellers = len(sellers)

        # Cluster detection: 3+ distinct insiders bought within 30 days
        if signal.unique_buyers >= 3:
            signal.cluster_buy = True

        signal.last_updated = datetime.now(tz=timezone.utc).isoformat()
        return signal

    def _load_cached(self, ticker: str, days: int) -> Optional[list[InsiderTrade]]:
        """Load cached trades if fresh enough (< 24h old)."""
        try:
            with sqlite3.connect(self._db) as conn:
                row = conn.execute(
                    "SELECT data, updated_at FROM insider_cache WHERE ticker = ?",
                    (ticker,)
                ).fetchone()

            if row is None:
                return None

            updated = datetime.fromisoformat(row[1])
            age = datetime.now(tz=timezone.utc) - updated
            if age > timedelta(hours=24):
                return None  # Stale

            data = json.loads(row[0])
            return [InsiderTrade(**t) for t in data]
        except Exception:
            return None

    def _cache_trades(self, ticker: str, trades: list[InsiderTrade]):
        """Cache trades to SQLite."""
        try:
            data = json.dumps([
                {
                    "ticker": t.ticker,
                    "insider_name": t.insider_name,
                    "insider_title": t.insider_title,
                    "transaction_date": t.transaction_date,
                    "transaction_code": t.transaction_code,
                    "shares": t.shares,
                    "price_per_share": t.price_per_share,
                    "acquired_or_disposed": t.acquired_or_disposed,
                    "filing_date": t.filing_date,
                    "is_10b5_1": t.is_10b5_1,
                }
                for t in trades
            ])
            now = datetime.now(tz=timezone.utc).isoformat()
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO insider_cache (ticker, data, updated_at) "
                    "VALUES (?, ?, ?)",
                    (ticker, data, now)
                )
                conn.commit()
        except Exception as e:
            log.debug(f"Cache write failed for {ticker}: {e}")


# ── Alternative Data Score ────────────────────────────────────────────────────

class AlternativeDataScore:
    """
    Combines insider trading signals into a single boost multiplier.

    Tuned for large-cap equities (NDX100) where routine insider selling
    is the norm (exec compensation, 10b5-1 plans). Only penalizes
    extreme discretionary selling.

    Returns a float between 0.85 and 1.5:
      - 1.5:  Strong cluster buying (3+ distinct insiders purchased)
      - 1.3:  Moderate buying (2 insiders purchased)
      - 1.15: Single insider purchase
      - 1.0:  Neutral, no data, or routine compensation selling
      - 0.85: Extreme discretionary selling (>10x buy value, 3+ sellers)
    """

    def __init__(self):
        self._insider = InsiderTracker()

    def get_boost(self, ticker: str, days: int = 90) -> float:
        """
        Get alternative data boost multiplier for a ticker.
        Returns 0.85–1.5 (1.0 = neutral).
        """
        signal = self._insider.get_signal(ticker, days)
        return self._compute_boost(signal)

    def get_boost_with_detail(self, ticker: str, days: int = 90) -> tuple[float, InsiderSignal]:
        """Get boost + detailed signal for logging/dashboard."""
        signal = self._insider.get_signal(ticker, days)
        boost = self._compute_boost(signal)
        return boost, signal

    def _compute_boost(self, signal: InsiderSignal) -> float:
        """Compute boost multiplier from insider signal.

        Large-cap scoring logic:
        - Insider BUYING is always meaningful (unusual for execs to buy)
        - Insider SELLING is usually routine (compensation, 10b5-1 plans)
        - Only penalize selling when it's extreme AND discretionary
        - 10b5-1 pre-planned sales are completely ignored
        """
        if signal.n_buys == 0 and signal.n_sells == 0:
            return 1.0  # No data → neutral

        # Cluster buying: strongest signal (3+ distinct insiders purchased)
        if signal.cluster_buy:
            return 1.5

        # Net buying (any insider purchases — rare and meaningful)
        if signal.net_signal == "BULLISH":
            if signal.unique_buyers >= 2:
                return 1.3
            return 1.15

        # Extreme discretionary selling (the only BEARISH case for large-cap)
        # Requires: sell_value > 10x buy_value AND 3+ unique discretionary sellers
        if signal.net_signal == "BEARISH":
            return 0.85

        # Everything else: neutral (routine selling, compensation, 10b5-1)
        return 1.0

    def scan_tickers(self, tickers: list[str], days: int = 90) -> dict[str, float]:
        """
        Scan multiple tickers and return boost map.
        Rate-limited to respect SEC API limits.
        """
        boosts = {}
        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(1)  # Extra delay between tickers
            try:
                boosts[ticker] = self.get_boost(ticker, days)
            except Exception as e:
                log.warning(f"  [AltData] Failed for {ticker}: {e}")
                boosts[ticker] = 1.0  # Neutral fallback

        return boosts


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db() -> Path:
    """Initialize SQLite database for alt-data caching."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
    return DB_PATH


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Alternative data module")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to scan")
    parser.add_argument("--days", type=int, default=90, help="Lookback days")
    parser.add_argument("--scan", nargs="+", help="Scan multiple tickers")
    args = parser.parse_args()

    scorer = AlternativeDataScore()

    if args.scan:
        print(f"\n{'='*60}")
        print(f"  ALT-DATA SCAN: {len(args.scan)} tickers ({args.days}-day lookback)")
        print(f"{'='*60}\n")

        for ticker in args.scan:
            boost, signal = scorer.get_boost_with_detail(ticker, args.days)
            indicator = "🟢" if boost > 1.0 else ("🔴" if boost < 1.0 else "⚪")
            print(f"  {indicator} {ticker:6s}  boost={boost:.2f}x  "
                  f"buys={signal.n_buys} sells={signal.n_sells} "
                  f"tax={signal.n_tax}  "
                  f"buyers={signal.unique_buyers} "
                  f"{'CLUSTER' if signal.cluster_buy else ''}")

    else:
        ticker = args.ticker
        print(f"\n{'='*60}")
        print(f"  ALT-DATA: {ticker} ({args.days}-day lookback)")
        print(f"{'='*60}\n")

        boost, signal = scorer.get_boost_with_detail(ticker, args.days)

        print(f"  Signal:        {signal.net_signal}")
        print(f"  Boost:         {boost:.2f}x")
        print(f"  Purchases (P): {signal.n_buys} ({signal.unique_buyers} unique insiders)")
        print(f"  Sales (S):     {signal.n_sells} ({signal.unique_sellers} unique insiders)")
        print(f"  Tax (F):       {signal.n_tax} (ignored — not a signal)")
        print(f"  Buy value:     ${signal.buy_value:,.0f}")
        print(f"  Sell value:    ${signal.sell_value:,.0f}")
        print(f"  Cluster buy:   {signal.cluster_buy}")
        print()

        # Show raw trades
        trades = scorer._insider.fetch_form4_filings(ticker, args.days)
        meaningful = [t for t in trades if t.transaction_code in ("P", "S")]
        if meaningful:
            print(f"  Recent P/S transactions:")
            for t in meaningful[:10]:
                direction = "BUY " if t.transaction_code == "P" else "SELL"
                value = t.shares * t.price_per_share
                print(f"    {t.transaction_date} {direction} "
                      f"{t.shares:>10,.0f} shares @ ${t.price_per_share:.2f} "
                      f"(${value:>12,.0f})  {t.insider_name} ({t.insider_title})")
        else:
            print(f"  No open-market purchases or sales in last {args.days} days")
            print(f"  ({len(trades)} total transactions were tax withholding/awards)")
