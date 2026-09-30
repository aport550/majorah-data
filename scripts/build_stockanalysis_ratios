#!/usr/bin/env python3
"""Download the latest 20 available historical quarterly ratio columns.

Python 3.9+. Install: python3 -m pip install pandas requests lxml
Run from project root: python3 build_stockanalysis_ratios.py
Test: python3 build_stockanalysis_ratios.py --tickers NVDA AAPL
Slower: python3 build_stockanalysis_ratios.py --sleep 5

Input: data/universe.csv, required column Ticker.
Optional StockAnalysisPath column overrides the stock page, e.g. /stocks/brk-b/
or /quote/tsx/SHOP/. Otherwise dots in tickers become hyphens in /stocks/{ticker}/.
No API key. Requests are sequential with 3-4 seconds between requests by default.
Successful pages are cached for 7 days; --refresh ignores the cache.

Output: data/stockanalysis_ratios.csv (one row per ticker/fiscal quarter).
Current/TTM columns excluded. quarter_rank=1 means latest historical quarter.
Missing source values stay blank, including negative/zero values if supplied.
No invented quarters for recent IPOs or incomplete source coverage.
Coverage/errors: data/stockanalysis_ratios_status.csv.
Up to 20 quarterly columns PER STOCK, not 20 nonmissing values per metric.
These are website-reported ratios, not a vintage point-in-time backtest dataset.
Buyback yield is stored in percentage points: 2.13% becomes 2.13, not 0.0213.
Negative buyback yield values are preserved (dilution).
Period labels are fiscal, so fiscal years may differ from calendar years.

429 responses respect Retry-After; persistent 429 or 401/403 stops the run.
No proxies, browser impersonation, paywall bypasses, or concurrent requests.
Saved checkpoints and HTML cache let reruns avoid repeating successful requests.
For GitHub Actions, persist data/stockanalysis_cache with actions/cache if desired.
"""
import argparse
import datetime as dt
from email.utils import parsedate_to_datetime
import hashlib
from io import StringIO
import math
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlparse

import pandas as pd
import requests

METRICS = {
    'forward_pe': ('forward pe', 'forward p/e', 'forward pe ratio'),
    'ps_ratio': ('ps ratio', 'p/s ratio'),
    'peg_ratio': ('peg ratio', 'peg'),
    'pb_ratio': ('pb ratio', 'p/b ratio'),
    'ev_ebitda': ('ev/ebitda ratio', 'ev/ebitda'),
    'ev_sales': ('ev/sales ratio', 'ev/sales'),
    'ev_fcf': ('ev/fcf ratio', 'ev/fcf'),
    'current_ratio': ('current ratio',),
    'buyback_yield_dilution_pct': ('buyback yield / dilution', 'buyback yield/dilution'),
}
COLUMNS = ['Ticker', 'quarter_rank', 'fiscal_quarter', 'period_end', *METRICS,
           'source_url', 'fetched_at_utc']
STATUS_COLUMNS = ['Ticker', 'status', 'quarters', 'missing_metrics', 'missing_cells',
                  'source_url', 'cache_used', 'message']


class StopRun(RuntimeError):
    pass


def normalize(value):
    return re.sub(r'\s+', ' ', str(value).replace('\xa0', ' ')).strip().lower()


def numeric(value):
    text = str(value).strip().replace(',', '').replace('−', '-').removesuffix('%')
    if text in ('', '-', '—', '–', 'N/A', 'n/a', 'nan', 'None'):
        return None
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    try:
        result = float(text)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def parts(column):
    return list(column) if isinstance(column, tuple) else [column]


def full_date(values):
    for value in reversed(values):
        match = re.search(r'\b([A-Z][a-z]{2,8} \d{1,2}, \d{4})\b', str(value))
        if match:
            parsed = pd.to_datetime(match.group(1), errors='coerce')
            if pd.notna(parsed):
                return parsed.date().isoformat()
        match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', str(value))
        if match:
            return match.group(1)
    return ''


def parse_tables(html, ticker, url, fetched):
    tables = pd.read_html(StringIO(html), flavor='lxml')
    by_quarter, seen_metrics = {}, set()
    alias_map = {alias: metric for metric, aliases in METRICS.items() for alias in aliases}
    for table in tables:
        if table.empty:
            continue
        period_row = next((row for _, row in table.iterrows()
                           if normalize(row.iloc[0]) == 'period ending'), None)
        for _, row in table.iterrows():
            metric = alias_map.get(normalize(row.iloc[0]))
            if not metric:
                continue
            seen_metrics.add(metric)
            for col_index in range(1, len(table.columns)):
                headers = parts(table.columns[col_index])
                quarter = next((str(x).strip() for x in headers
                                if re.fullmatch(r'Q[1-4]\s+\d{4}', str(x).strip())), None)
                if quarter is None:
                    continue  # excludes Current, TTM, annual columns, ads
                candidates = headers + ([period_row.iloc[col_index]] if period_row is not None else [])
                date = full_date(candidates)
                record = by_quarter.setdefault(quarter, {
                    'Ticker': ticker, 'fiscal_quarter': quarter, 'period_end': date,
                    **{m: None for m in METRICS}, 'source_url': url, 'fetched_at_utc': fetched,
                })
                if date:
                    if record['period_end'] and date != record['period_end']:
                        raise ValueError('Conflicting period dates across tables')
                    record['period_end'] = date
                record[metric] = numeric(row.iloc[col_index])
    if not by_quarter or not seen_metrics:
        raise ValueError('No expected quarterly ratio tables found; layout/access may have changed')
    # Fiscal year/quarter ordering works for non-calendar fiscal years too.
    ordered = sorted(by_quarter.values(),
                     key=lambda r: (int(r['fiscal_quarter'].split()[1]), int(r['fiscal_quarter'][1])),
                     reverse=True)[:20]
    for rank, row in enumerate(ordered, 1):
        row['quarter_rank'] = rank
    if not any(r[m] is not None for r in ordered for m in METRICS):
        raise ValueError('Quarterly labels found but all nine metrics are missing')
    return ordered, sorted(set(METRICS) - seen_metrics)


def source_url(ticker, override):
    if pd.notna(override) and str(override).strip():
        path = str(override).strip()
        if path.startswith('https://'):
            parsed = urlparse(path)
            if parsed.netloc != 'stockanalysis.com':
                raise ValueError('StockAnalysisPath must point to stockanalysis.com')
            path = parsed.path
        if not re.fullmatch(r'/(?:stocks/[A-Za-z0-9.-]+|quote/[A-Za-z0-9.-]+/[A-Za-z0-9.-]+)(?:/financials/ratios)?/?', path):
            raise ValueError('Invalid StockAnalysisPath; use /stocks/nvda/ or /quote/tsx/SHOP/')
    else:
        symbol = ticker.lower().replace('.', '-')
        if not re.fullmatch(r'[a-z0-9-]+', symbol):
            raise ValueError('Unsupported ticker format; supply StockAnalysisPath')
        path = '/stocks/' + symbol + '/'
    path = path.rstrip('/')
    if not path.endswith('/financials/ratios'):
        path += '/financials/ratios'
    return 'https://stockanalysis.com' + path + '/?p=quarterly'


class Fetcher:
    def __init__(self, args):
        self.args = args
        self.cache = Path(args.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Majorah-Ratio-Research/1.0'
        self.last_request = None

    def get(self, url):
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + '.html')
        if not self.args.refresh and path.exists() and time.time()-path.stat().st_mtime < self.args.cache_days*86400:
            fetched = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()
            return path.read_text(encoding='utf-8'), fetched, True, path
        for attempt in range(3):
            if self.last_request is not None:
                delay = self.args.sleep + random.uniform(0, 1)
                time.sleep(max(0, delay-(time.monotonic()-self.last_request)))
            try:
                response = self.session.get(url, timeout=40)
            except requests.RequestException as exc:
                self.last_request = time.monotonic()
                if attempt == 2:
                    raise RuntimeError('Network failure: ' + type(exc).__name__) from None
                time.sleep(5 * (attempt+1))
                continue
            self.last_request = time.monotonic()
            if response.status_code in (401, 403):
                raise StopRun(f'HTTP {response.status_code}: access blocked; stopping requests')
            if response.status_code == 429:
                if attempt == 2:
                    raise StopRun('Persistent HTTP 429; stopping requests, rerun later')
                header = response.headers.get('Retry-After', '')
                try:
                    wait = float(header)
                except ValueError:
                    try:
                        wait = (parsedate_to_datetime(header) - dt.datetime.now(dt.timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        wait = 30 * (attempt+1)
                wait = max(30, wait)
                if wait > 300:
                    raise StopRun(f'HTTP 429 requests {wait:.0f}s cooldown; stopping, rerun later')
                print(f'  Rate limited; waiting {wait:.0f}s', flush=True)
                # Short sleep chunks keep Ctrl-C responsive.
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    time.sleep(min(5, max(0, deadline-time.monotonic())))
                continue
            if response.status_code >= 500 and attempt < 2:
                time.sleep(5 * (attempt+1))
                continue
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}')
            if urlparse(response.url).path.rstrip('/') != urlparse(url).path.rstrip('/'):
                raise ValueError('Unexpected redirect; check ticker / StockAnalysisPath')
            return response.text, dt.datetime.now(dt.timezone.utc).isoformat(), False, path
        raise RuntimeError('Request retries exhausted')


def atomic_csv(rows, columns, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(temp, index=False)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--universe', default='data/universe.csv')
    parser.add_argument('--tickers', nargs='+', help='Small test instead of universe CSV')
    parser.add_argument('--output', default='data/stockanalysis_ratios.csv')
    parser.add_argument('--status-output', default='data/stockanalysis_ratios_status.csv')
    parser.add_argument('--sleep', type=float, default=3, help='Minimum seconds between requests (default 3 plus jitter)')
    parser.add_argument('--cache-dir', default='data/stockanalysis_cache')
    parser.add_argument('--cache-days', type=float, default=7)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.sleep < 0 or args.cache_days < 0:
        parser.error('sleep and cache-days must be nonnegative')
    if Path(args.output).resolve() == Path(args.status_output).resolve():
        parser.error('Output and status paths must differ')
    universe = pd.DataFrame({'Ticker': args.tickers}) if args.tickers else pd.read_csv(args.universe, dtype=str)
    if 'Ticker' not in universe:
        parser.error('Universe requires a Ticker column')
    universe = universe.dropna(subset=['Ticker']).copy()
    universe['Ticker'] = universe.Ticker.str.strip().str.upper()
    universe = universe[universe.Ticker != ''].drop_duplicates('Ticker')
    records = universe.to_dict('records')
    if not records:
        parser.error('No tickers supplied')
    fetcher = Fetcher(args)
    rows, statuses = [], []
    stopped = False
    try:
        for index, record in enumerate(records, 1):
            ticker = record['Ticker']
            status = {'Ticker': ticker, 'status': 'failed', 'quarters': 0, 'source_url': ''}
            print(f'[{index}/{len(records)}] {ticker}', flush=True)
            try:
                url = source_url(ticker, record.get('StockAnalysisPath'))
                status['source_url'] = url
                html, fetched, cached, path = fetcher.get(url)
                status['cache_used'] = cached
                result, missing = parse_tables(html, ticker, url, fetched)
                if not cached:
                    temp = path.with_suffix('.tmp')
                    temp.write_text(html, encoding='utf-8')
                    temp.replace(path)
                rows.extend(result)
                blanks = sum(r[m] is None for r in result for m in METRICS)
                status.update(status='ok' if len(result) == 20 and not blanks else 'partial',
                              quarters=len(result), missing_metrics=';'.join(missing), missing_cells=blanks,
                              message='' if all(r['period_end'] for r in result) else 'Some exact period dates unavailable')
                print(f'  {len(result)} quarters, {blanks} blank cells' + (' [cached]' if cached else ''), flush=True)
            except StopRun as exc:
                status.update(status='blocked', message=str(exc))
                statuses.append(status)
                stopped = True
                print('  ' + str(exc), flush=True)
                break
            except (ValueError, RuntimeError, ImportError) as exc:
                status['message'] = str(exc)
                print('  Skipped: ' + str(exc), flush=True)
            statuses.append(status)
            if index % 10 == 0:
                if rows:
                    atomic_csv(rows, COLUMNS, args.output)
                atomic_csv(statuses, STATUS_COLUMNS, args.status_output)
    except KeyboardInterrupt:
        stopped = True
        print('\nInterrupted; saving completed tickers.', flush=True)
    finally:
        completed = {s['Ticker'] for s in statuses}
        statuses.extend({'Ticker': r['Ticker'], 'status': 'not_attempted', 'message': 'Run stopped early'}
                        for r in records if r['Ticker'] not in completed)
        if rows:
            atomic_csv(rows, COLUMNS, args.output)
        atomic_csv(statuses, STATUS_COLUMNS, args.status_output)
        fetcher.session.close()
    successes = sum(s['status'] in ('ok', 'partial') for s in statuses)
    print(f'Completed: {successes}/{len(records)} tickers; {len(rows)} historical rows.')
    print(f'Status: {args.status_output}')
    if rows:
        print(f'Data: {args.output} (this run only; check status for omissions)')
    else:
        print('No data written; any previous ratio CSV was left unchanged.')
    return 2 if stopped or not successes else 0


if __name__ == '__main__':
    sys.exit(main())
