#!/usr/bin/env python3
"""
Lakera Guard Evaluator (Large Dataset Optimized, Single-File Output + Summary)

- Streams CSV/JSONL/Parquet/JSON in chunks
- Concurrency-limited async requests with keep-alive
- Retries with exponential backoff + jitter
- Append-only output with --resume (per-dataset resume in a single CSV)
- Quiet by default; optional progress bar (--progress)
- Single merged CSV via --output; adds a `dataset` column
- Confusion matrix + metrics via --summary-out
"""

import os
import sys
import argparse
import asyncio
import time
import random
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Iterable, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required deps with better error handling
def check_dependencies():
    """Check and import required dependencies."""
    missing_deps = []
    
    global aiohttp, pd, tqdm, TQDM_AVAILABLE
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        print(f"Missing required dependencies: {', '.join(missing_deps)}", file=sys.stderr)
        print("Install with: pip install " + " ".join(missing_deps), file=sys.stderr)
        sys.exit(1)
    
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        logger.debug("tqdm not available, progress bars disabled")
    
    return aiohttp, pd

# Check dependencies at module load
aiohttp, pd = check_dependencies()

# ---------------------------- Configuration ----------------------------

class Config:
    """Configuration constants and defaults."""
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CONCURRENCY = 50  # Optimized for 100k+ prompt processing
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_RETRIES = 6
    DEFAULT_BACKOFF_BASE = 0.5
    DEFAULT_BACKOFF_MAX = 10.0
    DEFAULT_OUTPUT_DIR = "lakera_results"
    API_URL = "https://api.lakera.ai/v2/guard/results"

    # Rate limiting (requests per second)
    DEFAULT_RATE_LIMIT = None  # None = no limit, set to avoid API quota exhaustion
    
    # Threat and safe keywords for label parsing
    THREAT_KEYWORDS = [
        "jailbreak", "adversarial", "attack", "prompt_injection", "injection",
        "harmful", "dangerous", "toxic", "malicious", "threatening", "violent",
        "inappropriate", "unsafe", "illegal", "criminal", "unethical",
        "true", "yes", "1", "positive", "threat"
    ]
    
    SAFE_KEYWORDS = [
        "benign", "safe", "harmless", "acceptable", "appropriate", "ethical",
        "legal", "legitimate", "allowed", "clean", "wholesome", "normal",
        "false", "no", "0", "negative"
    ]
    
    # Column name candidates
    TEXT_COLUMN_CANDIDATES = [
        "prompt", "text", "content", "message", "input", "query", "goal", 
        "instruction", "conversations", "chat", "dialogue", "messages", 
        "question", "user_input", "request", "task"
    ]
    
    LABEL_COLUMN_CANDIDATES = [
        "type", "label", "labels", "ground_truth", "groundtruth", "gt", 
        "class", "category", "classification", "target", "y", "output", 
        "is_safe", "safety", "risk", "harmful", "toxicity", "jailbreak", 
        "adversarial", "behavior", "behaviour", "intent", "malicious", "benign"
    ]


# ---------------------------- Helpers ----------------------------

def parse_boolish_label(label) -> Optional[bool]:
    """Map common dataset labels to True(THREAT)/False(SAFE)/None."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    
    if isinstance(label, bool):
        return label
    
    if isinstance(label, (int, float)):
        return bool(int(label))
    
    s = str(label).strip().lower()
    
    # Check threat keywords
    if any(t in s for t in Config.THREAT_KEYWORDS):
        return True
    
    # Check safe keywords
    if any(t in s for t in Config.SAFE_KEYWORDS):
        return False
    
    return None


def analyze_prediction(results: List[Dict]) -> str:
    """Reduce Lakera detector results -> 'THREAT [...]' or 'SAFE'."""
    if not results:
        return "SAFE"
    
    hits = []
    threat_levels = {"l1_confident", "l2_very_likely", "l3_likely"}
    
    for d in results:
        level = d.get("result", "l5_unlikely")
        if level in threat_levels:
            detector_type = d.get("detector_type", "unknown")
            hits.append(f"{detector_type}({level})")
    
    return f"THREAT [{', '.join(hits)}]" if hits else "SAFE"


def ensure_out_path(input_path: str, output_dir: str = Config.DEFAULT_OUTPUT_DIR,
                    explicit_output: Optional[str] = None,
                    output_prefix: str = "lakera") -> Path:
    """Ensure output path exists and return it."""
    if explicit_output:
        out_path = Path(explicit_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outdir / f"{output_prefix}_{stem}_{ts}.csv"


def read_last_index_for_dataset(out_path: Path, dataset_name: str) -> int:
    """Return next index for this dataset when resuming a merged CSV.

    Optimized to read from end of file for large datasets (100k+ rows).
    """
    if not out_path.exists():
        return 0

    try:
        import csv

        # Read last N lines efficiently (tail-like approach)
        lines_to_check = 10000  # Check last 10k lines for this dataset

        with out_path.open("rb") as f:
            # Get file size
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            if file_size == 0:
                return 0

            # Read from end in chunks
            buffer_size = min(1024 * 1024, file_size)  # 1MB buffer
            f.seek(max(0, file_size - buffer_size))

            # Read and decode
            tail_data = f.read().decode('utf-8', errors='ignore')

        # Split into lines and process from end
        lines = tail_data.split('\n')

        # Find header to get column indices
        header_line = None
        with out_path.open("r", encoding="utf-8") as f:
            header_line = f.readline().strip()

        if not header_line:
            return 0

        reader = csv.DictReader([header_line] + lines[-lines_to_check:])

        last = -1
        for row in reader:
            try:
                if row.get("dataset") == dataset_name:
                    idx = int(row.get("index", -1))
                    last = max(last, idx)
            except (ValueError, TypeError):
                continue

        return last + 1 if last >= 0 else 0
    except Exception as e:
        logger.warning(f"Could not read last index for resume: {e}")
        return 0


def write_rows(out_path: Path, rows: List[Dict], header_written: bool) -> bool:
    """Append rows to CSV. Returns whether header was written."""
    import csv
    
    if not rows:
        return header_written
    
    columns = ["dataset", "index", "prompt", "ground_truth", "prediction", 
               "latency_ms", "detector_count"]
    
    write_header = not header_written and not out_path.exists()
    
    try:
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})
        return True
    except Exception as e:
        logger.error(f"Failed to write rows: {e}")
        raise


# ---------------------------- Readers (chunked) ----------------------------

def detect_columns(df: pd.DataFrame, text_col: Optional[str], 
                  label_col: Optional[str]) -> Tuple[str, Optional[str]]:
    """Detect text/label columns on the first chunk; reuse thereafter."""
    
    # Detect text column
    if text_col and text_col in df.columns:
        tc = text_col
    else:
        # Try exact matches first
        lowcols = {c.lower(): c for c in df.columns}
        tc = None
        
        for candidate in Config.TEXT_COLUMN_CANDIDATES:
            if candidate in lowcols:
                tc = lowcols[candidate]
                break
        
        # If no match, use heuristic by length
        if tc is None:
            best, best_len = None, -1
            for col in df.columns:
                if df[col].dtype == "object":
                    sample = df[col].dropna().astype(str)
                    if len(sample) == 0:
                        continue
                    avg_len = sample.head(min(50, len(sample))).str.len().mean()
                    if avg_len > best_len:
                        best, best_len = col, avg_len
            
            if best is None:
                raise ValueError(f"Could not detect text column. Available: {list(df.columns)}")
            tc = best
    
    # Detect label column
    lc = None
    if label_col and label_col in df.columns:
        lc = label_col
    else:
        lowcols = {c.lower(): c for c in df.columns}
        
        for candidate in Config.LABEL_COLUMN_CANDIDATES:
            if candidate in lowcols:
                lc = lowcols[candidate]
                break
        
        # Content-based detection if no exact match
        if lc is None:
            for col in df.columns:
                if col == tc or df[col].dtype != "object":
                    continue
                
                sample_vals = df[col].dropna().head(min(50, len(df[col]))).tolist()
                if not sample_vals:
                    continue
                
                vals = [str(v).lower() for v in sample_vals]
                label_indicators = ["safe", "unsafe", "jailbreak", "benign", 
                                   "harmful", "true", "false", "0", "1"]
                
                if any(any(indicator in v for indicator in label_indicators) for v in vals):
                    lc = col
                    break
    
    logger.info(f"Detected columns - Text: {tc}, Label: {lc}")
    return tc, lc


def iter_csv_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """Iterate over CSV file in chunks."""
    try:
        for df in pd.read_csv(path, chunksize=chunksize):
            yield df
    except Exception as e:
        logger.error(f"Error reading CSV {path}: {e}")
        raise


def iter_jsonl_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """Iterate over JSONL file in chunks."""
    buf = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    buf.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
                
                if len(buf) >= chunksize:
                    yield pd.DataFrame(buf)
                    buf = []
        
        if buf:
            yield pd.DataFrame(buf)
    except Exception as e:
        logger.error(f"Error reading JSONL {path}: {e}")
        raise


def iter_parquet_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """Iterate over Parquet file in chunks."""
    try:
        # Try using pyarrow for efficient row group reading
        import pyarrow.parquet as pq
        
        pf = pq.ParquetFile(path)
        batch = []
        
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            df = table.to_pandas()
            batch.append(df)
            
            while sum(len(x) for x in batch) >= chunksize:
                accumulated = pd.concat(batch, ignore_index=True)
                chunk = accumulated.iloc[:chunksize].copy()
                yield chunk
                remaining = accumulated.iloc[chunksize:].copy()
                batch = [remaining] if len(remaining) > 0 else []
        
        if batch and any(len(x) > 0 for x in batch):
            yield pd.concat(batch, ignore_index=True)
            
    except ImportError:
        # Fallback to pandas if pyarrow not available
        logger.info("pyarrow not available, using pandas for parquet reading")
        df = pd.read_parquet(path)
        for start in range(0, len(df), chunksize):
            yield df.iloc[start:start+chunksize].copy()
    except Exception as e:
        logger.error(f"Error reading Parquet {path}: {e}")
        raise


def chunk_iter_for(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """Get appropriate chunk iterator based on file extension."""
    ext = Path(path).suffix.lower()
    
    readers = {
        ".csv": iter_csv_chunks,
        ".jsonl": iter_jsonl_chunks,
        ".parquet": iter_parquet_chunks,
    }
    
    if ext in readers:
        return readers[ext](path, chunksize)
    elif ext == ".json":
        df = pd.read_json(path)
        return (df.iloc[i:i+chunksize].copy() for i in range(0, len(df), chunksize))
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .jsonl, .json, .parquet")


# ---------------------------- HTTP Client ----------------------------

class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, rate_limit: Optional[float]):
        """
        Initialize rate limiter.

        Args:
            rate_limit: Maximum requests per second (None = no limit)
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        if self.rate_limit is None:
            return

        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            min_interval = 1.0 / self.rate_limit

            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


class LakeraClient:
    """Async HTTP client for Lakera Guard API."""

    def __init__(self, api_key: str, project_id: Optional[str], timeout_s: float,
                 concurrency: int, retries: int, backoff_base: float, backoff_max: float,
                 rate_limit: Optional[float] = None):
        self.api_key = api_key
        self.project_id = project_id
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.url = Config.API_URL
        self.sem = asyncio.Semaphore(concurrency)
        self.rate_limiter = RateLimiter(rate_limit)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        conn = aiohttp.TCPConnector(
            limit=0,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=conn,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            trust_env=True,  # Honor proxy environment variables
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            # Small delay to ensure connections are closed
            await asyncio.sleep(0.25)

    async def _once(self, prompt: str) -> Tuple[List, float, Optional[str], int]:
        """Single API request attempt."""
        if self.session is None:
            raise RuntimeError("Session not initialized")
        
        body = {"messages": [{"role": "user", "content": prompt}]}
        if self.project_id:
            body["project_id"] = self.project_id
        
        t0 = time.time()
        
        try:
            async with self.session.post(self.url, json=body) as resp:
                status = resp.status
                latency = time.time() - t0
                
                if status == 200:
                    data = await resp.json()
                    return data.get("results", []), latency, None, status
                else:
                    try:
                        text = await resp.text()
                    except Exception:
                        text = f"HTTP {status}"
                    return [], 0.0, text, status
        except Exception as e:
            return [], 0.0, str(e), 500

    async def check(self, prompt: str) -> Dict:
        """Check prompt with retries and backoff."""
        async with self.sem:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Add micro-jitter to prevent thundering herd
            await asyncio.sleep(random.uniform(0, 0.05))

            delay = self.backoff_base
            last_error = None

            for attempt in range(1, self.retries + 1):
                try:
                    results, latency, err_text, status = await self._once(prompt)
                    
                    if status == 200:
                        return {
                            "ok": True, 
                            "results": results, 
                            "latency": latency
                        }
                    
                    # Retry on rate limit or server errors
                    if status in (429, 500, 502, 503, 504):
                        last_error = f"HTTP {status}: {err_text or 'Server error'}"
                        if attempt < self.retries:
                            jittered_delay = min(delay, self.backoff_max) + random.uniform(0, 0.25)
                            await asyncio.sleep(jittered_delay)
                            delay *= 2
                        continue
                    
                    # Non-retryable error
                    return {
                        "ok": False, 
                        "error": f"{status}: {err_text or 'error'}"
                    }
                    
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = str(e)
                    if attempt < self.retries:
                        jittered_delay = min(delay, self.backoff_max) + random.uniform(0, 0.25)
                        await asyncio.sleep(jittered_delay)
                        delay *= 2
                    continue
            
            return {
                "ok": False, 
                "error": f"Max retries exceeded. Last error: {last_error}"
            }


# ---------------------------- Main processing ----------------------------

def estimate_total_rows(path: str) -> Optional[int]:
    """Estimate total rows in file for progress tracking."""
    try:
        ext = Path(path).suffix.lower()

        if ext == ".csv":
            # Quick line count for CSV
            with open(path, 'rb') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        elif ext == ".jsonl":
            with open(path, 'rb') as f:
                return sum(1 for _ in f)
        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
                return pq.read_metadata(path).num_rows
            except ImportError:
                df = pd.read_parquet(path)
                return len(df)
        elif ext == ".json":
            df = pd.read_json(path)
            return len(df)
    except Exception as e:
        logger.debug(f"Could not estimate total rows: {e}")

    return None


async def process_file(
    path: str,
    api_key: str,
    project_id: Optional[str],
    text_col: Optional[str],
    label_col: Optional[str],
    max_rows: Optional[int],
    chunksize: int,
    concurrency: int,
    timeout: float,
    retries: int,
    backoff_base: float,
    backoff_max: float,
    out_path: Path,
    resume: bool,
    show_progress: bool,
    dataset_name: str,
    rate_limit: Optional[float] = None
) -> None:
    """Process a single file with the Lakera API."""

    logger.info(f"Processing {path} as dataset '{dataset_name}'")

    total_written = 0
    header_written = out_path.exists() and out_path.stat().st_size > 0
    start_index = read_last_index_for_dataset(out_path, dataset_name) if resume else 0

    if resume and start_index > 0:
        logger.info(f"Resuming from index {start_index}")

    chunk_iter = chunk_iter_for(path, chunksize)
    detected_text = None
    detected_label = None
    global_index = 0

    # Estimate total rows for progress tracking
    total_rows_estimate = estimate_total_rows(path)
    if max_rows is not None:
        total_rows_estimate = min(total_rows_estimate or max_rows, max_rows)

    if total_rows_estimate and start_index > 0:
        total_rows_estimate = max(0, total_rows_estimate - start_index)

    # Setup progress bar if requested
    pbar = None
    start_time = time.time()
    if show_progress and TQDM_AVAILABLE:
        from tqdm import tqdm
        pbar = tqdm(
            total=total_rows_estimate,
            unit="rows",
            desc=f"Processing {Path(path).name}",
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    try:
        async with LakeraClient(
            api_key, project_id, timeout, concurrency,
            retries, backoff_base, backoff_max, rate_limit
        ) as client:
            
            for df in chunk_iter:
                # Detect columns on first chunk
                if detected_text is None:
                    detected_text, detected_label = detect_columns(df, text_col, label_col)
                
                df = df.reset_index(drop=True)
                indices = list(range(global_index, global_index + len(df)))
                global_index += len(df)
                
                # Prepare rows to process
                rows_to_process = []
                for i_rel, i_abs in enumerate(indices):
                    # Skip already processed rows
                    if i_abs < start_index:
                        continue
                    
                    # Check max rows limit
                    if max_rows is not None and (i_abs - start_index) >= max_rows:
                        break
                    
                    # Extract ground truth if available
                    gt = None
                    if detected_label and detected_label in df.columns:
                        try:
                            gt = parse_boolish_label(df.iloc[i_rel][detected_label])
                        except Exception as e:
                            logger.warning(f"Error parsing label at row {i_abs}: {e}")
                    
                    # Extract text
                    try:
                        text = str(df.iloc[i_rel][detected_text])
                        if text and text != 'nan':
                            rows_to_process.append((i_abs, text, gt))
                    except Exception as e:
                        logger.warning(f"Error extracting text at row {i_abs}: {e}")
                        continue
                
                if not rows_to_process:
                    if max_rows is not None and (global_index - start_index) >= max_rows:
                        break
                    continue
                
                # Create async tasks for API calls
                tasks = [
                    asyncio.create_task(client.check(prompt)) 
                    for (_, prompt, _) in rows_to_process
                ]
                results = await asyncio.gather(*tasks)
                
                # Process results
                out_rows = []
                for (i_abs, prompt, gt), res in zip(rows_to_process, results):
                    if res.get("ok"):
                        pred = analyze_prediction(res.get("results", []))
                        out_rows.append({
                            "dataset": dataset_name,
                            "index": i_abs,
                            "prompt": prompt[:1000],  # Truncate very long prompts
                            "ground_truth": "THREAT" if gt is True else "SAFE" if gt is False else "N/A",
                            "prediction": pred,
                            "latency_ms": f"{res.get('latency', 0)*1000:.2f}",
                            "detector_count": len(res.get("results", []))
                        })
                    else:
                        logger.warning(f"Failed to process row {i_abs}: {res.get('error')}")
                
                # Write results
                if out_rows:
                    header_written = write_rows(out_path, out_rows, header_written)
                    total_written += len(out_rows)
                
                if pbar:
                    pbar.update(len(rows_to_process))
                
                # Check if we've hit max rows
                if max_rows is not None and total_written >= max_rows:
                    logger.info(f"Reached max rows limit ({max_rows})")
                    break
    
    finally:
        if pbar:
            pbar.close()

    # Log performance summary
    elapsed = time.time() - start_time
    throughput = total_written / elapsed if elapsed > 0 else 0
    logger.info(f"Completed {Path(path).name} -> {out_path} | "
                f"dataset={dataset_name} | written rows: {total_written} | "
                f"time: {elapsed:.1f}s | throughput: {throughput:.1f} rows/s")


def write_confusion_summary(out_path: Path, summary_out: Optional[Path]) -> None:
    """Compute confusion matrix & metrics from a (possibly huge) CSV."""
    
    logger.info("Computing confusion matrix and metrics...")
    
    tp = tn = fp = fn = 0
    
    try:
        for df in pd.read_csv(out_path, chunksize=200000, usecols=["ground_truth", "prediction"]):
            # Filter to rows with valid ground truth
            df = df[df["ground_truth"].isin(["SAFE", "THREAT"])].copy()
            
            if df.empty:
                continue
            
            y_true = df["ground_truth"].eq("THREAT")
            y_pred = df["prediction"].astype(str).str.startswith("THREAT")
            
            tp += (y_true & y_pred).sum()
            tn += (~y_true & ~y_pred).sum()
            fp += (~y_true & y_pred).sum()
            fn += (y_true & ~y_pred).sum()
    
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return
    
    total = tp + tn + fp + fn
    
    if total == 0:
        logger.warning("No valid labeled data found for metrics computation")
        return
    
    # Compute metrics with safe division
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"  True Positives:  {tp:8d}")
    print(f"  False Positives: {fp:8d}")
    print(f"  False Negatives: {fn:8d}")
    print(f"  True Negatives:  {tn:8d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*60}\n")
    
    # Write to file if requested
    if summary_out:
        import csv
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with summary_out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=["tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"]
                )
                writer.writeheader()
                writer.writerow({
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "accuracy": f"{acc:.6f}",
                    "precision": f"{prec:.6f}",
                    "recall": f"{rec:.6f}",
                    "f1": f"{f1:.6f}"
                })
            logger.info(f"Summary written to {summary_out}")
        except Exception as e:
            logger.error(f"Failed to write summary: {e}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Lakera Guard Evaluator - Production-ready dataset evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single dataset
  %(prog)s --datasets data.csv --env YOUR_API_KEY
  
  # Process multiple datasets with resume capability
  %(prog)s --datasets train.csv test.csv --resume --progress
  
  # Custom output and summary
  %(prog)s --datasets data.jsonl --output results.csv --summary-out metrics.csv
        """
    )
    
    # Required arguments
    parser.add_argument("--datasets", nargs="+", required=True, 
                       help="Input files (.csv, .jsonl, .json, .parquet)")
    
    # API configuration
    parser.add_argument("--env", help="Lakera API key (fallback to LAKERA_API_KEY env var)")
    parser.add_argument("--project-id", help="Lakera project ID (fallback to LAKERA_PROJECT_ID env var)")
    
    # Column configuration
    parser.add_argument("--text-column", help="Text column name (auto-detected if not specified)")
    parser.add_argument("--label-column", help="Label column name (auto-detected if not specified)")
    
    # Processing configuration
    parser.add_argument("--max-rows", type=int, help="Process at most N rows per dataset")
    parser.add_argument("--chunksize", type=int, default=Config.DEFAULT_CHUNK_SIZE,
                       help=f"Streaming chunk size (default: {Config.DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--concurrency", type=int, default=Config.DEFAULT_CONCURRENCY,
                       help=f"Concurrent HTTP requests (default: {Config.DEFAULT_CONCURRENCY}). "
                            f"Higher values (50-100) recommended for 100k+ datasets. "
                            f"Lower (8-20) if hitting rate limits.")
    
    # Network configuration
    parser.add_argument("--timeout", type=float, default=Config.DEFAULT_TIMEOUT,
                       help=f"Per-request timeout seconds (default: {Config.DEFAULT_TIMEOUT})")
    parser.add_argument("--retries", type=int, default=Config.DEFAULT_RETRIES,
                       help=f"Max retries per request (default: {Config.DEFAULT_RETRIES})")
    parser.add_argument("--backoff-base", type=float, default=Config.DEFAULT_BACKOFF_BASE,
                       help=f"Base backoff seconds (default: {Config.DEFAULT_BACKOFF_BASE})")
    parser.add_argument("--backoff-max", type=float, default=Config.DEFAULT_BACKOFF_MAX,
                       help=f"Max backoff seconds (default: {Config.DEFAULT_BACKOFF_MAX})")
    parser.add_argument("--rate-limit", type=float, default=Config.DEFAULT_RATE_LIMIT,
                       help="Rate limit in requests per second (default: None = no limit). "
                            "Use to prevent API quota exhaustion (e.g., --rate-limit 10)")
    
    # Output configuration
    parser.add_argument("--output", help="Explicit output CSV path (merges all datasets)")
    parser.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                       help=f"Output directory (default: {Config.DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--output-prefix", default="lakera",
                       help="Prefix for output filenames (default: lakera). "
                            "Output format: {prefix}_{filename}_{timestamp}.csv")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last processed row (append mode)")
    parser.add_argument("--progress", action="store_true",
                       help="Show progress bar (requires tqdm)")
    parser.add_argument("--dataset-name", help="Override dataset name (defaults to input filename)")
    parser.add_argument("--summary-out", help="Write confusion matrix/metrics CSV to this path")
    
    # Logging configuration
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Get API key
    api_key = args.env or os.getenv("LAKERA_API_KEY")
    if not api_key:
        print("Error: API key required. Provide via --env or LAKERA_API_KEY environment variable", file=sys.stderr)
        sys.exit(2)
    
    project_id = args.project_id or os.getenv("LAKERA_PROJECT_ID")
    
    # Validate input files
    valid_files = []
    for path in args.datasets:
        if not Path(path).exists():
            logger.error(f"File not found: {path}")
        else:
            valid_files.append(path)
    
    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)
    
    # Process each dataset
    try:
        for path in valid_files:
            out_path = ensure_out_path(path, args.output_dir, args.output, args.output_prefix)
            dataset_name = args.dataset_name or Path(path).stem

            logger.info(f"Starting processing of {path}")
            
            asyncio.run(process_file(
                path=path,
                api_key=api_key,
                project_id=project_id,
                text_col=args.text_column,
                label_col=args.label_column,
                max_rows=args.max_rows,
                chunksize=args.chunksize,
                concurrency=args.concurrency,
                timeout=args.timeout,
                retries=args.retries,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
                out_path=out_path,
                resume=args.resume,
                show_progress=args.progress,
                dataset_name=dataset_name,
                rate_limit=args.rate_limit
            ))
            
    except KeyboardInterrupt:
        print("\n[Interrupt] Stopping gracefully...")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    
    # Generate summary if requested
    if args.output and Path(args.output).exists():
        write_confusion_summary(
            Path(args.output), 
            Path(args.summary_out) if args.summary_out else None
        )
    
    logger.info("All processing complete")


if __name__ == "__main__":
    main()
