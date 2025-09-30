#!/usr/bin/env python3
"""
Lakera Guard Evaluator (Large Dataset Optimized, Single-File Output + Summary)

- Streams CSV/JSONL/Parquet/JSON in chunks
- Concurrency-limited async requests with keep-alive
- Retries with exponential backoff + jitter
- Append-only output with --resume (per-dataset resume inside a single CSV)
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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Iterable, Tuple

# Required deps
try:
    import aiohttp
except ImportError as e:
    print(f"aiohttp missing: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(f"pandas missing: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm  # optional
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False


# ---------------------------- Helpers ----------------------------

def parse_boolish_label(label) -> Optional[bool]:
    """Map common dataset labels to True(THREAT)/False(SAFE)/None."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    if isinstance(label, bool):
        return label
    if isinstance(label, (int, float)):
        return bool(label)
    s = str(label).strip().lower()
    threat = [
        'jailbreak', 'adversarial', 'attack', 'prompt_injection', 'injection',
        'harmful', 'dangerous', 'toxic', 'malicious', 'threatening', 'violent',
        'inappropriate', 'unsafe', 'illegal', 'criminal', 'unethical',
        'true', 'yes', '1', 'positive', 'threat'
    ]
    safe = [
        'benign', 'safe', 'harmless', 'acceptable', 'appropriate', 'ethical',
        'legal', 'legitimate', 'allowed', 'clean', 'wholesome', 'normal',
        'false', 'no', '0', 'negative'
    ]
    if any(t in s for t in threat):
        return True
    if any(t in s for t in safe):
        return False
    return None


def analyze_prediction(results: List[Dict]) -> str:
    """Reduce Lakera detector results -> 'THREAT [...]' or 'SAFE'."""
    hits = []
    for d in results:
        level = d.get('result', 'l5_unlikely')
        if level in ('l1_confident', 'l2_very_likely', 'l3_likely'):
            hits.append(f"{d.get('detector_type','?')}({level})")
    return f"THREAT [{', '.join(hits)}]" if hits else "SAFE"


def ensure_out_path(input_path: str, output_dir: str = "lakera_results",
                    explicit_output: Optional[str] = None) -> Path:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if explicit_output:
        return Path(explicit_output)
    stem = Path(input_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return outdir / f"lakera_{stem}_{ts}.csv"


def read_last_index_for_dataset(out_path: Path, dataset_name: str) -> int:
    """Return next index for this dataset when resuming a merged CSV."""
    if not out_path.exists():
        return 0
    try:
        last = -1
        for df in pd.read_csv(out_path, chunksize=200000, usecols=["dataset", "index"]):
            m = df["dataset"] == dataset_name
            if m.any():
                last = max(last, int(df.loc[m, "index"].max()))
        return last + 1 if last >= 0 else 0
    except Exception:
        return 0


def write_rows(out_path: Path, rows: List[Dict], header_written: bool) -> bool:
    """Append rows to CSV. Returns whether header was written."""
    import csv
    if not rows:
        return header_written
    cols = ["dataset", "index", "prompt", "ground_truth", "prediction", "latency_ms", "detector_count"]
    write_header = not header_written and not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return header_written or write_header


# ---------------------------- Readers (chunked) ----------------------------

def detect_columns(df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]) -> Tuple[str, Optional[str]]:
    """Detect text/label columns on the first chunk; reuse thereafter."""
    # text col
    if text_col and text_col in df.columns:
        tc = text_col
    else:
        candidates = ['prompt', 'text', 'content', 'message', 'input', 'query', 'goal', 'instruction',
                      'conversations', 'chat', 'dialogue', 'messages', 'question', 'user_input', 'request', 'task']
        tc = None
        lowcols = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in lowcols:
                tc = lowcols[c]
                break
        if tc is None:
            # heuristic by length
            best, best_len = None, -1
            for c in df.columns:
                if df[c].dtype == "object":
                    s = df[c].dropna().astype(str)
                    if len(s) == 0:
                        continue
                    avglen = s.head(50).str.len().mean()
                    if avglen > best_len:
                        best, best_len = c, avglen
            if best is None:
                raise ValueError(f"Could not detect text column. Available: {list(df.columns)}")
            tc = best

    # label col
    lc = None
    if label_col and label_col in df.columns:
        lc = label_col
    else:
        lcands = ['type', 'label', 'labels', 'ground_truth', 'groundtruth', 'gt', 'class', 'category',
                  'classification', 'target', 'y', 'output', 'is_safe', 'safety', 'risk', 'harmful',
                  'toxicity', 'jailbreak', 'adversarial', 'behavior', 'behaviour', 'intent', 'malicious', 'benign']
        lowcols = {c.lower(): c for c in df.columns}
        for c in lcands:
            if c in lowcols:
                lc = lowcols[c]
                break
        if lc is None:
            # content-based quick scan
            for c in df.columns:
                if c == tc or df[c].dtype != "object":
                    continue
                vals = [str(v).lower() for v in df[c].dropna().head(50).tolist()]
                if any(any(t in v for t in ('safe', 'unsafe', 'jailbreak', 'benign', 'harmful', 'true', 'false', '0', '1')) for v in vals):
                    lc = c
                    break
    return tc, lc


def iter_csv_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    for df in pd.read_csv(path, chunksize=chunksize):
        yield df


def iter_jsonl_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(buf) >= chunksize:
                yield pd.DataFrame(buf)
                buf = []
    if buf:
        yield pd.DataFrame(buf)


def iter_parquet_chunks(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    # Best-effort: use pyarrow row groups if available, else load and slice
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        batch = []
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            df = table.to_pandas()
            batch.append(df)
            while sum(len(x) for x in batch) >= chunksize:
                acc = pd.concat(batch, ignore_index=True)
                yield acc.iloc[:chunksize].copy()
                batch = [acc.iloc[chunksize:].copy()]
        if batch:
            yield pd.concat(batch, ignore_index=True)
    except Exception:
        df = pd.read_parquet(path)
        for start in range(0, len(df), chunksize):
            yield df.iloc[start:start+chunksize].copy()


def chunk_iter_for(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return iter_csv_chunks(path, chunksize)
    elif ext == ".jsonl":
        return iter_jsonl_chunks(path, chunksize)
    elif ext == ".json":
        df = pd.read_json(path)
        return (df.iloc[i:i+chunksize].copy() for i in range(0, len(df), chunksize))
    elif ext == ".parquet":
        return iter_parquet_chunks(path, chunksize)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .jsonl, .json, .parquet")


# ---------------------------- HTTP Client ----------------------------

class LakeraClient:
    def __init__(self, api_key: str, project_id: Optional[str], timeout_s: float,
                 concurrency: int, retries: int, backoff_base: float, backoff_max: float):
        self.api_key = api_key
        self.project_id = project_id
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.url = "https://api.lakera.ai/v2/guard/results"
        self.sem = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, enable_cleanup_closed=True)
        # trust_env=True to honor HTTPS_PROXY/HTTP_PROXY/NO_PROXY if present
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=conn,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            trust_env=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def _once(self, prompt: str):
        assert self.session is not None
        body = {"messages": [{"role": "user", "content": prompt}]}
        if self.project_id:
            body["project_id"] = self.project_id
        t0 = time.time()
        async with self.session.post(self.url, json=body) as resp:
            status = resp.status
            if status == 200:
                data = await resp.json()
                latency = time.time() - t0
                return data.get("results", []), latency, None, status
            else:
                try:
                    text = await resp.text()
                except Exception:
                    text = f"HTTP {status}"
                return [], 0.0, text, status

    async def check(self, prompt: str) -> Dict:
        async with self.sem:
            await asyncio.sleep(random.uniform(0, 0.05))  # micro-jitter
            delay = self.backoff_base
            for _ in range(1, self.retries + 1):
                try:
                    results, latency, err_text, status = await self._once(prompt)
                    if status == 200:
                        return {"ok": True, "results": results, "latency": latency}
                    if status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(min(delay, self.backoff_max) + random.uniform(0, 0.25))
                        delay *= 2
                        continue
                    return {"ok": False, "error": f"{status}: {err_text or 'error'}"}
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(min(delay, self.backoff_max) + random.uniform(0, 0.25))
                    delay *= 2
            return {"ok": False, "error": "max retries exceeded"}


# ---------------------------- Main processing ----------------------------

async def process_file(path: str, api_key: str, project_id: Optional[str],
                       text_col: Optional[str], label_col: Optional[str],
                       max_rows: Optional[int],
                       chunksize: int, concurrency: int,
                       timeout: float, retries: int,
                       backoff_base: float, backoff_max: float,
                       out_path: Path, resume: bool,
                       show_progress: bool,
                       dataset_name: str) -> None:
    total_written = 0
    header_written = out_path.exists() and out_path.stat().st_size > 0
    start_index = read_last_index_for_dataset(out_path, dataset_name) if resume else 0

    chunk_iter = chunk_iter_for(path, chunksize)
    detected_text = None
    detected_label = None
    global_index = 0

    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(unit="rows", desc=f"Processing {Path(path).name}", leave=False)
    else:
        pbar = None

    async with LakeraClient(api_key, project_id, timeout, concurrency, retries, backoff_base, backoff_max) as client:
        for df in chunk_iter:
            if detected_text is None:
                detected_text, detected_label = detect_columns(df, text_col, label_col)

            df = df.reset_index(drop=True)
            indices = list(range(global_index, global_index + len(df)))
            global_index += len(df)

            rows_to_process = []
            for i_rel, i_abs in enumerate(indices):
                if i_abs < start_index:
                    continue
                if max_rows is not None and (i_abs - start_index) >= max_rows:
                    break
                gt = None
                if detected_label and detected_label in df.columns:
                    gt = parse_boolish_label(df.iloc[i_rel][detected_label])
                rows_to_process.append((i_abs, str(df.iloc[i_rel][detected_text]), gt))
            if not rows_to_process:
                if max_rows is not None and (global_index - start_index) >= max_rows:
                    break
                continue

            tasks = [asyncio.create_task(client.check(prompt)) for (_, prompt, _) in rows_to_process]
            results = await asyncio.gather(*tasks)

            out_rows = []
            for (i_abs, prompt, gt), res in zip(rows_to_process, results):
                if res.get("ok"):
                    pred = analyze_prediction(res["results"])
                    out_rows.append({
                        "dataset": dataset_name,
                        "index": i_abs,
                        "prompt": prompt,
                        "ground_truth": "THREAT" if gt is True else "SAFE" if gt is False else "N/A",
                        "prediction": pred,
                        "latency_ms": f"{res['latency']*1000:.2f}",
