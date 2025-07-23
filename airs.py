#!/usr/bin/env python3
"""
airs_batch_scan.py – Bulk‑scan prompts with Palo Alto Networks
AI Runtime Security (AIRS) via the pan‑aisecurity Python SDK.

Enhanced version with:
• Support for parquet, CSV, and JSONL files
• Results output to palo_alto_results directory
• Confusion matrix and performance metrics
• Latency tracking for each prompt
• Comprehensive performance analysis
• Smart auto-detection for jailbreak and security datasets
• Enhanced ground truth inference
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import dotenv
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_recall_fscore_support
)
import aisecurity
from tabulate import tabulate

from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
from aisecurity.generated_openapi_client import (
    AsyncScanObject,
    ScanRequest,
    ScanRequestContentsInner,
    ScanIdResult,
)
from aisecurity.scan.asyncio.scanner import Scanner

# --------------------------------------------------------------------------- #
#                               Constants                                     #
# --------------------------------------------------------------------------- #

# Polling configuration - adjusted for rate limits
DEFAULT_POLL_ATTEMPTS = 10  # Reduced from 20
POLL_INTERVAL_SECONDS = 3   # Increased from 2 to be safer with rate limits

# Rate limiting constants
MAX_SCAN_IDS_PER_QUERY = 5  # Palo Alto's limit
RATE_LIMIT_DELAY = 1.2      # Seconds between queries (50 requests/minute to be safe)

# Display configuration
TEXT_TRUNCATE_LENGTH = 80
DISPLAY_WIDTH = 120
DIVIDER = "=" * DISPLAY_WIDTH
SUBDIV = "-" * DISPLAY_WIDTH

# Batch configuration
DEFAULT_BATCH_SIZE = 1000

# Results directory
RESULTS_DIR = "palo_alto_results"

# Violation type mappings
PROMPT_VIOLATION_FIELDS = [
    "agent",
    "dlp",
    "injection",
    "toxic_content",
    "url_cats",
]
RESPONSE_VIOLATION_FIELDS = [
    "dlp",
    "toxic_content",
    "url_cats",
    "db_security",
    "ungrounded",
]

# Display names for violations
VIOLATION_DISPLAY_NAMES = {
    "toxic_content": "toxic",
    "url_cats": "url",
    "db_security": "db_sec",
}

# --------------------------------------------------------------------------- #
#                               Logging Setup                                 #
# --------------------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(format=LOG_FORMAT, stream=sys.stdout)
log = logging.getLogger("airs-batch-scan")


def configure_logging(level_str: Optional[str], debug_flag: bool) -> None:
    """Configure logging levels for the application and dependencies."""
    if debug_flag:
        level = logging.DEBUG
    else:
        level = getattr(logging, level_str.upper() if level_str else "INFO", logging.INFO)
    
    logging.getLogger().setLevel(level)
    logging.getLogger("aisecurity").setLevel(level)
    logging.getLogger("aiohttp.client").setLevel(level)


# --------------------------------------------------------------------------- #
#                             Utility Functions                               #
# --------------------------------------------------------------------------- #


def batched(iterable, n: int):
    """
    Batch an iterable into chunks of size n.
    For Python 3.12+, uses the built-in itertools.batched.
    For earlier versions, provides a simple implementation.
    """
    import itertools

    # Try to use the built-in if available
    if hasattr(itertools, "batched"):
        yield from itertools.batched(iterable, n)
        return

    # Simple implementation for older Python versions
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, n))
        if not batch:
            return
        yield batch


def setup_results_directory() -> pathlib.Path:
    """Create and return the results directory path."""
    results_path = pathlib.Path(RESULTS_DIR)
    results_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (results_path / "detailed").mkdir(exist_ok=True)
    (results_path / "metrics").mkdir(exist_ok=True)
    (results_path / "confusion_matrix").mkdir(exist_ok=True)
    
    return results_path


def auto_detect_column_mappings(data: List[Dict], filename: str = "") -> Dict[str, Optional[str]]:
    """
    Auto-detect column mappings for prompt, response, and ground truth fields.
    Returns a mapping of standard field names to detected column names.
    """
    if not data or not isinstance(data[0], dict):
        return {"prompt": None, "response": None, "expected_label": None, "inferred_ground_truth": None}
    
    sample_row = data[0]
    columns = list(sample_row.keys())
    log.debug("Available columns: %s", columns)
    
    mappings = {
        "prompt": None,
        "response": None,
        "expected_label": None,
        "inferred_ground_truth": None  # For cases where we can infer all are malicious/benign
    }
    
    # Prompt field detection (case-insensitive)
    prompt_candidates = [
        "prompt", "text", "input", "question", "query", "message", 
        "content", "user_input", "prompt_text", "user_message",
        "instruction", "request", "user_query", "goal", "redteam_query",
        "jailbreak_query", "attack_prompt", "harmful_prompt"
    ]
    
    for col in columns:
        col_lower = col.lower()
        if col_lower in [c.lower() for c in prompt_candidates]:
            mappings["prompt"] = col
            log.debug("Auto-detected prompt field: %s", col)
            break
    
    # Response field detection
    response_candidates = [
        "response", "output", "answer", "reply", "result", 
        "assistant_response", "bot_response", "completion",
        "generated_text", "model_response", "ai_response", "target"
    ]
    
    for col in columns:
        col_lower = col.lower()
        if col_lower in [c.lower() for c in response_candidates]:
            mappings["response"] = col
            log.debug("Auto-detected response field: %s", col)
            break
    
    # Ground truth/label field detection
    label_candidates = [
        "expected_label", "ground_truth", "label", "true_label",
        "classification", "category", "class", "target", "truth",
        "gold_standard", "annotation", "tag", "verdict", "expected",
        "correct_label", "actual_label", "is_malicious", "is_safe",
        "policy", "behavior", "violation_type", "harm_category"
    ]
    
    for col in columns:
        col_lower = col.lower()
        if col_lower in [c.lower() for c in label_candidates]:
            mappings["expected_label"] = col
            log.debug("Auto-detected ground truth field: %s", col)
            break
    
    # Special handling for boolean/binary fields that might indicate malicious/benign
    if not mappings["expected_label"]:
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["malicious", "toxic", "harmful", "safe", "benign"]):
                # Check if it's a boolean field
                sample_values = [str(row.get(col, "")).lower() for row in data[:10] if row.get(col) is not None]
                boolean_values = {"true", "false", "1", "0", "yes", "no"}
                if any(val in boolean_values for val in sample_values):
                    mappings["expected_label"] = col
                    log.debug("Auto-detected boolean ground truth field: %s", col)
                    break
    
    # Filename-based inference for datasets where all samples have the same label
    filename_lower = filename.lower()
    if not mappings["expected_label"]:
        # Check if filename suggests all samples are malicious
        malicious_indicators = ["jailbreak", "redteam", "attack", "harmful", "toxic", "adversarial", "exploit"]
        benign_indicators = ["safe", "harmless", "clean", "normal", "legitimate"]
        
        if any(indicator in filename_lower for indicator in malicious_indicators):
            mappings["inferred_ground_truth"] = "malicious"
            log.info("Inferred from filename '%s': all samples are malicious", filename)
        elif any(indicator in filename_lower for indicator in benign_indicators):
            mappings["inferred_ground_truth"] = "benign"
            log.info("Inferred from filename '%s': all samples are benign", filename)
    
    # Content-based inference - check if all samples seem to have harmful content
    if not mappings["expected_label"] and not mappings["inferred_ground_truth"]:
        # Check for columns that contain violation/harm categories
        harm_indicating_columns = []
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ["policy", "category", "behavior", "violation"]):
                # Sample values from this column
                sample_values = [str(row.get(col, "")).lower() for row in data[:20] if row.get(col)]
                
                # Check if values indicate harmful content
                harmful_indicators = [
                    "harm", "abuse", "violence", "hate", "fraud", "illegal", "unethical",
                    "harassment", "discrimination", "malware", "economic harm", "physical harm",
                    "government decision", "bias", "privacy violation", "child abuse"
                ]
                
                harmful_count = sum(1 for val in sample_values 
                                  if any(indicator in val for indicator in harmful_indicators))
                
                if len(sample_values) > 0 and harmful_count > len(sample_values) * 0.8:  # 80% of samples seem harmful
                    harm_indicating_columns.append((col, harmful_count, len(sample_values)))
        
        if harm_indicating_columns:
            best_col = max(harm_indicating_columns, key=lambda x: x[1])
            log.info("Detected harm-indicating column '%s': %d/%d samples contain harmful indicators", 
                    best_col[0], best_col[1], best_col[2])
            mappings["inferred_ground_truth"] = "malicious"
    
    return mappings


def normalize_ground_truth_value(value: Any, column_name: str = "") -> Optional[str]:
    """
    Normalize ground truth values to 'benign' or 'malicious'.
    Enhanced to handle policy violations, harm categories, and jailbreak indicators.
    """
    if value is None or value == "":
        return None
    
    str_value = str(value).lower().strip()
    
    # Direct mapping
    if str_value in ["benign", "malicious"]:
        return str_value
    
    # Jailbreak/attack indicators - map to malicious
    jailbreak_indicators = [
        "jailbreak", "attack", "adversarial", "harmful", "red_team", "redteam"
    ]
    
    if str_value in jailbreak_indicators:
        return "malicious"
    
    # Boolean-style mappings
    malicious_indicators = [
        "true", "1", "yes", "positive", "toxic", "harmful", "unsafe", 
        "bad", "violation", "threat", "malicious", "dangerous"
    ]
    benign_indicators = [
        "false", "0", "no", "negative", "clean", "safe", "good", 
        "benign", "harmless", "ok", "normal"
    ]
    
    if str_value in malicious_indicators:
        return "malicious"
    elif str_value in benign_indicators:
        return "benign"
    
    # Policy/harm category mappings - these typically indicate malicious content
    harm_categories = [
        "harassment", "discrimination", "malware", "hacking", "physical harm",
        "economic harm", "fraud", "deception", "disinformation", "sexual",
        "adult content", "privacy", "expert advice", "government decision",
        "unethical behavior", "hate speech", "political sensitivity",
        "illegal activity", "bias", "violence", "animal abuse",
        "unlicensed advice", "privacy violation", "health consultation",
        "child abuse", "defamation", "body shaming", "xenophobia",
        "ableism", "sexism", "racial discrimination", "ageism", "enslavement",
        "holocaust glorification", "sex discrimination"
    ]
    
    # Check if the value contains any harm category (partial matching)
    for harm_cat in harm_categories:
        if harm_cat in str_value:
            return "malicious"
    
    # Context-aware mapping based on column name
    col_lower = column_name.lower()
    if "malicious" in col_lower or "toxic" in col_lower or "harmful" in col_lower:
        # For columns like "is_malicious", True means malicious
        if str_value in ["true", "1", "yes"]:
            return "malicious"
        elif str_value in ["false", "0", "no"]:
            return "benign"
    elif "safe" in col_lower or "benign" in col_lower:
        # For columns like "is_safe", True means benign
        if str_value in ["true", "1", "yes"]:
            return "benign"
        elif str_value in ["false", "0", "no"]:
            return "malicious"
    elif "policy" in col_lower or "category" in col_lower or "behavior" in col_lower or "type" in col_lower:
        # Policy/category/behavior/type columns typically indicate violations = malicious
        # Exception: if the value is explicitly "benign" or similar, keep it benign
        if str_value in ["none", "null", "normal", "clean"]:
            return "benign"
        elif len(str_value) > 2:  # Any substantial category/type value = malicious
            return "malicious"
    
    # If we can't determine, log a warning but don't fail
    log.warning("Could not normalize ground truth value: '%s' (from column: %s) - treating as None", value, column_name)
    return None


def _normalise_yaml_json(data: Any) -> List[Dict[str, Optional[str]]]:
    """
    Normalise YAML/JSON structures to a list[dict(prompt, response, expected_label)].
    """
    if isinstance(data, dict):
        data = list(data.values())
    
    normalised = []
    for item in data:
        if isinstance(item, (list, tuple)):
            prompt = item[0] if len(item) > 0 else None
            response = item[1] if len(item) > 1 else None
            expected_label = item[2] if len(item) > 2 else None
            normalised.append({
                "prompt": prompt, 
                "response": response,
                "expected_label": expected_label
            })
        elif isinstance(item, dict):
            normalised.append({
                "prompt": (item.get("prompt") or item.get("input") or item.get("question")),
                "response": (item.get("response") or item.get("output") or item.get("answer")),
                "expected_label": (item.get("expected_label") or item.get("ground_truth") or item.get("label")),
            })
        else:  # bare string
            normalised.append({"prompt": str(item), "response": None, "expected_label": None})
    
    return normalised


def load_input_file(
    path: pathlib.Path, 
    prompt_field: Optional[str] = None, 
    ground_truth_field: Optional[str] = None
) -> List[Dict[str, Optional[str]]]:
    """
    Parse CSV, JSON, JSONL, Parquet, or YAML into a list of standardized dictionaries.
    Auto-detects field mappings unless explicitly specified.
    Enhanced to handle jailbreak datasets and harm categories.
    """
    log.info("Loading input file: %s", path)
    ext = path.suffix.lower()
    filename = path.name
    
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            rows = df.to_dict("records")
        elif ext == ".parquet":
            df = pd.read_parquet(path)
            rows = df.to_dict("records")
        elif ext == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            log.warning("Skipping invalid JSON on line %d: %s", line_num, e)
        elif ext in (".yml", ".yaml"):
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            rows = _normalise_yaml_json(data)
        elif ext == ".json":
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            rows = _normalise_yaml_json(data)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        log.error("Failed to parse file %s: %s", path, e)
        raise

    if not rows:
        log.warning("No data found in input file")
        return []

    # Auto-detect column mappings unless explicitly provided
    auto_mappings = auto_detect_column_mappings(rows, filename)
    
    # Use provided field names or fall back to auto-detected ones
    prompt_col = prompt_field or auto_mappings.get("prompt")
    response_col = auto_mappings.get("response")  # Always auto-detect response
    ground_truth_col = ground_truth_field or auto_mappings.get("expected_label")
    inferred_ground_truth = auto_mappings.get("inferred_ground_truth")
    
    log.info("Field mappings - Prompt: %s, Response: %s, Ground Truth: %s", 
             prompt_col, response_col, ground_truth_col)
    
    if inferred_ground_truth:
        log.info("Inferred ground truth: All samples labeled as '%s'", inferred_ground_truth)
    
    if not prompt_col:
        available_cols = list(rows[0].keys()) if rows else []
        raise ValueError(
            f"Could not auto-detect prompt field. Available columns: {available_cols}. "
            f"Please specify --prompt-field parameter."
        )
    
    # Normalize the data structure
    normalized_rows = []
    for row_idx, row in enumerate(rows):
        try:
            if isinstance(row, dict):
                # Extract prompt
                prompt_value = row.get(prompt_col)
                if prompt_value is None and len(row) == 1:
                    # Handle single-column data where the column contains the prompt
                    prompt_value = list(row.values())[0]
                
                # Extract response
                response_value = row.get(response_col) if response_col else None
                
                # Extract and normalize ground truth
                ground_truth_value = None
                if inferred_ground_truth:
                    # Use inferred ground truth for all samples
                    ground_truth_value = inferred_ground_truth
                elif ground_truth_col:
                    raw_gt_value = row.get(ground_truth_col)
                    ground_truth_value = normalize_ground_truth_value(raw_gt_value, ground_truth_col)
                
                normalized_row = {
                    "prompt": prompt_value,
                    "response": response_value,
                    "expected_label": ground_truth_value,
                    "original_row_index": row_idx,
                    "source_column_info": {
                        "prompt_col": prompt_col,
                        "response_col": response_col,
                        "ground_truth_col": ground_truth_col,
                        "inferred_ground_truth": inferred_ground_truth
                    }
                }
            else:
                # Handle list/tuple format - legacy support
                normalized_row = {
                    "prompt": str(row[0]) if len(row) > 0 else None,
                    "response": str(row[1]) if len(row) > 1 else None,
                    "expected_label": normalize_ground_truth_value(row[2]) if len(row) > 2 else None,
                    "original_row_index": row_idx,
                    "source_column_info": {}
                }
            normalized_rows.append(normalized_row)
        except Exception as e:
            log.warning("Error processing row %d: %s", row_idx, e)
            continue

    # Report statistics
    total_rows = len(normalized_rows)
    rows_with_prompts = sum(1 for row in normalized_rows if row.get("prompt"))
    rows_with_ground_truth = sum(1 for row in normalized_rows if row.get("expected_label"))
    
    # Analyze prompt lengths
    prompt_lengths = [len(str(row.get("prompt", ""))) for row in normalized_rows if row.get("prompt")]
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    max_prompt_length = max(prompt_lengths) if prompt_lengths else 0
    
    log.info("Data statistics:")
    log.info("  Total rows: %d", total_rows)
    log.info("  Rows with prompts: %d", rows_with_prompts)
    log.info("  Rows with ground truth: %d", rows_with_ground_truth)
    log.info("  Prompt length - Avg: %d chars, Max: %d chars", int(avg_prompt_length), max_prompt_length)
    
    if max_prompt_length > 1500:
        log.info("  ⚠️  Very long prompts detected - ensuring full content preservation")
    
    if rows_with_ground_truth > 0:
        ground_truth_values = [row["expected_label"] for row in normalized_rows 
                              if row.get("expected_label")]
        benign_count = sum(1 for val in ground_truth_values if val == "benign")
        malicious_count = sum(1 for val in ground_truth_values if val == "malicious")
        log.info("  Ground truth distribution - Benign: %d, Malicious: %d", 
                 benign_count, malicious_count)
        
        if inferred_ground_truth:
            log.info("  Ground truth source: Inferred from filename/content patterns")
        else:
            log.info("  Ground truth source: Column '%s'", ground_truth_col)

    return normalized_rows


def build_scan_objects(
    scan_contents: List[Dict[str, Optional[str]]],
    ai_profile: AiProfile,
) -> Tuple[List[AsyncScanObject], Dict[int, Dict[str, Optional[str]]]]:
    """Build AsyncScanObject list and content mapping."""
    async_objects = []
    content_map = {}
    
    for idx, sc in enumerate(scan_contents):
        req_id = idx + 1
        content_map[req_id] = sc
        
        # Ensure we have valid prompt text
        prompt_text = sc.get("prompt")
        response_text = sc.get("response")
        
        # Skip empty prompts
        if not prompt_text or str(prompt_text).strip() == "":
            log.warning("Skipping empty prompt at req_id %d", req_id)
            continue
        
        try:
            # Create content object - only include non-None fields
            content_inner = ScanRequestContentsInner()
            content_inner.prompt = str(prompt_text).strip()
            if response_text:
                content_inner.response = str(response_text).strip()
            
            # Build scan request with required fields
            scan_request = ScanRequest(
                ai_profile=ai_profile,
                contents=[content_inner],
            )
            
            async_objects.append(
                AsyncScanObject(
                    req_id=req_id,
                    scan_req=scan_request,
                )
            )
        except Exception as e:
            log.warning("Failed to create scan object for req_id %d: %s", req_id, e)
            continue
    
    log.debug("Constructed %d AsyncScanObject(s)", len(async_objects))
    return async_objects, content_map


async def run_batches(
    async_objects: List[AsyncScanObject],
    batch_size: int,
    endpoint_override: Optional[str] = None,
) -> Tuple[List[Any], Dict[int, float]]:
    """
    Submit batches concurrently with enhanced error handling and retry logic.
    """
    scanner = Scanner()
    if endpoint_override:
        scanner.api_endpoint = endpoint_override
    
    try:
        batches = list(batched(async_objects, batch_size))
        log.info("Submitting %d batch(es)…", len(batches))
        
        # Track latency for each request
        latency_map = {}
        successful_responses = []
        
        for i, batch in enumerate(batches):
            batch_num = i + 1
            log.info("Processing batch %d/%d (%d objects)", batch_num, len(batches), len(batch))
            
            # Record start time for latency calculation
            batch_start_time = time.time()
            for obj in batch:
                latency_map[obj.req_id] = batch_start_time

            # Try to submit this batch with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    log.debug("Batch %d attempt %d", batch_num, attempt + 1)
                    
                    # Submit batch
                    response = await scanner.async_scan(batch)
                    
                    # Calculate latency
                    batch_end_time = time.time()
                    batch_latency = batch_end_time - batch_start_time
                    per_request_latency = batch_latency / len(batch)
                    
                    # Update latency map
                    for obj in batch:
                        latency_map[obj.req_id] = per_request_latency
                    
                    successful_responses.append(response)
                    log.info("Batch %d submitted successfully", batch_num)
                    break
                    
                except Exception as e:
                    log.warning("Batch %d attempt %d failed: %s", batch_num, attempt + 1, str(e))
                    
                    # Check if it's a retryable error
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['internal server error', '500', 'timeout', 'connection']):
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            log.info("Retrying batch %d in %d seconds...", batch_num, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # For non-retryable errors or final attempt, log and continue
                    log.error("Batch %d failed permanently: %s", batch_num, str(e))
                    # Set default latency for failed batch
                    for obj in batch:
                        latency_map[obj.req_id] = 0.0
                    break
            
            # Rate limiting between batches
            if i < len(batches) - 1:
                await asyncio.sleep(0.5)  # Small delay between batches
        
        log.info("Completed batch submission: %d successful out of %d total", 
                len(successful_responses), len(batches))
        
        return successful_responses, latency_map
    
    except Exception as e:
        log.error("Critical error in batch processing: %s", str(e))
        raise
    finally:
        try:
            await scanner.close()
        except Exception as e:
            log.warning("Error closing scanner: %s", str(e))


def pretty_print_batch_results(batch_results: List[Any]) -> None:
    """Print batch submission results."""
    for idx, res in enumerate(batch_results, start=1):
        print(
            f"[Batch {idx}]  received={res.received!s:<5}  "
            f"scan_id={res.scan_id}  report_id={res.report_id}"
        )


def get_violations(detected_obj: Any, violation_fields: List[str]) -> List[str]:
    """Extract violations from a detection object."""
    violations = []
    for field in violation_fields:
        if getattr(detected_obj, field, False):
            display_name = VIOLATION_DISPLAY_NAMES.get(field, field)
            violations.append(display_name)
    return violations


def calculate_confusion_matrix(
    predicted_labels: List[str], 
    true_labels: List[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Calculate confusion matrix and simplified performance metrics."""
    # Handle cases where true_labels might contain None values
    filtered_pairs = [(p, t) for p, t in zip(predicted_labels, true_labels) if t is not None]
    
    if not filtered_pairs:
        log.warning("No ground truth labels found for confusion matrix calculation")
        return np.array([]), {}
    
    filtered_pred, filtered_true = zip(*filtered_pairs)
    
    try:
        # Create confusion matrix
        cm = confusion_matrix(filtered_true, filtered_pred, labels=["benign", "malicious"])
        
        # Calculate overall metrics (weighted averages)
        accuracy = accuracy_score(filtered_true, filtered_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_true, filtered_pred, labels=["benign", "malicious"], average="weighted", zero_division=0
        )
        
        # Also get macro averages for comparison
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred, labels=["benign", "malicious"], average="macro", zero_division=0
        )
        
        # Get total support counts
        _, _, _, support_per_class = precision_recall_fscore_support(
            filtered_true, filtered_pred, labels=["benign", "malicious"], average=None, zero_division=0
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,  # Weighted average precision
            "recall": recall,        # Weighted average recall  
            "f1_score": f1,         # Weighted average F1-score
            "precision_macro": precision_macro,  # Macro average (for reference)
            "recall_macro": recall_macro,        # Macro average (for reference)
            "f1_macro": f1_macro,               # Macro average (for reference)
            "support_benign": support_per_class[0] if len(support_per_class) > 0 else 0,
            "support_malicious": support_per_class[1] if len(support_per_class) > 1 else 0,
            "total_samples": len(filtered_pairs)
        }
        
        return cm, metrics
    except Exception as e:
        log.error("Error calculating metrics: %s", e)
        return np.array([]), {}


async def retrieve_and_display_results(
    scanner: Scanner,
    batch_results: List[Any],
    content_map: Dict[int, Dict[str, Optional[str]]],
    latency_map: Dict[int, float],
    results_dir: pathlib.Path,
    input_file_path: pathlib.Path,
) -> Dict[str, Any]:
    """
    Retrieve scan results, display them, and save detailed results with performance metrics.
    """
    # Collect all scan IDs
    scan_ids = [res.scan_id for res in batch_results]

    log.info("Retrieving scan results for %d scan(s)...", len(scan_ids))

    # Query for scan results with polling to wait for all results
    total_expected = len(content_map)
    scan_results = []

    for attempt in range(DEFAULT_POLL_ATTEMPTS):
        try:
            scan_results = await scanner.query_by_scan_ids(scan_ids=scan_ids)
            log.debug(
                "Polling attempt %d: Retrieved %d/%d scan results",
                attempt + 1,
                len(scan_results),
                total_expected,
            )

            if len(scan_results) >= total_expected:
                log.info("All %d results received", len(scan_results))
                break

            if attempt < DEFAULT_POLL_ATTEMPTS - 1:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            log.warning("Error retrieving results on attempt %d: %s", attempt + 1, e)
            if attempt < DEFAULT_POLL_ATTEMPTS - 1:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    if len(scan_results) < total_expected:
        log.warning(
            "Only received %d out of %d expected results after %d attempts",
            len(scan_results),
            total_expected,
            DEFAULT_POLL_ATTEMPTS,
        )

    # Process results
    detailed_results = []
    predicted_labels = []
    true_labels = []
    
    # Track violation types
    violation_types = {
        vtype: 0 for vtype in PROMPT_VIOLATION_FIELDS + RESPONSE_VIOLATION_FIELDS
    }

    log.debug("Processing %d scan results", len(scan_results))
    
    # Debug: Log the first few results to check for duplication
    if scan_results:
        log.debug("First scan result structure: %s", scan_results[0])
        if len(scan_results) > 1:
            log.debug("Second scan result structure: %s", scan_results[1])
    
    for idx, result in enumerate(scan_results):
        try:
            # Check if this is a ScanIdResult with a nested result
            if hasattr(result, "result") and result.result:
                scan_res = result.result
                req_id = result.req_id
                original_content = content_map.get(req_id, {})

                # Enhanced debugging for duplicate detection
                log.debug(
                    "Processing result %d: req_id=%s, scan_id=%s, category=%s, action=%s",
                    idx,
                    req_id,
                    getattr(result, 'scan_id', 'unknown'),
                    scan_res.category,
                    scan_res.action,
                )

                prompt_text = original_content.get("prompt", "N/A")
                response_text = original_content.get("response", "N/A") or "N/A"
                expected_label = original_content.get("expected_label")
                category = scan_res.category
                action = scan_res.action
                latency = latency_map.get(req_id, 0.0)

                # Log unique prompt text to verify we're processing different prompts
                if idx < 5:
                    log.debug("Processing prompt %d (first 200 chars): %s...", idx, prompt_text[:200])

                # Get violation details with safer attribute access
                prompt_detected = getattr(scan_res, 'prompt_detected', None)
                response_detected = getattr(scan_res, 'response_detected', None)

                # Initialize violation detection dictionaries
                prompt_detected_details = {}
                response_detected_details = {}
                
                # Count violations and build detection details
                if prompt_detected:
                    for key in PROMPT_VIOLATION_FIELDS:
                        detected = getattr(prompt_detected, key, False)
                        prompt_detected_details[key] = detected
                        if detected:
                            violation_types[key] += 1
                else:
                    # If no prompt_detected object, set all to False
                    for key in PROMPT_VIOLATION_FIELDS:
                        prompt_detected_details[key] = False

                if response_detected:
                    for key in RESPONSE_VIOLATION_FIELDS:
                        detected = getattr(response_detected, key, False)
                        response_detected_details[key] = detected
                        if detected:
                            violation_types[key] += 1
                else:
                    # If no response_detected object, set all to False
                    for key in RESPONSE_VIOLATION_FIELDS:
                        response_detected_details[key] = False

                # Build violation lists
                prompt_violations = get_violations(prompt_detected, PROMPT_VIOLATION_FIELDS) if prompt_detected else []
                response_violations = get_violations(response_detected, RESPONSE_VIOLATION_FIELDS) if response_detected else []

                # Store detailed result with unique identifier
                detailed_result = {
                    "req_id": req_id,
                    "scan_id": getattr(result, 'scan_id', f'unknown_{idx}'),  # Add scan_id for debugging
                    "prompt": prompt_text,
                    "response": response_text,
                    "predicted_label": category,
                    "expected_label": expected_label,
                    "action": action,
                    "latency_seconds": latency,
                    "prompt_violations": prompt_violations,
                    "response_violations": response_violations,
                    "prompt_detected_details": prompt_detected_details,
                    "response_detected_details": response_detected_details,
                    "timestamp": datetime.now().isoformat(),
                    "processing_index": idx  # Add processing order for debugging
                }
                detailed_results.append(detailed_result)
                
                # Collect labels for confusion matrix
                predicted_labels.append(category)
                true_labels.append(expected_label)
                
                # Debug log for first few results to verify uniqueness
                if idx < 3:
                    log.debug("Result %d details: category=%s, violations=%s, req_id=%s", 
                             idx, category, prompt_violations + response_violations, req_id)
                    
        except Exception as e:
            log.warning("Error processing result %d: %s", idx, e)
            continue

    # Check for duplicate results based on first 100 characters (for logging only)
    unique_scan_ids = set(r.get('scan_id', 'unknown') for r in detailed_results)
    unique_req_ids = set(r.get('req_id', 'unknown') for r in detailed_results)
    unique_prompts = set(r.get('prompt', '')[:100] for r in detailed_results)  # First 100 chars for uniqueness check only
    
    log.info("Results diversity check:")
    log.info("  Total results: %d", len(detailed_results))
    log.info("  Unique scan IDs: %d", len(unique_scan_ids))
    log.info("  Unique request IDs: %d", len(unique_req_ids))
    log.info("  Unique prompts (first 100 chars): %d", len(unique_prompts))
    
    if len(unique_req_ids) < len(detailed_results):
        log.warning("POTENTIAL ISSUE: Duplicate request IDs detected!")
    
    if len(unique_prompts) < min(5, len(detailed_results)):
        log.warning("POTENTIAL ISSUE: Very few unique prompts detected!")

    # Calculate confusion matrix and performance metrics
    confusion_mat, performance_metrics = calculate_confusion_matrix(predicted_labels, true_labels)
    
    # Add latency statistics
    latencies = [r["latency_seconds"] for r in detailed_results]
    latency_stats = {
        "mean_latency": np.mean(latencies) if latencies else 0,
        "median_latency": np.median(latencies) if latencies else 0,
        "min_latency": np.min(latencies) if latencies else 0,
        "max_latency": np.max(latencies) if latencies else 0,
        "std_latency": np.std(latencies) if latencies else 0,
        "total_processing_time": sum(latencies)
    }

    # Save all results in a single comprehensive CSV file
    try:
        input_filename = input_file_path.stem
        comprehensive_output_path = results_dir / f"{input_filename}_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Prepare comprehensive results
        comprehensive_data = []
        
        # Section 1: Scan Results Header
        comprehensive_data.extend([
            ["=== SCAN RESULTS ==="],
            ["Input File", input_file_path.name],
            ["Total Scans", len(scan_results)],
            ["Unique Request IDs", len(unique_req_ids)],
            ["Unique Scan IDs", len(unique_scan_ids)],
            ["Scan Timestamp", datetime.now().isoformat()],
            [""],  # Empty row for separation
        ])
        
        # Section 2: Detailed Results
        comprehensive_data.extend([
            ["=== DETAILED SCAN RESULTS ==="],
            # Headers for detailed results
            ["req_id", "scan_id", "prompt", "response", "predicted_label", "expected_label", "action", 
             "latency_seconds", "prompt_violations", "response_violations"] + 
            [f"prompt_{field}" for field in PROMPT_VIOLATION_FIELDS] +
            [f"response_{field}" for field in RESPONSE_VIOLATION_FIELDS] + ["timestamp", "processing_index"]
        ])
        
        # Add detailed results data
        for result in detailed_results:
            row = [
                result["req_id"],
                result.get("scan_id", "unknown"),
                result["prompt"],  # Full prompt, no truncation
                result["response"] if result["response"] and result["response"] != "N/A" else "",  # Full response
                result["predicted_label"],
                result.get("expected_label", ""),
                result["action"],
                f"{result['latency_seconds']:.3f}",
                "; ".join(result["prompt_violations"]) if result["prompt_violations"] else "",
                "; ".join(result["response_violations"]) if result["response_violations"] else "",
            ]
            
            # Add prompt detection details
            for field in PROMPT_VIOLATION_FIELDS:
                row.append(result["prompt_detected_details"].get(field, False))
            
            # Add response detection details  
            for field in RESPONSE_VIOLATION_FIELDS:
                row.append(result["response_detected_details"].get(field, False))
            
            row.append(result["timestamp"])
            row.append(result.get("processing_index", ""))
            comprehensive_data.append(row)
        
        # Add separation
        comprehensive_data.extend([
            [""],
            [""],
        ])
        
        # Section 3: Performance Metrics
        if performance_metrics and performance_metrics.get("total_samples", 0) > 0:
            comprehensive_data.extend([
                ["=== PERFORMANCE METRICS ==="],
                ["Metric", "Value"],
                ["Accuracy", f"{performance_metrics['accuracy']:.3f}"],
                ["Precision", f"{performance_metrics['precision']:.3f}"],
                ["Recall", f"{performance_metrics['recall']:.3f}"],
                ["F1-Score", f"{performance_metrics['f1_score']:.3f}"],
                ["Total Samples with Ground Truth", int(performance_metrics['total_samples'])],
                ["Support (Benign)", int(performance_metrics['support_benign'])],
                ["Support (Malicious)", int(performance_metrics['support_malicious'])],
                [""],
            ])
            
            # Confusion Matrix
            if confusion_mat.size > 0:
                comprehensive_data.extend([
                    ["=== CONFUSION MATRIX ==="],
                    ["", "Predicted Benign", "Predicted Malicious"],
                    ["True Benign", int(confusion_mat[0, 0]), int(confusion_mat[0, 1])],
                    ["True Malicious", int(confusion_mat[1, 0]), int(confusion_mat[1, 1])],
                    [""],
                ])
        else:
            comprehensive_data.extend([
                ["=== PERFORMANCE METRICS ==="],
                ["No ground truth labels available for performance metrics"],
                [""],
            ])
        
        # Section 4: Latency Statistics
        comprehensive_data.extend([
            ["=== LATENCY STATISTICS ==="],
            ["Metric", "Value"],
            ["Mean Latency (seconds)", f"{latency_stats['mean_latency']:.3f}"],
            ["Median Latency (seconds)", f"{latency_stats['median_latency']:.3f}"],
            ["Min Latency (seconds)", f"{latency_stats['min_latency']:.3f}"],
            ["Max Latency (seconds)", f"{latency_stats['max_latency']:.3f}"],
            ["Std Deviation (seconds)", f"{latency_stats['std_latency']:.3f}"],
            ["Total Processing Time (seconds)", f"{latency_stats['total_processing_time']:.3f}"],
            [""],
        ])
        
        # Section 5: Violation Breakdown
        comprehensive_data.extend([
            ["=== VIOLATION TYPES BREAKDOWN ==="],
            ["Violation Type", "Count"],
        ])
        
        for vtype, count in violation_types.items():
            if count > 0:
                comprehensive_data.append([vtype.replace("_", " ").title(), int(count)])
        
        if not any(count > 0 for count in violation_types.values()):
            comprehensive_data.append(["No specific violations detected", ""])
        
        comprehensive_data.extend([
            [""],
        ])
        
        # Section 6: Summary
        malicious_count = len([r for r in detailed_results if r["predicted_label"] == "malicious"])
        benign_count = len([r for r in detailed_results if r["predicted_label"] == "benign"])
        
        comprehensive_data.extend([
            ["=== SUMMARY ==="],
            ["Metric", "Value"],
            ["Total Scans", len(detailed_results)],
            ["Malicious Detected", malicious_count],
            ["Benign Detected", benign_count],
            ["Average Latency (seconds)", f"{latency_stats['mean_latency']:.3f}"],
            ["Total Processing Time (seconds)", f"{latency_stats['total_processing_time']:.3f}"],
        ])
        
        if performance_metrics and performance_metrics.get("total_samples", 0) > 0:
            comprehensive_data.extend([
                ["Overall Accuracy", f"{performance_metrics['accuracy']:.3f}"],
                ["Samples with Ground Truth", int(performance_metrics['total_samples'])],
            ])
        else:
            comprehensive_data.append(["Ground Truth Available", "No"])
        
        # Save comprehensive CSV
        with comprehensive_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(comprehensive_data)
        
        log.info("Comprehensive results saved to %s", comprehensive_output_path)
        
    except Exception as e:
        log.error("Failed to save comprehensive results: %s", e)

    # Display results
    display_scan_results(detailed_results, performance_metrics, latency_stats, violation_types, confusion_mat)

    return {
        "detailed_results": detailed_results,
        "performance_metrics": performance_metrics,
        "latency_statistics": latency_stats,
        "violation_types": violation_types,
        "confusion_matrix": confusion_mat
    }


def display_scan_results(
    detailed_results: List[Dict[str, Any]],
    performance_metrics: Dict[str, Any],
    latency_stats: Dict[str, float],
    violation_types: Dict[str, int],
    confusion_matrix: np.ndarray,
) -> None:
    """Display comprehensive scan results including performance metrics."""
    print("\n" + DIVIDER)
    print("AI RUNTIME SECURITY SCAN RESULTS & PERFORMANCE ANALYSIS".center(DISPLAY_WIDTH))
    print(DIVIDER)

    # Separate results by category
    malicious_results = [r for r in detailed_results if r["predicted_label"] == "malicious"]
    benign_results = [r for r in detailed_results if r["predicted_label"] == "benign"]

    # Display sample results
    if malicious_results:
        print(f"\n🚨 MALICIOUS PROMPTS ({len(malicious_results)} detected)")
        print(SUBDIV)
        headers = ["Prompt", "Violations", "Action", "Latency (s)"]
        table_data = []
        for r in malicious_results[:10]:
            # Only truncate for display purposes, not in CSV output
            prompt_display = (
                r["prompt"][:TEXT_TRUNCATE_LENGTH] + "..." 
                if len(r["prompt"]) > TEXT_TRUNCATE_LENGTH 
                else r["prompt"]
            )
            violations = ", ".join(r["prompt_violations"]) if r["prompt_violations"] else "policy violation"
            table_data.append([prompt_display, violations, r["action"], f"{r['latency_seconds']:.3f}"])
        
        print(tabulate(
            table_data,
            headers=headers,
            tablefmt="rounded_outline",
            maxcolwidths=[70, 20, 10, 10],
            numalign="left",
        ))
        if len(malicious_results) > 10:
            print(f"... and {len(malicious_results) - 10} more malicious prompts")

    if benign_results:
        print(f"\n✅ BENIGN PROMPTS ({len(benign_results)} detected)")
        print(SUBDIV)
        headers = ["Prompt", "Action", "Latency (s)"]
        table_data = []
        for r in benign_results[:5]:
            # Only truncate for display purposes, not in CSV output
            prompt_display = (
                r["prompt"][:TEXT_TRUNCATE_LENGTH] + "..." 
                if len(r["prompt"]) > TEXT_TRUNCATE_LENGTH 
                else r["prompt"]
            )
            table_data.append([prompt_display, r["action"], f"{r['latency_seconds']:.3f}"])
        
        print(tabulate(
            table_data,
            headers=headers,
            tablefmt="rounded_outline",
            maxcolwidths=[90, 10, 10],
            numalign="left",
        ))
        if len(benign_results) > 5:
            print(f"... and {len(benign_results) - 5} more benign prompts")

    # Performance Metrics (only show if we have ground truth)
    if performance_metrics and performance_metrics.get("total_samples", 0) > 0:
        print("\n📊 PERFORMANCE METRICS")
        print(SUBDIV)
        perf_data = [
            ["Accuracy", f"{performance_metrics['accuracy']:.3f}"],
            ["Precision", f"{performance_metrics['precision']:.3f}"],
            ["Recall", f"{performance_metrics['recall']:.3f}"],
            ["F1-Score", f"{performance_metrics['f1_score']:.3f}"],
        ]
        print(tabulate(
            perf_data,
            headers=["Metric", "Value"],
            tablefmt="rounded_outline",
            numalign="left",
        ))

        # Confusion Matrix
        if confusion_matrix.size > 0:
            print("\n🎯 CONFUSION MATRIX")
            print(SUBDIV)
            cm_data = [
                ["", "Predicted Benign", "Predicted Malicious"],
                ["True Benign", str(confusion_matrix[0, 0]), str(confusion_matrix[0, 1])],
                ["True Malicious", str(confusion_matrix[1, 0]), str(confusion_matrix[1, 1])],
            ]
            print(tabulate(
                cm_data,
                tablefmt="rounded_outline",
                numalign="center",
            ))
    else:
        print("\n📊 PERFORMANCE METRICS")
        print(SUBDIV)
        print("⚠️  No ground truth labels found - performance metrics unavailable")
        print("   Add ground truth labels to your data or use --ground-truth-field parameter")

    # Latency Statistics
    print("\n⏱️ LATENCY STATISTICS")
    print(SUBDIV)
    latency_data = [
        ["Mean Latency", f"{latency_stats['mean_latency']:.3f} seconds"],
        ["Median Latency", f"{latency_stats['median_latency']:.3f} seconds"],
        ["Min Latency", f"{latency_stats['min_latency']:.3f} seconds"],
        ["Max Latency", f"{latency_stats['max_latency']:.3f} seconds"],
        ["Std Deviation", f"{latency_stats['std_latency']:.3f} seconds"],
        ["Total Processing Time", f"{latency_stats['total_processing_time']:.3f} seconds"],
    ]
    print(tabulate(
        latency_data,
        headers=["Metric", "Value"],
        tablefmt="rounded_outline",
        numalign="left",
    ))

    # Violation Types Summary
    print("\n📋 VIOLATION TYPES BREAKDOWN")
    print(SUBDIV)
    violation_data = []
    for vtype, count in violation_types.items():
        if count > 0:
            violation_data.append([vtype.replace("_", " ").title(), count])

    if violation_data:
        print(tabulate(
            violation_data,
            headers=["Violation Type", "Count"],
            tablefmt="rounded_outline",
            numalign="left",
        ))
    else:
        print("No specific violations detected")

    # Final Summary
    print("\n📈 SUMMARY")
    print(SUBDIV)
    summary_data = [
        ["Total Scans", len(detailed_results)],
        ["Malicious Detected", len(malicious_results)],
        ["Benign Detected", len(benign_results)],
        ["Average Latency", f"{latency_stats['mean_latency']:.3f}s"],
        ["Total Processing Time", f"{latency_stats['total_processing_time']:.3f}s"],
    ]
    
    if performance_metrics and performance_metrics.get("total_samples", 0) > 0:
        summary_data.extend([
            ["Overall Accuracy", f"{performance_metrics['accuracy']:.3f}"],
            ["Samples with Ground Truth", performance_metrics["total_samples"]],
        ])
    else:
        summary_data.append(["Ground Truth Available", "No"])
    
    print(tabulate(
        summary_data,
        headers=["Metric", "Value"],
        tablefmt="rounded_outline",
        numalign="left",
    ))

    print("\n" + DIVIDER)


# --------------------------------------------------------------------------- #
#                                 Main Entry                                  #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Main entry point for the AI Runtime Security batch scanner."""
    dotenv.load_dotenv()  # safe even if .env is absent

    parser = argparse.ArgumentParser(
        description="Bulk scan prompts with AIRS (pan-aisecurity SDK). Enhanced with performance metrics and multi-format support."
    )
    parser.add_argument(
        "--file", 
        required=True, 
        type=pathlib.Path, 
        help="Input file (CSV, JSON, JSONL, Parquet, or YAML)"
    )
    parser.add_argument(
        "--prompt-field", 
        help="Column name for prompt text (auto-detected if not specified)"
    )
    parser.add_argument(
        "--ground-truth-field", 
        help="Column name for ground truth labels (auto-detected if not specified)"
    )
    parser.add_argument(
        "--profile-name", 
        help="AI Profile name (overrides env)"
    )
    parser.add_argument(
        "--profile-id", 
        help="AI Profile ID (overrides env)"
    )
    parser.add_argument(
        "--endpoint", 
        help="Custom API endpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of items per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shortcut for --log-level DEBUG (overrides)",
    )
    parser.add_argument(
        "--force-individual",
        action="store_true",
        help="Force individual requests instead of batching (for debugging)"
    )
    args = parser.parse_args()

    configure_logging(args.log_level, args.debug)

    try:
        _run(args)
    except KeyboardInterrupt:
        log.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as exc:
        log.fatal("Execution aborted due to an error: %s", exc, exc_info=True)
        sys.exit(1)


def _run(args: argparse.Namespace) -> None:
    """Run the batch scanner with the provided arguments."""
    # Validate arguments
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    
    if not args.file.exists():
        raise FileNotFoundError(f"Input file not found: {args.file}")

    # Setup results directory
    results_dir = setup_results_directory()
    log.info("Results will be saved to: %s", results_dir.absolute())

    # Validate API key
    api_key = os.getenv("PANW_AI_SEC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key missing – set PANW_AI_SEC_API_KEY in env or .env file."
        )

    if not api_key.strip():
        raise RuntimeError("API key is empty")

    # Determine the correct endpoint
    endpoint = args.endpoint or os.getenv("PANW_AI_SEC_API_ENDPOINT")
    if not endpoint:
        # Default to US endpoint
        endpoint = "https://service.api.aisecurity.paloaltonetworks.com"
        log.info("Using default US endpoint: %s", endpoint)
    else:
        log.info("Using specified endpoint: %s", endpoint)

    # Initialize SDK with proper error handling and validation
    try:
        log.info("Initializing Palo Alto Networks AI Security SDK...")
        aisecurity.init(
            api_key=api_key.strip(),
            api_endpoint=endpoint,
        )
        log.info("SDK initialized successfully")
        
        # Test basic connectivity (optional)
        log.info("Testing API connectivity...")
        
    except Exception as e:
        log.error("Failed to initialize SDK: %s", e)
        log.error("Troubleshooting tips:")
        log.error("1. Verify API key is correct: PANW_AI_SEC_API_KEY")
        log.error("2. Check network connectivity to: %s", endpoint)
        log.error("3. Try different endpoint (US/EU): --endpoint https://service-de.api.aisecurity.paloaltonetworks.com")
        log.error("4. Verify deployment profile is active in Strata Cloud Manager")
        raise RuntimeError(f"SDK initialization failed: {e}")

    # Validate profile configuration
    profile_name = args.profile_name or os.getenv("PANW_AI_PROFILE_NAME")
    profile_id = args.profile_id or os.getenv("PANW_AI_PROFILE_ID")
    if not (profile_name or profile_id):
        raise RuntimeError(
            "Provide --profile-name or --profile-id (or matching env var)."
        )

    # Create AI profile with validation
    try:
        if profile_name:
            ai_profile = AiProfile(profile_name=profile_name.strip())
            log.info("Using AI profile name: %s", profile_name)
        else:
            ai_profile = AiProfile(profile_id=profile_id.strip())
            log.info("Using AI profile ID: %s", profile_id)
    except Exception as e:
        log.error("Failed to create AI profile: %s", e)
        log.error("Troubleshooting: Verify profile exists in Strata Cloud Manager")
        raise RuntimeError(f"Invalid AI profile configuration: {e}")

    # Load input file with enhanced format support
    try:
        scan_contents = load_input_file(
            args.file, 
            prompt_field=args.prompt_field, 
            ground_truth_field=args.ground_truth_field
        )
    except Exception as e:
        log.error("Failed to load input file: %s", e)
        raise

    if not scan_contents:
        log.warning("Input file contained zero prompts – nothing to do.")
        return

    # Filter out invalid entries
    valid_contents = []
    for idx, content in enumerate(scan_contents):
        if not content.get("prompt") or not str(content["prompt"]).strip():
            log.warning("Skipping empty prompt at index %d", idx)
            continue
        valid_contents.append(content)

    if not valid_contents:
        log.warning("No valid prompts found after filtering – nothing to do.")
        return

    log.info("Loaded %d valid prompts for scanning", len(valid_contents))

    # Check for ground truth labels
    has_ground_truth = any(item.get("expected_label") for item in valid_contents)
    if has_ground_truth:
        log.info("Ground truth labels detected - performance metrics will be calculated")
    else:
        log.info("No ground truth labels found - only basic statistics will be provided")

    # Build scan objects with error handling
    try:
        async_objects, content_map = build_scan_objects(valid_contents, ai_profile)
    except Exception as e:
        log.error("Failed to build scan objects: %s", e)
        raise RuntimeError(f"Request formatting error: {e}")

    if not async_objects:
        log.warning("No valid scan objects created – check your input data format.")
        return

    # Use smaller batch sizes to avoid API issues
    effective_batch_size = min(args.batch_size, 50)  # Limit to 50 to avoid issues
    if effective_batch_size != args.batch_size:
        log.info("Reducing batch size from %d to %d for stability", args.batch_size, effective_batch_size)

    # Force individual processing if requested for debugging
    if args.force_individual:
        effective_batch_size = 1
        log.info("DEBUGGING MODE: Processing requests individually")

    # Run batches with latency tracking
    start_time = time.time()
    try:
        log.info("Starting batch processing with batch size: %d", effective_batch_size)
        batch_results, latency_map = asyncio.run(
            run_batches(
                async_objects,
                batch_size=effective_batch_size,
                endpoint_override=endpoint,
            )
        )
    except Exception as e:
        log.error("Batch processing failed: %s", e)
        log.error("Troubleshooting suggestions:")
        log.error("1. Try smaller batch size: --batch-size 10")
        log.error("2. Try individual processing: --force-individual")
        log.error("3. Check API rate limits and quotas")
        log.error("4. Verify deployment profile permissions")
        raise RuntimeError(f"Scan request failed: {e}")
        
    total_batch_time = time.time() - start_time
    log.info("Total batch processing time: %.3f seconds", total_batch_time)
    
    if not batch_results:
        log.error("No successful batch results received")
        log.error("This indicates a fundamental API connectivity or configuration issue")
        return
    
    pretty_print_batch_results(batch_results)

    # Always retrieve and display detailed results with performance analysis
    scanner = Scanner()
    if endpoint:
        scanner.api_endpoint = endpoint

    try:
        detailed_results = asyncio.run(
            retrieve_and_display_results(
                scanner, batch_results, content_map, latency_map, results_dir, args.file
            )
        )

        # Save summary report
        input_filename = args.file.stem
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        summary_report = {
            "scan_summary": {
                "input_filename": args.file.name,
                "total_prompts": len(valid_contents),
                "total_batches": len(batch_results),
                "batch_size_used": effective_batch_size,
                "total_processing_time": total_batch_time,
                "has_ground_truth": has_ground_truth,
                "profile_used": profile_name or profile_id,
                "api_endpoint": endpoint,
                "scan_timestamp": datetime.now().isoformat(),
                "input_file": str(args.file),
            },
            "results_summary": {
                "malicious_count": len([r for r in detailed_results["detailed_results"] if r["predicted_label"] == "malicious"]),
                "benign_count": len([r for r in detailed_results["detailed_results"] if r["predicted_label"] == "benign"]),
                "average_latency": float(detailed_results["latency_statistics"]["mean_latency"]),
                "total_violations": int(sum(detailed_results["violation_types"].values())),
            }
        }

        # Add performance metrics if available (with numpy conversion)
        if detailed_results["performance_metrics"]:
            summary_report["performance_metrics"] = convert_numpy_types(detailed_results["performance_metrics"])

        try:
            summary_path = results_dir / f"{input_filename}_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False)
            
            log.info("Summary report saved to: %s", summary_path)
            log.info("All results saved to directory: %s", results_dir.absolute())
        except Exception as e:
            log.error("Failed to save summary report: %s", e)

    except Exception as e:
        log.error("Failed to retrieve detailed results: %s", e)
        log.info("Basic batch submission completed successfully despite retrieval error")
    finally:
        try:
            asyncio.run(scanner.close())
        except Exception as e:
            log.warning("Error closing scanner: %s", e)


if __name__ == "__main__":
    main()
