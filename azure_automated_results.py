#!/usr/bin/env python3
"""
Azure Prompt Shield Batch Processor - Python 3.6+ Compatible
Fixed for older Python versions without asyncio.run()
"""

import asyncio
import aiohttp
import pandas as pd
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
from pathlib import Path
import csv
from dataclasses import dataclass
from enum import Enum
import argparse
import sys

# Python 3.6 compatibility for asyncio.run()
def run_async_main(coro):
    """Compatibility function for asyncio.run() in Python < 3.7"""
    try:
        # Try Python 3.7+ method
        return asyncio.run(coro)
    except AttributeError:
        # Fallback for Python 3.6
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Performance metrics will be limited.")
    SKLEARN_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    print("Warning: pyarrow not available. Parquet support disabled.")
    PYARROW_AVAILABLE = False

# Configure logging
azure_results_dir = Path("azure_results")
azure_results_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(azure_results_dir / 'azure_prompt_shield_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PromptResult:
    prompt: str
    decision: str
    latency_ms: float
    category: str
    original_type: str
    confidence_score: float = 0.0
    error_message: str = ""
    timestamp: str = ""

class AzurePromptShieldBatchProcessor:
    def __init__(self, 
                 endpoint_url: str,
                 api_key: str,
                 max_concurrent_requests: int = 10,
                 rate_limit_per_minute: int = 60,
                 timeout_seconds: int = 30,
                 output_directory: str = "azure_results",
                 prompt_column: str = None,
                 ground_truth_column: str = None):
        """Initialize the Azure Prompt Shield Batch Processor"""
        self.endpoint_url = endpoint_url.rstrip('/')
        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout_seconds = timeout_seconds
        
        # Column configuration
        self.prompt_column = prompt_column
        self.ground_truth_column = ground_truth_column
        
        # Create output directory
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        logger.info(f"Output directory set to: {self.output_directory.absolute()}")
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_times = []
        
        # Results storage
        self.results: List[PromptResult] = []

    def _detect_prompt_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the prompt column from common variations"""
        common_prompt_columns = [
            'prompt', 'text', 'input', 'question', 'query', 'content', 
            'message', 'user_input', 'prompt_text', 'input_text'
        ]
        
        available_columns = [col.lower() for col in df.columns]
        
        for col_variant in common_prompt_columns:
            if col_variant in available_columns:
                original_col = next(col for col in df.columns if col.lower() == col_variant)
                logger.info(f"Auto-detected prompt column: '{original_col}'")
                return original_col
        
        raise ValueError(f"Could not auto-detect prompt column. Available columns: {list(df.columns)}. "
                        f"Please specify --prompt-column parameter.")

    def _detect_ground_truth_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the ground truth column from common variations"""
        common_gt_columns = [
            'type', 'category', 'label', 'class', 'ground_truth', 'gt', 
            'classification', 'target', 'y', 'true_label', 'actual'
        ]
        
        available_columns = [col.lower() for col in df.columns]
        
        for col_variant in common_gt_columns:
            if col_variant in available_columns:
                original_col = next(col for col in df.columns if col.lower() == col_variant)
                logger.info(f"Auto-detected ground truth column: '{original_col}'")
                return original_col
        
        logger.warning("Could not auto-detect ground truth column. Performance metrics will be limited.")
        return None

    def _validate_columns(self, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
        """Validate and determine the correct column names to use"""
        
        # Determine prompt column
        if self.prompt_column:
            if self.prompt_column not in df.columns:
                raise ValueError(f"Specified prompt column '{self.prompt_column}' not found. "
                               f"Available columns: {list(df.columns)}")
            prompt_col = self.prompt_column
            logger.info(f"Using specified prompt column: '{prompt_col}'")
        else:
            prompt_col = self._detect_prompt_column(df)
        
        # Determine ground truth column
        if self.ground_truth_column:
            if self.ground_truth_column not in df.columns:
                raise ValueError(f"Specified ground truth column '{self.ground_truth_column}' not found. "
                               f"Available columns: {list(df.columns)}")
            gt_col = self.ground_truth_column
            logger.info(f"Using specified ground truth column: '{gt_col}'")
        else:
            gt_col = self._detect_ground_truth_column(df)
        
        return prompt_col, gt_col

    def load_prompts_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load prompts from CSV, JSON, JSONL, or Parquet file"""
        file_path = Path(file_path)
        prompts = []
        
        try:
            if file_path.suffix.lower() in ['.csv', '.parquet']:
                # Load data based on file type
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', escapechar='\\')
                else:  # .parquet
                    if not PYARROW_AVAILABLE:
                        raise ValueError("Parquet support requires pyarrow. Install with: pip install pyarrow")
                    df = pd.read_parquet(file_path, engine='pyarrow')
                
                logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
                
                # Validate and get column names
                prompt_col, gt_col = self._validate_columns(df)
                
                # Process rows
                for idx, row in df.iterrows():
                    prompt_text = str(row[prompt_col]).strip()
                    
                    # Skip empty prompts or NaN values
                    if pd.isna(row[prompt_col]) or not prompt_text or prompt_text.lower() in ['nan', 'null', '']:
                        continue
                    
                    # Get ground truth if available
                    if gt_col and gt_col in row.index:
                        category = str(row[gt_col]).strip()
                    else:
                        category = 'unknown'
                        
                    prompt_data = {
                        'prompt': prompt_text,
                        'category': category
                    }
                    prompts.append(prompt_data)
                    
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            prompt_key = self.prompt_column or 'prompt'
                            gt_key = self.ground_truth_column or next((k for k in ['category', 'type', 'label'] if k in item), 'category')
                            
                            if prompt_key not in item:
                                raise ValueError(f"JSON items must contain '{prompt_key}' key. Available keys: {list(item.keys())}")
                            
                            prompt_data = {
                                'prompt': str(item[prompt_key]).strip(),
                                'category': str(item.get(gt_key, 'unknown')).strip()
                            }
                            prompts.append(prompt_data)
                    else:
                        raise ValueError("JSON file must contain an array of objects")
                        
            elif file_path.suffix.lower() == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                item = json.loads(line)
                                
                                prompt_key = self.prompt_column or 'prompt'
                                gt_key = self.ground_truth_column or next((k for k in ['category', 'type', 'label'] if k in item), 'category')
                                
                                if prompt_key not in item:
                                    logger.warning(f"Line {line_num}: Missing '{prompt_key}' key. Available keys: {list(item.keys())}")
                                    continue
                                
                                prompt_data = {
                                    'prompt': str(item[prompt_key]).strip(),
                                    'category': str(item.get(gt_key, 'unknown')).strip()
                                }
                                prompts.append(prompt_data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                                continue
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .csv, .json, .jsonl, .parquet")
                
            logger.info(f"Successfully loaded {len(prompts)} valid prompts from {file_path}")
            
            if prompts:
                sample_prompt = prompts[0]
                logger.info(f"Sample prompt: {sample_prompt['prompt'][:100]}...")
                logger.info(f"Sample category: {sample_prompt['category']}")
                
                categories = {}
                for p in prompts:
                    cat = p['category']
                    categories[cat] = categories.get(cat, 0) + 1
                logger.info(f"Category distribution: {categories}")
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts from {file_path}: {str(e)}")
            raise

    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we've hit the rate limit
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)

    async def _call_prompt_shield_api(self, session: aiohttp.ClientSession, prompt: str) -> Dict[str, Any]:
        """Make API call to Azure Prompt Shield endpoint"""
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.api_key,
            'User-Agent': 'Azure-PromptShield-BatchProcessor/1.0'
        }
        
        payload = {
            'userPrompt': prompt,
            'documents': []
        }
        
        start_time = time.time()
        
        try:
            # Try different API endpoints and versions
            api_urls = [
                f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2023-10-01",
                f"{self.endpoint_url}/contentsafety/text:analyze?api-version=2023-10-01",
                f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview"
            ]
            
            for api_url in api_urls:
                try:
                    # Adjust payload for different endpoints
                    if 'analyze' in api_url:
                        current_payload = {
                            'text': prompt,
                            'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                            'outputType': 'FourSeverityLevels'
                        }
                    else:
                        current_payload = payload
                    
                    async with session.post(
                        api_url,
                        headers=headers,
                        json=current_payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                    ) as response:
                        end_time = time.time()
                        latency_ms = (end_time - start_time) * 1000
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.debug(f"API success with endpoint: {api_url}")
                            return {
                                'success': True,
                                'data': result,
                                'latency_ms': latency_ms,
                                'status_code': response.status,
                                'endpoint_used': api_url
                            }
                        elif response.status != 404:
                            # Log non-404 errors
                            error_text = await response.text()
                            logger.warning(f"API endpoint {api_url} returned {response.status}: {error_text[:200]}")
                            
                except Exception as e:
                    logger.debug(f"Failed to call {api_url}: {e}")
                    continue
            
            # If all endpoints failed
            return {
                'success': False,
                'error': "All API endpoints returned 404 or failed. Check endpoint URL and ensure Content Safety service is accessible.",
                'latency_ms': (time.time() - start_time) * 1000,
                'status_code': 404
            }
                    
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Request timeout after {self.timeout_seconds} seconds",
                'latency_ms': (time.time() - start_time) * 1000,
                'status_code': 408
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Request failed: {str(e)}",
                'latency_ms': (time.time() - start_time) * 1000,
                'status_code': 500
            }

    async def _process_single_prompt(self, session: aiohttp.ClientSession, prompt_data: Dict[str, Any]) -> PromptResult:
        """Process a single prompt through the API"""
        async with self.semaphore:
            await self._rate_limit_check()
            
            prompt = prompt_data['prompt']
            category = prompt_data['category']
            
            api_result = await self._call_prompt_shield_api(session, prompt)
            
            if api_result['success']:
                shield_result = api_result['data']
                
                # Handle different API response formats
                if 'attackDetected' in shield_result:
                    # Shield Prompt API response
                    attack_detected = shield_result.get('attackDetected', False)
                    decision = "BLOCKED" if attack_detected else "ALLOWED"
                elif 'categoriesAnalysis' in shield_result:
                    # Content Safety API response
                    blocked = any(
                        cat.get('severity', 0) > 2 
                        for cat in shield_result.get('categoriesAnalysis', [])
                    )
                    decision = "BLOCKED" if blocked else "ALLOWED"
                else:
                    decision = "ALLOWED"  # Default if unclear
                
                confidence_score = 0.0
                if 'analyzeResults' in shield_result:
                    for result in shield_result['analyzeResults']:
                        if 'confidenceScore' in result:
                            confidence_score = max(confidence_score, result['confidenceScore'])
                
                return PromptResult(
                    prompt=prompt,
                    decision=decision,
                    latency_ms=api_result['latency_ms'],
                    category=category,
                    original_type=category,
                    confidence_score=confidence_score,
                    error_message="",
                    timestamp=datetime.now().isoformat()
                )
            else:
                return PromptResult(
                    prompt=prompt,
                    decision="ERROR",
                    latency_ms=api_result['latency_ms'],
                    category=category,
                    original_type=category,
                    confidence_score=0.0,
                    error_message=api_result['error'],
                    timestamp=datetime.now().isoformat()
                )

    async def process_batch(self, prompts: List[Dict[str, Any]]) -> List[PromptResult]:
        """Process a batch of prompts asynchronously"""
        logger.info(f"Starting batch processing of {len(prompts)} prompts")
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self._process_single_prompt(session, prompt_data) for prompt_data in prompts]
            
            batch_size = min(self.max_concurrent_requests, len(tasks))
            results = []
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        error_result = PromptResult(
                            prompt="ERROR",
                            decision="ERROR",
                            latency_ms=0.0,
                            category="error",
                            original_type="error",
                            confidence_score=0.0,
                            error_message=str(result),
                            timestamp=datetime.now().isoformat()
                        )
                        results.append(error_result)
                    else:
                        results.append(result)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
                
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.1)
        
        self.results = results
        logger.info(f"Completed batch processing. Total results: {len(results)}")
        return results

    def save_results_to_csv(self, output_file: str = None) -> str:
        """Save results to CSV file"""
        if not self.results:
            raise ValueError("No results to save. Run process_batch first.")
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"azure_prompt_shield_results_{timestamp}.csv"
        
        if not Path(output_file).name.startswith('azure_'):
            output_file = f"azure_{Path(output_file).name}"
        
        output_path = self.output_directory / output_file
        
        data = []
        for result in self.results:
            data.append({
                'prompt': result.prompt,
                'original_type': result.original_type,
                'decision': result.decision,
                'latency_ms': result.latency_ms,
                'confidence_score': result.confidence_score,
                'error_message': result.error_message,
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Results saved to {output_path}")
        return str(output_path)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics from the results"""
        if not self.results:
            return {}
        
        total_prompts = len(self.results)
        successful_requests = len([r for r in self.results if r.decision != "ERROR"])
        failed_requests = total_prompts - successful_requests
        
        blocked_prompts = len([r for r in self.results if r.decision == "BLOCKED"])
        allowed_prompts = len([r for r in self.results if r.decision == "ALLOWED"])
        
        latencies = [r.latency_ms for r in self.results if r.decision != "ERROR"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        
        category_stats = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = {
                    'total': 0, 'blocked': 0, 'allowed': 0, 'errors': 0
                }
            category_stats[result.category]['total'] += 1
            if result.decision == "BLOCKED":
                category_stats[result.category]['blocked'] += 1
            elif result.decision == "ALLOWED":
                category_stats[result.category]['allowed'] += 1
            elif result.decision == "ERROR":
                category_stats[result.category]['errors'] += 1
        
        summary = {
            'total_prompts': total_prompts,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'blocked_prompts': blocked_prompts,
            'allowed_prompts': allowed_prompts,
            'success_rate': (successful_requests / total_prompts) * 100 if total_prompts > 0 else 0,
            'block_rate': (blocked_prompts / successful_requests) * 100 if successful_requests > 0 else 0,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'category_breakdown': category_stats
        }
        
        return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Azure Prompt Shield Batch Processor - Python 3.6+ Compatible"
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input file path (supports .csv, .json, .jsonl, .parquet)')
    parser.add_argument('--endpoint', '-e', type=str,
                       help='Azure Content Safety endpoint URL')
    parser.add_argument('--api-key', '-k', type=str,
                       help='Azure Content Safety API key')
    parser.add_argument('--prompt-column', '-p', type=str,
                       help='Name of the column containing prompts')
    parser.add_argument('--ground-truth-column', '-g', type=str,
                       help='Name of the column containing ground truth labels')
    parser.add_argument('--concurrent', '-c', type=int, default=10,
                       help='Maximum concurrent requests (default: 10)')
    parser.add_argument('--rate-limit', '-r', type=int, default=60,
                       help='Rate limit per minute (default: 60)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--output-dir', '-o', type=str, default='azure_results',
                       help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Load and validate input file without making API calls')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--list-columns', action='store_true',
                       help='List available columns in the input file and exit')
    
    return parser.parse_args()

async def main():
    """Main function compatible with Python 3.6+"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get credentials
    endpoint_url = args.endpoint or os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    api_key = args.api_key or os.getenv("AZURE_CONTENT_SAFETY_KEY")
    
    if not endpoint_url:
        logger.error("Azure endpoint URL required via --endpoint or AZURE_CONTENT_SAFETY_ENDPOINT")
        sys.exit(1)
    
    if not api_key and not args.dry_run and not args.list_columns:
        logger.error("Azure API key required via --api-key or AZURE_CONTENT_SAFETY_KEY")
        sys.exit(1)
    
    # Initialize processor
    processor = AzurePromptShieldBatchProcessor(
        endpoint_url=endpoint_url,
        api_key=api_key,
        max_concurrent_requests=args.concurrent,
        rate_limit_per_minute=args.rate_limit,
        timeout_seconds=args.timeout,
        output_directory=args.output_dir,
        prompt_column=args.prompt_column,
        ground_truth_column=args.ground_truth_column
    )
    
    try:
        # Handle utility options
        if args.list_columns:
            file_path = Path(args.input)
            if file_path.suffix.lower() in ['.csv', '.parquet']:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, nrows=0)
                else:
                    if not PYARROW_AVAILABLE:
                        print("Parquet support requires pyarrow")
                        return
                    df = pd.read_parquet(file_path).head(0)
                
                print(f"\nAvailable columns in {file_path.name}:")
                for i, col in enumerate(df.columns, 1):
                    print(f"  {i:2d}. {col}")
                print(f"\nTotal: {len(df.columns)} columns")
                return
        
        # Load prompts
        print(f"Loading prompts from: {args.input}")
        prompts = processor.load_prompts_from_file(args.input)
        
        print(f"Loaded {len(prompts)} prompts")
        
        # Show distribution
        type_counts = {}
        for prompt_data in prompts:
            prompt_type = prompt_data['category']
            type_counts[prompt_type] = type_counts.get(prompt_type, 0) + 1
        
        print("Dataset distribution:")
        for prompt_type, count in type_counts.items():
            print(f"  {prompt_type}: {count} prompts")
        
        # Dry run check
        if args.dry_run:
            print(f"\nDry run completed successfully!")
            print(f"Would process {len(prompts)} prompts with {args.concurrent} concurrent requests")
            return
        
        # Process batch
        print(f"\nStarting processing with Azure Prompt Shield...")
        print(f"Concurrent requests: {args.concurrent}")
        print(f"Rate limit: {args.rate_limit} requests/minute")
        results = await processor.process_batch(prompts)
        
        # Save results
        csv_file = processor.save_results_to_csv()
        print(f"CSV results saved to: {csv_file}")
        
        print(f"\nAll output files saved to: {processor.output_directory.absolute()}")
        
        # Generate and print summary
        summary = processor.generate_summary_report()
        print("\n=== SUMMARY REPORT ===")
        print(f"Total prompts processed: {summary['total_prompts']}")
        print(f"Success rate: {summary['success_rate']:.2f}%")
        print(f"Block rate: {summary['block_rate']:.2f}%")
        print(f"Average latency: {summary['avg_latency_ms']:.2f}ms")
        
        print("\nCategory Breakdown:")
        for category, stats in summary['category_breakdown'].items():
            block_rate = (stats['blocked'] / (stats['blocked'] + stats['allowed']) * 100) if (stats['blocked'] + stats['allowed']) > 0 else 0
            print(f"  {category}: {stats['total']} total, {stats['blocked']} blocked ({block_rate:.1f}%), {stats['allowed']} allowed, {stats['errors']} errors")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Use compatibility function instead of asyncio.run()
    run_async_main(main())
