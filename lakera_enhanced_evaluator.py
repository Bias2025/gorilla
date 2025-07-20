#!/usr/bin/env python3
"""
Simple Lakera Guard Evaluator with Verbose Startup
==================================================
"""

import os
import sys
import argparse
import asyncio
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

print("🚀 Starting Lakera Guard Evaluator...")
print("📍 Current directory:", os.getcwd())

# Check dependencies immediately
print("📦 Checking dependencies...")
try:
    import aiohttp
    print("  ✅ aiohttp")
except ImportError as e:
    print(f"  ❌ aiohttp: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("  ✅ pandas")
except ImportError as e:
    print(f"  ❌ pandas: {e}")
    sys.exit(1)

try:
    from tqdm import tqdm
    print("  ✅ tqdm")
except ImportError as e:
    print(f"  ❌ tqdm: {e}")
    sys.exit(1)

print("✅ All dependencies loaded successfully")

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'lakera_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleGroundTruthParser:
    """Simple ground truth parser"""
    
    @staticmethod
    def parse_label(label) -> Optional[bool]:
        """Parse label - anything with harmful content = True (THREAT)"""
        if pd.isna(label) or label is None:
            return None
        
        if isinstance(label, bool):
            return label
        
        if isinstance(label, (int, float)):
            return bool(label)
        
        if isinstance(label, str):
            label_lower = label.lower().strip()
            
            # Jailbreak/adversarial indicators (these are THREATS)
            threat_indicators = [
                'jailbreak', 'adversarial', 'attack', 'prompt_injection', 'injection',
                'harmful', 'dangerous', 'toxic', 'malicious', 'threatening', 'violent',
                'inappropriate', 'unsafe', 'illegal', 'criminal', 'unethical',
                'true', 'yes', '1', 'positive', 'threat'
            ]
            
            # Safe/benign indicators (these are SAFE)
            safe_indicators = [
                'benign', 'safe', 'harmless', 'acceptable', 'appropriate', 'ethical',
                'legal', 'legitimate', 'allowed', 'clean', 'wholesome', 'normal',
                'false', 'no', '0', 'negative'
            ]
            
            # Check for threat indicators
            for indicator in threat_indicators:
                if indicator in label_lower:
                    return True
            
            # Check for safe indicators
            for indicator in safe_indicators:
                if indicator in label_lower:
                    return False
        
        return None

class SimpleLakeraEvaluator:
    """Simple Lakera evaluator with verbose output"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = 'https://api.lakera.ai/v2/guard/results'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.session = None
        
        logger.info(f"🔧 Initialized evaluator")
        logger.info(f"   API URL: {self.url}")
        logger.info(f"   API Key length: {len(api_key)}")
        
    async def __aenter__(self):
        """Create session"""
        logger.info("🌐 Creating HTTP session...")
        
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers
        )
        
        logger.info("✅ HTTP session created")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session"""
        if self.session:
            await self.session.close()
            logger.info("🔌 HTTP session closed")
    
    async def test_api(self) -> bool:
        """Test API with a simple request"""
        logger.info("🧪 Testing API connection...")
        
        try:
            test_prompt = "Hello, how are you today?"
            body = {"messages": [{"role": "user", "content": test_prompt}]}
            
            logger.info(f"📤 Sending test request: {test_prompt}")
            
            async with self.session.post(self.url, json=body) as response:
                logger.info(f"📨 Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    logger.info(f"✅ API test successful - {len(results)} detectors responded")
                    
                    # Show first few detectors
                    for i, detector in enumerate(results[:3]):
                        detector_type = detector.get('detector_type', 'unknown')
                        result = detector.get('result', 'unknown')
                        logger.info(f"   Detector {i+1}: {detector_type} = {result}")
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ API test failed: {response.status}")
                    logger.error(f"   Error: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ API test exception: {e}")
            return False
    
    async def check_single_prompt(self, prompt: str, index: int) -> Dict:
        """Check a single prompt"""
        try:
            start_time = time.time()
            body = {"messages": [{"role": "user", "content": prompt}]}
            
            if index < 3:  # Log details for first 3 prompts
                logger.info(f"🔍 Checking prompt {index}: {prompt[:60]}...")
            
            async with self.session.post(self.url, json=body) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = time.time() - start_time
                    results = data.get("results", [])
                    
                    if index < 3:
                        logger.info(f"   ✅ Success: {len(results)} detectors, {latency*1000:.0f}ms")
                        for detector in results[:2]:
                            logger.info(f"      {detector.get('detector_type')}: {detector.get('result')}")
                    
                    return {
                        "prompt": prompt,
                        "results": results,
                        "latency": latency,
                        "index": index,
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Prompt {index} failed: {response.status} - {error_text}")
                    return {
                        "prompt": prompt,
                        "results": [],
                        "latency": 0,
                        "index": index,
                        "status": "failed",
                        "error": f"{response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Prompt {index} exception: {e}")
            return {
                "prompt": prompt,
                "results": [],
                "latency": 0,
                "index": index,
                "status": "failed",
                "error": str(e)
            }
    
    def analyze_prediction(self, results: List[Dict]) -> str:
        """Simple prediction analysis"""
        threat_detectors = []
        
        for detector in results:
            detector_type = detector.get('detector_type', 'unknown')
            detector_result = detector.get('result', 'l5_unlikely')
            
            if detector_result in ['l1_confident', 'l2_very_likely', 'l3_likely']:
                threat_detectors.append(f"{detector_type}({detector_result})")
        
        if threat_detectors:
            return f"THREAT [{', '.join(threat_detectors)}]"
        else:
            return "SAFE"
    
    def save_results(self, results: List[Dict], ground_truth: List[Optional[bool]], 
                    input_filename: str):
        """Save results to CSV in lakera_results folder"""
        
        # Create lakera_results directory
        results_dir = Path("lakera_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate output filename: lakera_inputname_timestamp.csv
        input_stem = Path(input_filename).stem  # Remove .csv extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"lakera_{input_stem}_{timestamp}.csv"
        output_path = results_dir / output_filename
        
        logger.info(f"💾 Saving results to {output_path}")
        
        csv_data = []
        
        for result in results:
            if result.get('status') == 'success':
                index = result.get('index', 0)
                prompt = result.get('prompt', '')
                latency_ms = result.get('latency', 0) * 1000
                
                # Ground truth
                gt_label = "N/A"
                if ground_truth and index < len(ground_truth):
                    gt_value = ground_truth[index]
                    if gt_value is not None:
                        gt_label = "THREAT" if gt_value else "SAFE"
                
                # Prediction
                prediction = self.analyze_prediction(result.get('results', []))
                
                csv_data.append({
                    'index': index,
                    'prompt': prompt,
                    'ground_truth': gt_label,
                    'prediction': prediction,
                    'latency_ms': f"{latency_ms:.2f}",
                    'detector_count': len(result.get('results', []))
                })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Saved {len(df)} results to {output_path}")
        
        # Print summary
        threat_predictions = df['prediction'].str.contains('THREAT', na=False).sum()
        safe_predictions = len(df) - threat_predictions
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Output file: {output_path}")
        print(f"   Total processed: {len(df)}")
        print(f"   THREAT predictions: {threat_predictions}")
        print(f"   SAFE predictions: {safe_predictions}")
        
        if 'ground_truth' in df.columns:
            gt_available = (df['ground_truth'] != 'N/A').sum()
            print(f"   Ground truth available: {gt_available}")
        
        return str(output_path)

async def process_dataset_simple(file_path: str, api_key: str, 
                               text_column: Optional[str] = None,
                               label_column: Optional[str] = None,
                               max_rows: int = None):
    """Simple dataset processing with enhanced file format support"""
    
    logger.info(f"📁 Loading dataset: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        logger.error(f"❌ File not found: {file_path}")
        
        # Suggest similar files
        parent_dir = Path(file_path).parent
        similar_files = list(parent_dir.glob(f"*{Path(file_path).stem}*"))
        if similar_files:
            logger.info("🔍 Similar files found:")
            for f in similar_files:
                logger.info(f"   - {f}")
        
        return False
    
    # Determine file format and load data
    file_ext = Path(file_path).suffix.lower()
    logger.info(f"📄 File format: {file_ext}")
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            logger.info("✅ CSV file loaded successfully")
            
        elif file_ext == '.parquet':
            # Enhanced Parquet loading with error handling
            try:
                # Try with pandas first
                df = pd.read_parquet(file_path)
                logger.info("✅ Parquet file loaded with pandas")
            except Exception as pandas_error:
                logger.warning(f"⚠️ Pandas failed: {pandas_error}")
                
                # Try with pyarrow directly if available
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    logger.info("✅ Parquet file loaded with pyarrow")
                except ImportError:
                    logger.error("❌ pyarrow not available. Install with: pip install pyarrow")
                    return False
                except Exception as pyarrow_error:
                    logger.error(f"❌ PyArrow failed: {pyarrow_error}")
                    
                    # Try alternative approach - read as bytes and diagnose
                    try:
                        file_size = Path(file_path).stat().st_size
                        logger.info(f"📊 File size: {file_size} bytes")
                        
                        # Read first few bytes to check format
                        with open(file_path, 'rb') as f:
                            header = f.read(16)
                            logger.info(f"🔍 File header (hex): {header.hex()}")
                            logger.info(f"🔍 File header (ascii): {header}")
                        
                        # Check if it's actually a different format
                        if b'PAR1' not in header and b'parquet' not in header.lower():
                            logger.error("❌ File doesn't appear to be a valid Parquet file")
                            logger.info("💡 Try converting to CSV or check file integrity")
                        else:
                            logger.error("❌ Parquet file appears valid but cannot be read")
                    except Exception as diag_error:
                        logger.error(f"❌ Cannot diagnose file: {diag_error}")
                    
                    return False
            
        elif file_ext == '.jsonl':
            # Enhanced JSONL loading
            logger.info("📋 Loading JSONL file...")
            data = []
            line_count = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                json_obj = json.loads(line)
                                data.append(json_obj)
                                line_count += 1
                                
                                # Progress for large files
                                if line_count % 10000 == 0:
                                    logger.info(f"   Loaded {line_count:,} lines...")
                                    
                            except json.JSONDecodeError as json_err:
                                logger.warning(f"⚠️ Skipping invalid JSON on line {line_num}: {json_err}")
                                continue
                
                if not data:
                    logger.error("❌ No valid JSON objects found in JSONL file")
                    return False
                
                df = pd.DataFrame(data)
                logger.info(f"✅ JSONL file loaded: {line_count:,} records")
                
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        logger.info(f"🔄 Trying encoding: {encoding}")
                        with open(file_path, 'r', encoding=encoding) as f:
                            data = []
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        data.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                        
                        if data:
                            df = pd.DataFrame(data)
                            logger.info(f"✅ JSONL loaded with {encoding} encoding")
                            break
                    except Exception:
                        continue
                else:
                    logger.error("❌ Could not decode JSONL file with any encoding")
                    return False
        
        elif file_ext == '.json':
            # Regular JSON file
            logger.info("📋 Loading JSON file...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Check if it's a nested structure
                    if 'data' in json_data:
                        df = pd.DataFrame(json_data['data'])
                    elif 'rows' in json_data:
                        df = pd.DataFrame(json_data['rows'])
                    else:
                        # Try to convert dict to single-row DataFrame
                        df = pd.DataFrame([json_data])
                else:
                    logger.error("❌ Unsupported JSON structure")
                    return False
                
                logger.info("✅ JSON file loaded successfully")
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON format: {e}")
                return False
        
        else:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            logger.info("💡 Supported formats: .csv, .parquet, .jsonl, .json")
            return False
        
        logger.info(f"✅ Dataset loaded: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        
        # Show data types
        logger.info("📊 Column types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"   {col}: {dtype}")
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        logger.error("💡 Troubleshooting tips:")
        logger.error("   - Check file isn't corrupted")
        logger.error("   - Try opening in a text editor to verify format")
        logger.error("   - For Parquet: pip install pyarrow")
        logger.error("   - Consider converting to CSV format")
        return False
    
    # Detect columns
    if text_column and text_column in df.columns:
        detected_text_col = text_column
        logger.info(f"✅ Using specified text column: {detected_text_col}")
    else:
        # Enhanced auto-detection for different formats
        text_candidates = [
            # Common in datasets
            'prompt', 'text', 'content', 'message', 'input', 'query', 'goal', 'instruction',
            # HuggingFace datasets
            'conversations', 'chat', 'dialogue', 'messages',
            # Safety datasets  
            'question', 'user_input', 'request', 'task'
        ]
        
        detected_text_col = None
        
        # Exact matches first
        for candidate in text_candidates:
            for col in df.columns:
                if candidate.lower() == col.lower():
                    detected_text_col = col
                    logger.info(f"✅ Auto-detected text column: {detected_text_col} (exact match)")
                    break
            if detected_text_col:
                break
        
        # Partial matches if no exact match
        if not detected_text_col:
            for candidate in text_candidates:
                for col in df.columns:
                    if candidate.lower() in col.lower():
                        detected_text_col = col
                        logger.info(f"✅ Auto-detected text column: {detected_text_col} (partial match)")
                        break
                if detected_text_col:
                    break
        
        # Find by content length if still not found
        if not detected_text_col:
            text_candidates_by_length = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains text-like data
                    sample_data = df[col].dropna().head(10)
                    if len(sample_data) > 0:
                        avg_length = sample_data.astype(str).str.len().mean()
                        max_length = sample_data.astype(str).str.len().max()
                        
                        # Likely text if average > 50 chars and max > 100 chars
                        if avg_length > 50 and max_length > 100:
                            text_candidates_by_length.append((col, avg_length))
                            logger.info(f"📝 Text candidate: '{col}' (avg: {avg_length:.1f}, max: {max_length})")
            
            if text_candidates_by_length:
                detected_text_col = max(text_candidates_by_length, key=lambda x: x[1])[0]
                logger.info(f"✅ Auto-detected text column: {detected_text_col} (by length)")
        
        if not detected_text_col:
            logger.error(f"❌ No text column found. Available columns: {list(df.columns)}")
            logger.info("💡 Specify text column with --text-column parameter")
            return False
    
    # Enhanced label column detection
    if label_column and label_column in df.columns:
        detected_label_col = label_column
        logger.info(f"✅ Using specified label column: {detected_label_col}")
    else:
        # Enhanced auto-detection for labels
        label_candidates = [
            # Standard ML terms
            'type', 'label', 'labels', 'ground_truth', 'groundtruth', 'gt', 'class', 'category', 
            'classification', 'target', 'y', 'output',
            # Safety/jailbreak specific
            'is_safe', 'safety', 'risk', 'harmful', 'toxicity', 'jailbreak', 'adversarial',
            # Behavior analysis
            'behavior', 'behaviour', 'intent', 'malicious', 'benign'
        ]
        
        detected_label_col = None
        
        # Exact matches first
        for candidate in label_candidates:
            for col in df.columns:
                if candidate.lower() == col.lower():
                    detected_label_col = col
                    logger.info(f"✅ Auto-detected label column: {detected_label_col} (exact match)")
                    break
            if detected_label_col:
                break
        
        # Partial matches
        if not detected_label_col:
            for candidate in label_candidates:
                for col in df.columns:
                    if candidate.lower() in col.lower():
                        detected_label_col = col
                        logger.info(f"✅ Auto-detected label column: {detected_label_col} (partial match)")
                        break
                if detected_label_col:
                    break
        
        # Analyze columns by content if still not found
        if not detected_label_col:
            for col in df.columns:
                if col != detected_text_col and df[col].dtype == 'object':
                    unique_values = df[col].dropna().unique()
                    if 2 <= len(unique_values) <= 10:  # Reasonable number of categories
                        logger.info(f"🎯 Potential label column: '{col}' with values: {list(unique_values)}")
                        
                        # Check if values look like labels
                        label_like_score = 0
                        for value in unique_values:
                            value_str = str(value).lower()
                            if any(term in value_str for term in ['safe', 'unsafe', 'jailbreak', 'benign', 'harmful', 'true', 'false', '0', '1']):
                                label_like_score += 1
                        
                        if label_like_score > 0:
                            detected_label_col = col
                            logger.info(f"✅ Auto-detected label column: {detected_label_col} (by content)")
                            break
    
    # Analyze ground truth
    ground_truth = []
    if detected_label_col:
        logger.info(f"🎯 Analyzing ground truth column: {detected_label_col}")
        unique_values = df[detected_label_col].unique()
        logger.info(f"   Unique values: {list(unique_values)}")
        
        for label in df[detected_label_col]:
            parsed = SimpleGroundTruthParser.parse_label(label)
            ground_truth.append(parsed)
        
        threat_count = sum(1 for gt in ground_truth if gt is True)
        safe_count = sum(1 for gt in ground_truth if gt is False)
        unknown_count = sum(1 for gt in ground_truth if gt is None)
        
        logger.info(f"   Parsed as THREAT: {threat_count}")
        logger.info(f"   Parsed as SAFE: {safe_count}")
        logger.info(f"   Unknown: {unknown_count}")
    else:
        logger.warning("⚠️ No ground truth column found")
        ground_truth = [None] * len(df)
    
    # Extract texts
    texts = df[detected_text_col].astype(str).tolist()
    
    # Limit rows for testing
    if max_rows and max_rows < len(texts):
        texts = texts[:max_rows]
        ground_truth = ground_truth[:max_rows]
        logger.info(f"🔢 Limited to first {max_rows} rows for testing")
    
    # Process with API
    async with SimpleLakeraEvaluator(api_key) as evaluator:
        
        # Test API first
        if not await evaluator.test_api():
            logger.error("❌ API test failed - stopping")
            return False
        
        logger.info(f"🚀 Processing {len(texts)} prompts...")
        
        all_results = []
        
        # Process with progress bar
        with tqdm(total=len(texts), desc="Processing") as pbar:
            for i, text in enumerate(texts):
                result = await evaluator.check_single_prompt(text, i)
                all_results.append(result)
                pbar.update(1)
                
                # Small delay to be nice to API
                await asyncio.sleep(0.1)
        
        # Save results
        output_file = evaluator.save_results(all_results, ground_truth, file_path)
        
        return True

def main():
    """Main function with verbose startup"""
    print("🎯 Parsing arguments...")
    
    parser = argparse.ArgumentParser(description='Simple Lakera Guard Evaluator')
    parser.add_argument('--datasets', nargs='+', required=True, help='Dataset files')
    parser.add_argument('--env', required=True, help='API key')
    parser.add_argument('--text-column', help='Text column name')
    parser.add_argument('--label-column', help='Label column name')
    parser.add_argument('--max-rows', type=int, help='Limit rows for testing')
    
    args = parser.parse_args()
    
    print(f"✅ Arguments parsed:")
    print(f"   Datasets: {args.datasets}")
    print(f"   Text column: {args.text_column}")
    print(f"   Label column: {args.label_column}")
    print(f"   API key length: {len(args.env)}")
    if args.max_rows:
        print(f"   Max rows: {args.max_rows}")
    
    # Process each dataset
    for dataset_path in args.datasets:
        print(f"\n🎯 Processing: {dataset_path}")
        
        try:
            success = asyncio.run(process_dataset_simple(
                dataset_path, args.env, args.text_column, args.label_column, args.max_rows
            ))
            
            if success:
                print(f"✅ Successfully processed {dataset_path}")
            else:
                print(f"❌ Failed to process {dataset_path}")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error processing {dataset_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
