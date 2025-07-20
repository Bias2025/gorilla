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
            
            # JBB harmful categories - all should be THREAT
            harmful_indicators = [
                'harassment', 'discrimination', 'hate', 'violence', 'harm',
                'malware', 'hacking', 'fraud', 'deception', 'economic',
                'sexual', 'adult', 'privacy', 'disinformation', 'expert',
                'government', 'physical', 'defamation', 'shaming', 'xenophobia'
            ]
            
            # Check if any harmful indicator is in the label
            for indicator in harmful_indicators:
                if indicator in label_lower:
                    return True
            
            # Traditional threat indicators
            if any(word in label_lower for word in ['true', 'yes', '1', 'positive', 'threat']):
                return True
            
            # Safe indicators
            if any(word in label_lower for word in ['safe', 'false', 'no', '0', 'negative']):
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
    """Simple dataset processing"""
    
    logger.info(f"📁 Loading dataset: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        logger.error(f"❌ File not found: {file_path}")
        
        # Suggest similar files
        parent_dir = Path(file_path).parent
        similar_files = list(parent_dir.glob("*jbb*"))
        if similar_files:
            logger.info("🔍 Similar files found:")
            for f in similar_files:
                logger.info(f"   - {f}")
        
        return False
    
    # Load data
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"❌ Unsupported file format: {file_path}")
            return False
        
        logger.info(f"✅ Dataset loaded: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return False
    
    # Detect columns
    if text_column and text_column in df.columns:
        detected_text_col = text_column
        logger.info(f"✅ Using specified text column: {detected_text_col}")
    else:
        # Auto-detect
        text_candidates = ['Goal', 'text', 'prompt', 'content', 'message']
        detected_text_col = None
        for candidate in text_candidates:
            if candidate in df.columns:
                detected_text_col = candidate
                logger.info(f"✅ Auto-detected text column: {detected_text_col}")
                break
        
        if not detected_text_col:
            logger.error(f"❌ No text column found. Available: {list(df.columns)}")
            return False
    
    if label_column and label_column in df.columns:
        detected_label_col = label_column
        logger.info(f"✅ Using specified label column: {detected_label_col}")
    else:
        # Auto-detect
        label_candidates = ['Category', 'label', 'ground_truth', 'class']
        detected_label_col = None
        for candidate in label_candidates:
            if candidate in df.columns:
                detected_label_col = candidate
                logger.info(f"✅ Auto-detected label column: {detected_label_col}")
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