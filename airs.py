#!/usr/bin/env python3
"""
Palo Alto Networks AI Security Service Scanner

A production-ready script to process prompts from CSV/Parquet files
and scan them using the Palo Alto Networks AI Security Service API.

Usage:
    export PALO_ALTO_API_KEY="your_api_key"
    export PALO_ALTO_PROFILE_ID="your_profile_id"
    python scan.py --input data.csv --prompt-column "question"

Requirements:
    - pandas
    - requests
    - Python 3.8+
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Version and metadata
__version__ = "2.0.0"
__author__ = "Security Team"

# Default configuration values
DEFAULT_CONFIG = {
    'api': {
        'endpoint': 'https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request',
        'max_retries': 3,
        'retry_delay': 1,
        'timeout': 30,
        'batch_size': 100
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

# Global variables
config: Dict = {}
logger: logging.Logger = logging.getLogger(__name__)


class PaloAltoScanner:
    """Main scanner class for handling API requests and data processing."""
    
    def __init__(self, api_key: str, profile_id: str, config: Dict):
        """Initialize the scanner with API credentials and configuration."""
        self.api_key = api_key
        self.profile_id = profile_id
        self.config = config
        self.url = config['api']['endpoint']
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config['api']['max_retries'],
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'x-pan-token': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': f'PaloAltoScanner/{__version__}'
        })
        
        return session
    
    def send_request(self, payload: Dict) -> Tuple[requests.Response, float]:
        """Send a single request to the API with timing."""
        start_time = time.time()
        
        try:
            response = self.session.post(
                self.url,
                data=json.dumps(payload),
                timeout=self.config['api']['timeout']
            )
            latency = round(time.time() - start_time, 3)
            return response, latency
            
        except requests.exceptions.RequestException as e:
            latency = round(time.time() - start_time, 3)
            logger.error("Request failed: %s", str(e))
            # Create a mock response for error handling
            mock_response = requests.Response()
            mock_response.status_code = 0
            mock_response._content = json.dumps({"error": str(e)}).encode()
            return mock_response, latency
    
    def process_prompt(self, prompt: str, index: int) -> Dict:
        """Process a single prompt and return the result."""
        payload = {
            "tr_id": str(index),
            "ai_profile": {
                "profile_id": self.profile_id
            },
            "contents": [
                {
                    "prompt": prompt
                }
            ]
        }
        
        response, latency = self.send_request(payload)
        
        # Parse response
        result = {
            'prompt': prompt,
            'latency': latency,
            'status_code': response.status_code,
            'action': '',
            'category': '',
            'scan_id': '',
            'report_id': '',
            'profile_name': ''
        }
        
        if response.status_code == 200:
            try:
                data = response.json()
                result.update({
                    'action': data.get('action', ''),
                    'category': data.get('category', ''),
                    'scan_id': data.get('scan_id', ''),
                    'report_id': data.get('report_id', ''),
                    'profile_name': data.get('profile_name', '')
                })
            except json.JSONDecodeError:
                logger.warning("Invalid JSON response for prompt %d", index)
                result['status_code'] = 'INVALID_JSON'
        elif response.status_code == 0:
            result['status_code'] = 'CONNECTION_ERROR'
        
        return result


def load_config(config_file: str) -> Dict:
    """Load configuration from TOML file with fallback to defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'rb') as f:
                toml_config = tomllib.load(f)
                
            # Update nested dictionaries
            for section in ['api', 'logging']:
                if section in toml_config:
                    config[section].update(toml_config[section])
                    
            logger.info("Configuration loaded from: %s", config_file)
            
        except Exception as e:
            logger.warning("Error loading config file '%s': %s. Using defaults.", config_file, str(e))
    else:
        logger.info("Configuration file '%s' not found. Using defaults.", config_file)
    
    return config


def setup_logging(config: Dict) -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format=config['logging']['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"scan_{datetime.now().strftime('%Y%m%d')}.log")
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process prompts from CSV/Parquet files using Palo Alto Networks AI Security Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input prompts.csv
  %(prog)s --input data.parquet --prompt-column "question"
  %(prog)s --input data.csv --prompt-column "user_input" --config custom.toml

Environment Variables:
  PALO_ALTO_API_KEY      API key for Palo Alto Networks service (required)
  PALO_ALTO_PROFILE_ID   Profile ID for scanning (required)
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input file (CSV or Parquet format)'
    )
    
    parser.add_argument(
        '--prompt-column', 
        type=str, 
        default='prompt',
        help='Name of the column containing prompts (default: prompt)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.toml',
        help='Path to TOML configuration file (default: config.toml)'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'%(prog)s {__version__}'
    )
    
    return parser.parse_args()


def read_input_file(input_file: str, prompt_column: str) -> List[str]:
    """Read and validate prompts from CSV or Parquet file."""
    file_path = Path(input_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    if not file_path.is_file():
        raise ValueError(f"Path '{input_file}' is not a file")
    
    file_extension = file_path.suffix.lower()
    
    try:
        # Read file based on extension
        if file_extension == '.csv':
            df = pd.read_csv(input_file, encoding='utf-8')
        elif file_extension == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            raise ValueError(f"Unsupported file format '{file_extension}'. Only CSV and Parquet files are supported.")
        
        # Validate dataframe
        if df.empty:
            raise ValueError("Input file is empty")
        
        # Check if prompt column exists
        if prompt_column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Column '{prompt_column}' not found. Available columns: {available_columns}")
        
        # Extract and clean prompts
        prompts = df[prompt_column].dropna().astype(str).str.strip()
        prompts = prompts[prompts != ''].tolist()
        
        if not prompts:
            raise ValueError(f"No valid prompts found in column '{prompt_column}'")
        
        logger.info("Successfully loaded %d prompts from %s", len(prompts), input_file)
        return prompts
        
    except pd.errors.EmptyDataError:
        raise ValueError("Input file is empty or has no data")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing input file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading input file: {str(e)}")


def generate_output_filename(input_file: str) -> str:
    """Generate output filename with input filename prefix and timestamp suffix."""
    input_path = Path(input_file)
    input_name = input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{input_name}_{timestamp}.csv"


def write_result_to_csv(output_file: str, result: Dict, write_header: bool = False, prompt_column: str = 'prompt') -> None:
    """Write a single result to the CSV output file."""
    mode = 'w' if write_header else 'a'
    
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow([
                prompt_column, 'action', 'category', 'scan_id', 
                'report_id', 'profile_name', 'latency', 'status_code'
            ])
        
        # Only write data row if result contains data
        if result and 'prompt' in result:
            writer.writerow([
                result['prompt'], result['action'], result['category'],
                result['scan_id'], result['report_id'], result['profile_name'],
                result['latency'], result['status_code']
            ])


def process_prompts(input_file: str, profile_id: str, prompt_column: str) -> str:
    """Process all prompts from input file and save results."""
    start_time = time.time()
    logger.info("Processing started at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    try:
        # Read prompts
        prompts = read_input_file(input_file, prompt_column)
        total_prompts = len(prompts)
        
        # Initialize scanner
        api_key = os.getenv('PALO_ALTO_API_KEY')
        scanner = PaloAltoScanner(api_key, profile_id, config)
        
        # Generate output filename
        output_file = generate_output_filename(input_file)
        logger.info("Output will be saved to: %s", output_file)
        
        # Write CSV header
        write_result_to_csv(output_file, {}, write_header=True, prompt_column=prompt_column)
        
        # Process prompts with progress tracking
        successful_requests = 0
        failed_requests = 0
        
        for index, prompt in enumerate(prompts, start=1):
            if not prompt.strip():
                continue
            
            # Log progress for large datasets
            if index % 10 == 0 or index == 1:
                logger.info("Processing prompt %d/%d (%.1f%% complete)", 
                           index, total_prompts, (index/total_prompts)*100)
            
            try:
                # Process prompt
                result = scanner.process_prompt(prompt, index)
                
                # Write result
                write_result_to_csv(output_file, result, prompt_column=prompt_column)
                
                # Update counters
                if result['status_code'] == 200:
                    successful_requests += 1
                else:
                    failed_requests += 1
                
                # Log individual result
                logger.debug("Prompt %d processed - Status: %s, Latency: %s ms", 
                           index, result['status_code'], result['latency'])
                
            except Exception as e:
                logger.error("Failed to process prompt %d: %s", index, str(e))
                failed_requests += 1
                
                # Write error result
                error_result = {
                    'prompt': prompt, 'action': '', 'category': '',
                    'scan_id': '', 'report_id': '', 'profile_name': '',
                    'latency': 0, 'status_code': 'ERROR'
                }
                write_result_to_csv(output_file, error_result, prompt_column=prompt_column)
        
        # Calculate metrics
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Log summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info("Total prompts processed: %d", total_prompts)
        logger.info("Successful requests: %d", successful_requests)
        logger.info("Failed requests: %d", failed_requests)
        logger.info("Success rate: %.1f%%", (successful_requests/total_prompts)*100 if total_prompts > 0 else 0)
        logger.info("Total execution time: %.2f seconds (%.2f minutes)", total_duration, total_duration/60)
        logger.info("Average latency per request: %.2f seconds", total_duration/total_prompts if total_prompts > 0 else 0)
        logger.info("Results saved to: %s", output_file)
        logger.info("=" * 60)
        
        return output_file
        
    except Exception as e:
        logger.error("Critical error during processing: %s", str(e))
        raise


def validate_environment() -> Tuple[str, str]:
    """Validate required environment variables."""
    api_key = os.getenv('PALO_ALTO_API_KEY')
    profile_id = os.getenv('PALO_ALTO_PROFILE_ID')
    
    if not api_key:
        raise EnvironmentError(
            "API key not found. Please set PALO_ALTO_API_KEY environment variable."
        )
    
    if not profile_id:
        raise EnvironmentError(
            "Profile ID not found. Please set PALO_ALTO_PROFILE_ID environment variable."
        )
    
    return api_key, profile_id


def main() -> None:
    """Main function with comprehensive error handling."""
    global config
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Set up logging  
        setup_logging(config)
        
        # Log startup info
        logger.info("Starting Palo Alto Networks AI Security Scanner v%s", __version__)
        logger.info("Input file: %s", args.input)
        logger.info("Prompt column: %s", args.prompt_column)
        
        # Validate environment
        api_key, profile_id = validate_environment()
        logger.info("Environment variables validated successfully")
        
        # Process prompts
        output_file = process_prompts(args.input, profile_id, args.prompt_column)
        
        logger.info("Scan completed successfully. Results saved to: %s", output_file)
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
        
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        logger.error("Configuration/Input error: %s", str(e))
        sys.exit(1)
        
    except Exception as e:
        logger.critical("Unexpected error: %s", str(e), exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
