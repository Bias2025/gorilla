import requests
import json
import csv
import argparse
import logging
import time
import sys
import os
import tomllib
import pandas as pd
from datetime import datetime
from pathlib import Path

# Default configuration values
DEFAULT_CONFIG = {
    'api': {
        'endpoint': 'https://service.api.aisecurity.paloaltonetworks.com/v1/scan/sync/request',
        'max_retries': 3,
        'retry_delay': 1
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
}

def load_config(config_file):
    """Load configuration from TOML file."""
    config = DEFAULT_CONFIG.copy()
    
    # Load from TOML file if it exists
    if os.path.exists(config_file):
        with open(config_file, 'rb') as f:
            toml_config = tomllib.load(f)
            # Handle nested dictionaries
            if 'api' in toml_config:
                config['api'].update(toml_config['api'])
            if 'logging' in toml_config:
                config['logging'].update(toml_config['logging'])
    else:
        logger.warning("Configuration file '%s' not found. Using default configuration.", config_file)
    
    return config

# Global variables to be set after config is loaded
config = None
url = None
headers = None
logger = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process prompts from CSV/Parquet and scan with Palo Alto Networks AI Security Service')
    parser.add_argument('--config', type=str, default='config.toml', help='Path to TOML configuration file (default: config.toml)')
    parser.add_argument('--input', type=str, required=True, help='Path to input file (CSV or Parquet)')
    parser.add_argument('--prompt-column', type=str, default='prompt', help='Name of the column containing prompts (default: prompt)')
    return parser.parse_args()

def send_request_with_retry(payload, max_retries=None):
    """Send request with retry mechanism for client errors."""
    if max_retries is None:
        max_retries = config['api']['max_retries']
    
    for attempt in range(max_retries):
        try:
            # Start timing
            start_time = time.time()
            
            # Send request
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            # Calculate latency
            latency = round(time.time() - start_time, 3)
            
            # If successful or server error (5xx), return response
            if response.status_code < 400 or response.status_code >= 500:
                return response, latency
            
            # For client errors (4xx), retry if not the last attempt
            if attempt < max_retries - 1:
                logger.warning("Request failed with status %d. Retrying... (Attempt %d/%d)", response.status_code, attempt + 2, max_retries)
                time.sleep(config['api']['retry_delay'])
            else:
                return response, latency
                
        except Exception as e:
            logger.error("Request exception: %s", str(e))
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

def read_input_file(input_file, prompt_column):
    """Read prompts from CSV or Parquet file."""
    file_path = Path(input_file)
    
    if not file_path.exists():
        logger.error("Input file '%s' not found", input_file)
        sys.exit(1)
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(input_file)
        elif file_extension == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            logger.error("Unsupported file format '%s'. Only CSV and Parquet files are supported.", file_extension)
            sys.exit(1)
        
        # Check if the specified prompt column exists
        if prompt_column not in df.columns:
            logger.error("Column '%s' not found in input file. Available columns: %s", prompt_column, list(df.columns))
            sys.exit(1)
        
        return df[prompt_column].dropna().tolist()
    
    except Exception as e:
        logger.error("Error reading input file '%s': %s", input_file, str(e))
        sys.exit(1)

def generate_output_filename(input_file):
    """Generate output filename with input filename prefix and timestamp suffix."""
    input_path = Path(input_file)
    input_name = input_path.stem  # filename without extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{input_name}_{timestamp}.csv"

def process_prompts(input_file, profile_id, prompt_column):
    """Process prompts from input file and save results."""
    
    # Record start time
    process_start_time = time.time()
    logger.info("Processing started at %s", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process_start_time)))
    
    # Read prompts from input file
    prompts = read_input_file(input_file, prompt_column)
    
    # Generate output filename
    output_file = generate_output_filename(input_file)
    logger.info("Output will be saved to: %s", output_file)
    
    # Create output CSV with headers
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['prompt', 'action', 'category', 'scan_id', 'report_id', 'profile_name', 'latency', 'status_code'])
    
    # Process prompts
    for index, prompt in enumerate(prompts, start=1):
        # Skip empty prompts
        if not prompt.strip():
            continue
            
        prompt = prompt.strip()
        logger.info("Processing row %d: %s...", index, prompt[:50])
            
            # Prepare payload
            payload = {
                "tr_id": str(index),
                "ai_profile": {
                    "profile_id": profile_id
                },
                "contents": [
                    {
                        "prompt": prompt
                    }
                ]
            }
            
            try:
                # Send request with retry
                response, latency = send_request_with_retry(payload)
                
                # Extract values from response
                if response.status_code == 200:
                    data = response.json()
                    action = data.get('action', '')
                    category = data.get('category', '')
                    scan_id = data.get('scan_id', '')
                    report_id = data.get('report_id', '')
                    profile_name = data.get('profile_name', '')
                else:
                    # For error responses, use empty values
                    action = ''
                    category = ''
                    scan_id = ''
                    report_id = ''
                    profile_name = ''
                
                # Append results to CSV
                with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([prompt, action, category, scan_id, report_id, profile_name, latency, response.status_code])
                
                logger.info("Row %d processed successfully (Status: %d)", index, response.status_code)
                
            except Exception as e:
                logger.error("Failed to process row %d: %s", index, str(e))
                # Write error entry to CSV
                with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow([prompt, '', '', '', '', '', 0, 'ERROR'])
    
    # Calculate and log total execution time
    process_end_time = time.time()
    total_duration = process_end_time - process_start_time
    
    logger.info("Processing complete! Results saved to '%s'", output_file)
    logger.info("Processing ended at %s", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process_end_time)))
    logger.info("Total execution time: %.2f seconds (%.2f minutes)", total_duration, total_duration/60)

def main():
    """Main function."""
    global config, url, headers, logger
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    logger = logging.getLogger(__name__)
    
    # Get API key and Profile ID from environment variables
    api_key = os.getenv('PALO_ALTO_API_KEY')
    profile_id = os.getenv('PALO_ALTO_PROFILE_ID')
    
    # Validate required environment variables
    if not api_key:
        logger.error("API key not found. Please set PALO_ALTO_API_KEY environment variable.")
        sys.exit(1)
    if not profile_id:
        logger.error("Profile ID not found. Please set PALO_ALTO_PROFILE_ID environment variable.")
        sys.exit(1)
    
    # Set up API configuration
    url = config['api']['endpoint']
    headers = {
        'x-pan-token': api_key,
        'Content-Type': 'application/json'
    }
    
    logger.info("Starting prompt processing from '%s' with profile ID '%s'", args.input, profile_id)
    logger.info("Using prompt column: '%s'", args.prompt_column)
    logger.info("Configuration loaded from: %s", args.config)
    process_prompts(args.input, profile_id, args.prompt_column)

if __name__ == "__main__":
    main()
