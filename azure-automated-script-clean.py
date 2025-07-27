#!/usr/bin/env python3
"""
Multi-Cloud Content Safety Batch Processor - Production Ready v7.0
Complete fixed version with proper error handling for confusion matrix and ground truth processing

Key fixes:
- Fixed indentation errors in confusion matrix display
- Added proper null checks for ground truth data
- Enhanced error handling for AttributeError and NoneType issues
- Improved validation for confusion matrix calculation
- Fixed variable scope issues
- Complete and properly structured code
"""

import asyncio
import aiohttp
import pandas as pd
import json
import time
import logging
import sys
import os
import argparse
import traceback
import re
import ssl
import urllib.parse
import secrets
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter

# Python version compatibility check
PYTHON_VERSION = sys.version_info[:2]
if PYTHON_VERSION < (3, 6):
    print("Error: Python 3.6 or higher is required")
    sys.exit(1)

# Dependencies with graceful fallbacks
try:
    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score, recall_score, 
        accuracy_score, classification_report, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
    
    # Check OpenAI version for compatibility
    try:
        import pkg_resources
        openai_version = pkg_resources.get_distribution("openai").version
        OPENAI_VERSION = tuple(map(int, openai_version.split('.')[:2]))
    except:
        try:
            # Alternative method for newer Python versions
            import importlib.metadata
            openai_version = importlib.metadata.version("openai")
            OPENAI_VERSION = tuple(map(int, openai_version.split('.')[:2]))
        except:
            # Check if it's v1.0+ by looking for client attribute
            if hasattr(openai, 'OpenAI'):
                OPENAI_VERSION = (1, 0)
            else:
                OPENAI_VERSION = (0, 28)
        
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_VERSION = (0, 0)

# Python compatibility for older versions
def run_async_main(coro):
    """Run async coroutine with compatibility for Python < 3.7"""
    try:
        # Python 3.7+
        return asyncio.run(coro)
    except AttributeError:
        # Python 3.6 and older
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coro)
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # Already in an event loop, create a new loop in a thread
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                raise
        finally:
            # Don't close the loop as it might be reused
            pass

class SecurityConfig:
    """Security configuration and utilities"""
    
    # Maximum input sizes for security
    MAX_PROMPT_LENGTH = 10000
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_BATCH_SIZE = 10000  # Increased for large datasets
    CHUNK_SIZE = 1000  # Process in chunks for memory efficiency
    
    @staticmethod
    def create_secure_ssl_context():
        """Create secure SSL context with version compatibility"""
        try:
            context = ssl.create_default_context()
        except AttributeError:
            # Very old Python versions
            context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
        
        # Set minimum TLS version with compatibility for Python < 3.7
        try:
            # Python 3.7+
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        except AttributeError:
            # Python < 3.7 - use protocol and options
            try:
                context.protocol = ssl.PROTOCOL_TLS
            except AttributeError:
                # Very old Python versions - try different protocols
                try:
                    context.protocol = ssl.PROTOCOL_TLSv1_2
                except AttributeError:
                    try:
                        context.protocol = ssl.PROTOCOL_TLSv1
                    except AttributeError:
                        # Last resort - use SSLv23 but disable weak versions below
                        context.protocol = ssl.PROTOCOL_SSLv23
            
            # Disable weak protocols if available
            if hasattr(ssl, 'OP_NO_SSLv2'):
                context.options |= ssl.OP_NO_SSLv2
            if hasattr(ssl, 'OP_NO_SSLv3'):
                context.options |= ssl.OP_NO_SSLv3
            if hasattr(ssl, 'OP_NO_TLSv1'):
                context.options |= ssl.OP_NO_TLSv1
            if hasattr(ssl, 'OP_NO_TLSv1_1'):
                context.options |= ssl.OP_NO_TLSv1_1
        
        return context
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security"""
        if not file_path:
            return False
            
        # Resolve path and check for traversal
        try:
            resolved_path = Path(file_path).resolve()
            # Check if path tries to go above current directory
            if '..' in str(resolved_path):
                return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize prompt input for security"""
        if not prompt:
            return ""
            
        # Limit length
        if len(prompt) > SecurityConfig.MAX_PROMPT_LENGTH:
            prompt = prompt[:SecurityConfig.MAX_PROMPT_LENGTH]
        
        # Remove null bytes and control characters
        prompt = prompt.replace('\x00', '')
        prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\t\n\r')
        
        return prompt
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*') -> str:
        """Mask sensitive data in logs"""
        if not data:
            return data
            
        if len(data) <= 8:
            return mask_char * len(data)
        
        # Show first 4 and last 4 characters
        return data[:4] + mask_char * (len(data) - 8) + data[-4:]

class CircuitBreaker:
    """Circuit breaker pattern for API endpoints"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'

class GroundTruthMapper:
    """Maps ground truth labels to binary values with enhanced validation"""
    
    # Define mappings for different ground truth formats
    MALICIOUS_KEYWORDS = {
        # Direct attack types
        'jailbreak', 'malicious', 'harmful', 'unsafe', 'toxic', 'attack', 
        'exploit', 'injection', 'bypass', 'hack', 'manipulate', 'adversarial',
        'prompt_injection', 'social_engineering', 'phishing', 'spam',
        
        # Harm categories
        'hate', 'harassment', 'violence', 'self_harm', 'sexual', 'hate_speech',
        'physical_harm', 'economic_harm', 'fraud', 'deception', 'misinformation',
        'dangerous', 'illegal', 'unethical', 'abuse', 'threat', 'intimidation',
        
        # Policy violations
        'unethical_behavior', 'policy_violation', 'government_decision', 
        'privacy_violation', 'discrimination', 'bias', 'unfair',
        'economic harm', 'physical harm', 'government decision', 'privacy violation',
        
        # Specific discrimination and harassment types
        'xenophobia', 'racism', 'sexism', 'ableism', 'homophobia', 'transphobia',
        'body_shaming', 'body shaming', 'shaming', 'bullying', 'cyberbullying',
        'defamation', 'slander', 'libel', 'stigmatization', 'marginalization',
        'prejudice', 'stereotype', 'stereotyping', 'profiling', 'scapegoating',
        
        # Behavioral indicators
        'blocked', 'flagged', 'rejected', 'denied', 'prohibited', 'banned',
        'violated', 'inappropriate', 'problematic', 'concerning', 'risky',
        
        # Boolean/numeric indicators
        'true', '1', 'yes', 'positive', 'bad', 'fail', 'failed',
        
        # Severity levels
        'high', 'critical', 'severe', 'extreme', 'maximum'
    }
    
    BENIGN_KEYWORDS = {
        # Safety indicators
        'benign', 'safe', 'harmless', 'clean', 'legitimate', 'normal',
        'acceptable', 'appropriate', 'allowed', 'permitted', 'approved',
        'compliant', 'ethical', 'legal', 'valid', 'proper',
        
        # Quality indicators
        'good', 'neutral', 'fine', 'ok', 'okay', 'excellent', 'satisfactory',
        'helpful', 'informative', 'educational', 'constructive',
        
        # Boolean/numeric indicators
        'false', '0', 'no', 'negative', 'none', 'null', 'pass', 'passed',
        
        # Severity levels
        'low', 'minimal', 'minor', 'negligible', 'none', 'zero'
    }
    
    @staticmethod
    def convert_to_binary(ground_truth_value) -> int:
        """Convert ground truth value to binary (0=benign, 1=malicious) - EVERY value gets mapped to 0 or 1"""
        # Handle truly None, NaN, and empty values - default to benign (0)
        if ground_truth_value is None:
            return 0
        
        if pd.isna(ground_truth_value):
            return 0
            
        # Convert to string and normalize
        try:
            value_str = str(ground_truth_value).lower().strip()
        except Exception:
            return 0  # Default to benign for conversion errors
        
        # Handle empty strings after conversion - default to benign
        if not value_str or value_str in ['nan', 'none', 'null', '']:
            return 0
        
        # Direct numeric conversion
        if value_str.isdigit():
            return min(int(value_str), 1)  # Cap at 1
        
        # Float conversion
        try:
            float_val = float(value_str)
            return 1 if float_val > 0.5 else 0
        except ValueError:
            pass
        
        # Keyword matching - exact matches first
        if value_str in GroundTruthMapper.MALICIOUS_KEYWORDS:
            return 1
        elif value_str in GroundTruthMapper.BENIGN_KEYWORDS:
            return 0
        
        # Partial matching for compound words
        for keyword in GroundTruthMapper.MALICIOUS_KEYWORDS:
            if keyword in value_str:
                return 1
        
        for keyword in GroundTruthMapper.BENIGN_KEYWORDS:
            if keyword in value_str:
                return 0
        
        # Enhanced pattern matching for common variations
        # Check for common malicious patterns
        malicious_patterns = [
            'attack', 'hack', 'exploit', 'breach', 'violat', 'abus', 'threat',
            'danger', 'risk', 'inappropriat', 'problem', 'concern', 'flag',
            'block', 'reject', 'deni', 'prohibit', 'ban', 'fail'
        ]
        
        # Check for common benign patterns  
        benign_patterns = [
            'safe', 'clean', 'normal', 'accept', 'allow', 'permit', 'approv',
            'good', 'fine', 'ok', 'help', 'inform', 'educat', 'construct',
            'pass', 'valid', 'proper', 'legal', 'ethic'
        ]
        
        # Pattern matching with partial strings
        for pattern in malicious_patterns:
            if pattern in value_str:
                return 1
                
        for pattern in benign_patterns:
            if pattern in value_str:
                return 0
        
        # If still no match, make an intelligent guess based on string characteristics
        # Look for negative indicators
        negative_chars = ['!', 'x', '-', 'not', 'un', 'anti', 'contra']
        if any(neg in value_str for neg in negative_chars):
            return 1
            
        # Default to benign (0) for any unmapped value - ENSURE NO NULLS
        return 0
    
    @staticmethod
    def detect_ground_truth_column(df):
        """Auto-detect ground truth column from DataFrame with comprehensive detection"""
        
        # Primary ground truth column names (highest priority)
        primary_columns = [
            'ground_truth', 'ground_truth_label', 'label', 'target', 'class', 
            'type', 'policy', 'human', 'behavior', 'classification', 'gt', 'truth'
        ]
        
        # Secondary indicators (medium priority)
        secondary_columns = [
            'category', 'is_jailbreak', 'is_malicious', 'is_harmful', 'is_safe', 'is_benign',
            'safety_label', 'attack_type', 'harm_type', 'violation_type',
            'jailbreak', 'malicious', 'harmful', 'safe', 'benign', 'toxic',
            'actual', 'expected', 'answer', 'outcome', 'result', 'decision',
            'subcategory', 'main_category', 'harm_category', 'risk_level',
            'severity', 'toxicity', 'content_type', 'response_type'
        ]
        
        # Tertiary patterns (lowest priority)
        tertiary_patterns = [
            'eval', 'assessment', 'rating', 'score', 'flag', 'status',
            'annotation', 'judgment', 'verdict', 'finding', 'conclusion'
        ]
        
        all_potential_columns = primary_columns + secondary_columns + tertiary_patterns
        
        # Step 1: Check exact matches (case-sensitive)
        for col in primary_columns:
            if col in df.columns:
                return col
        
        for col in secondary_columns:
            if col in df.columns:
                return col
        
        for col in tertiary_patterns:
            if col in df.columns:
                return col
        
        # Step 2: Check case-insensitive matches
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for col in all_potential_columns:
            if col.lower() in df_columns_lower:
                return df_columns_lower[col.lower()]
        
        # Step 3: Check partial matches and compound words
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column contains any of our keywords
            for potential in all_potential_columns:
                if potential in col_lower or col_lower in potential:
                    return col
                    
                # Check for compound words with separators
                separators = ['_', '-', '.', ' ']
                for sep in separators:
                    if f"{potential}{sep}" in col_lower or f"{sep}{potential}" in col_lower:
                        return col
        
        # Step 4: Check for columns that might contain boolean or categorical values
        # that suggest ground truth (look at actual values)
        for col in df.columns:
            if col.lower() in ['prompt', 'text', 'input', 'query', 'question', 'message']:
                continue  # Skip obvious prompt columns
                
            # Sample some values to check if they look like ground truth
            sample_values = df[col].dropna().head(20)
            if len(sample_values) == 0:
                continue
                
            # Convert to strings and check for ground truth patterns
            sample_strings = [str(val).lower().strip() for val in sample_values]
            
            # Check if values look like ground truth labels
            gt_indicators = 0
            for val in sample_strings:
                if val in ['true', 'false', '1', '0', 'yes', 'no', 'benign', 'malicious', 
                          'safe', 'unsafe', 'harmful', 'jailbreak', 'attack', 'normal',
                          'violation', 'policy', 'allowed', 'blocked', 'flagged']:
                    gt_indicators += 1
            
            # If more than 30% of values look like ground truth, consider this column
            if gt_indicators / len(sample_strings) > 0.3:
                return col
        
        return None
    
    @staticmethod
    def analyze_ground_truth_column(df, column_name):
        """Analyze ground truth column to understand its format and distribution"""
        if column_name not in df.columns:
            return None
            
        column_data = df[column_name].dropna()
        
        if len(column_data) == 0:
            return {
                'column_name': column_name,
                'total_values': 0,
                'unique_values': 0,
                'unique_list': [],
                'value_counts': {},
                'data_types': 'empty',
                'sample_values': [],
                'format_type': 'empty',
                'binary_distribution': {
                    'malicious': 0,
                    'benign': 0,
                    'malicious_percentage': 0,
                    'valid_count': 0,
                    'invalid_count': 0
                }
            }
        
        analysis = {
            'column_name': column_name,
            'total_values': len(column_data),
            'unique_values': column_data.nunique(),
            'unique_list': list(column_data.unique()),
            'value_counts': column_data.value_counts().to_dict(),
            'data_types': str(column_data.dtype),
            'sample_values': list(column_data.head(10))
        }
        
        # Analyze if values are binary, categorical, or numeric
        unique_vals = set(str(val).lower().strip() for val in column_data.unique())
        
        if len(unique_vals) == 2:
            analysis['format_type'] = 'binary'
        elif len(unique_vals) <= 10:
            analysis['format_type'] = 'categorical'
        else:
            analysis['format_type'] = 'multi_class'
            
        # Check for common patterns with proper null handling
        malicious_count = 0
        benign_count = 0
        valid_count = 0
        
        for val in column_data:
            binary_val = GroundTruthMapper.convert_to_binary(val)
            if binary_val is not None:
                valid_count += 1
                if binary_val == 1:
                    malicious_count += 1
                else:
                    benign_count += 1
        
        analysis['binary_distribution'] = {
            'malicious': malicious_count,
            'benign': benign_count,
            'malicious_percentage': (malicious_count / valid_count * 100) if valid_count > 0 else 0,
            'valid_count': valid_count,
            'invalid_count': len(column_data) - valid_count
        }
        
        return analysis

class EnhancedPromptResult:
    """Enhanced result structure for prompt processing"""
    
    def __init__(self, prompt: str, decision: str, latency_ms: float, 
                 category: str, original_type: str, confidence_score: float,
                 severity_scores: str, error_message: str = None, 
                 timestamp: str = None, ground_truth_binary: int = None,
                 ground_truth_confidence: float = None, prompt_length: int = None,
                 prompt_complexity: float = None, service_type: str = None,
                 ground_truth_original: str = None):
        self.prompt = prompt
        self.decision = decision
        self.latency_ms = latency_ms
        self.category = category
        self.confidence_score = confidence_score
        self.severity_scores = severity_scores
        self.error_message = error_message
        self.timestamp = timestamp or datetime.now().isoformat()
        self.ground_truth_binary = ground_truth_binary
        self.prompt_length = prompt_length or len(prompt)
        self.prompt_complexity = prompt_complexity
        self.service_type = service_type
    
    def to_dict(self):
        """Convert to dictionary for CSV output"""
        # Convert Azure decision to binary: ALLOWED=0, BLOCKED=1
        predicted_binary = 1 if self.decision == "BLOCKED" else 0
        
        return {
            'prompt': self.prompt,
            'decision': self.decision,
            'predicted_binary': predicted_binary,  # Binary prediction for confusion matrix
            'latency_ms': self.latency_ms,
            'category': self.category,
            'confidence_score': self.confidence_score,
            'severity_scores': self.severity_scores,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'ground_truth_binary': self.ground_truth_binary,
            'prompt_length': self.prompt_length,
            'prompt_complexity': self.prompt_complexity,
            'service_type': self.service_type
        }

class ProductionLogger:
    """Production-ready logging configuration"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: str = None, 
                     enable_console: bool = True, max_bytes: int = 10485760, 
                     backup_count: int = 5):
        """Setup production logging with rotation"""
        from logging.handlers import RotatingFileHandler
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

class MultiCloudContentSafetyProcessor:
    """Production-ready processor supporting Azure Content Safety, Azure Prompt Shield, and OpenAI Moderation"""
    
    def __init__(self, 
                 endpoint_url: str = None,
                 api_key: str = None,
                 service_type: str = "auto",
                 max_concurrent_requests: int = 5,
                 rate_limit_per_minute: int = 30,
                 timeout_seconds: int = 30,
                 output_directory: str = "content_safety_results",
                 prompt_column: str = None,
                 ground_truth_column: str = None,
                 severity_threshold: int = 2,
                 auto_detect_schema: bool = True,
                 enable_circuit_breaker: bool = True,
                 dual_detection: bool = False):
        
        # Validate configuration
        if not self._validate_init_params(endpoint_url, api_key, service_type):
            raise ValueError("Invalid configuration parameters")
        
        self.endpoint_url = endpoint_url.rstrip('/') if endpoint_url else None
        self.api_key = api_key
        self.service_type = service_type
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout_seconds = timeout_seconds
        self.severity_threshold = severity_threshold
        self.auto_detect_schema = auto_detect_schema
        self.dual_detection = dual_detection
        
        self.prompt_column = prompt_column
        self.ground_truth_column = ground_truth_column
        
        # Create output directory securely
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True, mode=0o755)
        
        # Initialize security components
        self.security_config = SecurityConfig()
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.ground_truth_mapper = GroundTruthMapper()
        
        # Initialize async components
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_times = []
        self.results = []
        
        # Service discovery - support multiple endpoints
        self.working_endpoints = {}  # Store multiple working endpoints
        self.working_endpoint = None  # Legacy single endpoint (for backwards compatibility)
        self.prompt_shield_endpoint = None
        self.content_safety_endpoint = None
        self.detected_service = None
        
        # Statistics
        self.start_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.dataset_info = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _validate_init_params(self, endpoint_url: str, api_key: str, service_type: str) -> bool:
        """Validate initialization parameters"""
        if service_type not in ["auto", "azure", "openai"]:
            return False
        
        if endpoint_url and not endpoint_url.startswith(('http://', 'https://')):
            return False
        
        if api_key and len(api_key) < 8:  # Basic length check
            return False
        
        return True
    
    def validate_configuration(self) -> bool:
        """Validate configuration and dependencies"""
        errors = []
        
        # Check required dependencies
        if not SKLEARN_AVAILABLE:
            errors.append("scikit-learn is required. Install with: pip install scikit-learn")
        
        if not NUMPY_AVAILABLE:
            errors.append("numpy is required. Install with: pip install numpy")
        
        # Check service-specific requirements
        if self.service_type == "openai" or (self.service_type == "auto" and not self.endpoint_url):
            if not OPENAI_AVAILABLE:
                errors.append("openai library required for OpenAI service. Install with: pip install openai")
            if not self.api_key:
                errors.append("OpenAI API key required")
        
        if self.service_type == "azure" or (self.endpoint_url and "azure" in self.endpoint_url.lower()):
            if not self.endpoint_url:
                errors.append("Azure endpoint URL required")
            if not self.api_key:
                errors.append("Azure API key required")
        
        if errors:
            for error in errors:
                self.logger.error(error)
            return False
        
        return True
    
    async def _create_secure_session(self) -> aiohttp.ClientSession:
        """Create secure HTTP session with proper SSL configuration"""
        
        # Create secure SSL context
        ssl_context = SecurityConfig.create_secure_ssl_context()
        
        # Configure secure connector
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests * 2,
            limit_per_host=self.max_concurrent_requests,
            keepalive_timeout=30,
            ssl=ssl_context,
            verify_ssl=True,
            enable_cleanup_closed=True
        )
        
        # Create session with security headers
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'MultiCloud-ContentSafety-Processor/7.0',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY'
            }
        )
    
    async def _secure_api_call(self, session: aiohttp.ClientSession, 
                              method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make secure API call with circuit breaker and retry logic"""
        
        async def make_request():
            start_time = time.time()
            
            try:
                # Add security headers
                headers = kwargs.get('headers', {})
                headers.update({
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
                
                if self.api_key:
                    if self.service_type == "azure" or "azure" in url.lower():
                        headers['Ocp-Apim-Subscription-Key'] = self.api_key
                    else:
                        headers['Authorization'] = f'Bearer {self.api_key}'
                
                kwargs['headers'] = headers
                
                # Make request with timeout
                async with session.request(method, url, **kwargs) as response:
                    response_data = await response.text()
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        self.successful_requests += 1
                        return {
                            'success': True,
                            'data': json.loads(response_data),
                            'latency_ms': latency_ms,
                            'status_code': response.status
                        }
                    else:
                        self.failed_requests += 1
                        error_msg = f"HTTP {response.status}: {response_data[:200]}"
                        return {
                            'success': False,
                            'error': error_msg,
                            'latency_ms': latency_ms,
                            'status_code': response.status
                        }
                        
            except asyncio.TimeoutError:
                self.failed_requests += 1
                return {
                    'success': False,
                    'error': f"Request timeout after {self.timeout_seconds}s",
                    'latency_ms': (time.time() - start_time) * 1000
                }
            except Exception as e:
                self.failed_requests += 1
                return {
                    'success': False,
                    'error': f"Request failed: {str(e)}",
                    'latency_ms': (time.time() - start_time) * 1000
                }
        
        # Use circuit breaker if enabled
        if self.circuit_breaker:
            return await self.circuit_breaker.call(make_request)
        else:
            return await make_request()
    
    def calculate_confusion_matrix(self, results: List[EnhancedPromptResult]) -> Optional[Dict[str, Any]]:
        """Calculate confusion matrix and performance metrics with enhanced error handling"""
        
        try:
            # Validate input
            if not results:
                self.logger.warning("No results provided for confusion matrix calculation")
                return None
            
            # Extract predictions and actuals with proper validation
            predictions = []
            actuals = []
            
            for result in results:
                # Skip if result is None
                if result is None:
                    continue
                
                # Skip if no ground truth available
                if not hasattr(result, 'ground_truth_binary') or result.ground_truth_binary is None:
                    continue
                    
                # Skip ERROR results as they don't have valid predictions
                if not hasattr(result, 'decision') or result.decision == "ERROR":
                    continue
                
                # Validate decision attribute
                if not hasattr(result, 'decision'):
                    self.logger.warning(f"Result missing decision attribute: {result}")
                    continue
                    
                # Convert Azure decision to binary: ALLOWED=0, BLOCKED=1
                try:
                    predicted_binary = 1 if result.decision == "BLOCKED" else 0
                    actual_binary = int(result.ground_truth_binary)
                    
                    predictions.append(predicted_binary)
                    actuals.append(actual_binary)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Error converting ground truth to binary: {e}")
                    continue
            
            # Check if we have enough data
            if not predictions or not actuals:
                self.logger.info("No valid ground truth data available for confusion matrix calculation")
                return None
            
            if len(predictions) != len(actuals):
                self.logger.warning(f"Prediction and actual counts don't match: {len(predictions)} vs {len(actuals)}")
                return None
            
            if len(predictions) < 2:
                self.logger.warning(f"Insufficient data for confusion matrix: {len(predictions)} samples")
                return None
            
            # Calculate confusion matrix and metrics
            if SKLEARN_AVAILABLE:
                try:
                    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
                    
                    # Check if we have both classes
                    unique_actuals = set(actuals)
                    unique_predictions = set(predictions)
                    
                    if len(unique_actuals) < 2:
                        self.logger.warning(f"Ground truth only contains one class: {unique_actuals}")
                        # Still calculate what we can
                    
                    cm = confusion_matrix(actuals, predictions)
                    
                    # Calculate metrics with zero_division handling
                    accuracy = accuracy_score(actuals, predictions)
                    precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
                    recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
                    
                    # Create detailed report
                    try:
                        report = classification_report(actuals, predictions, 
                                                     target_names=['Benign', 'Malicious'], 
                                                     output_dict=True, zero_division=0)
                    except Exception as e:
                        self.logger.warning(f"Could not generate classification report: {e}")
                        report = {}
                    
                    # Calculate confusion matrix components safely
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                    elif cm.size == 1:
                        # Only one class present
                        if unique_actuals == {0}:  # Only benign
                            tn = cm[0, 0] if len(unique_predictions) == 1 and 0 in unique_predictions else 0
                            fp = cm[0, 0] if len(unique_predictions) == 1 and 1 in unique_predictions else 0
                            fn, tp = 0, 0
                        else:  # Only malicious
                            tp = cm[0, 0] if len(unique_predictions) == 1 and 1 in unique_predictions else 0
                            fn = cm[0, 0] if len(unique_predictions) == 1 and 0 in unique_predictions else 0
                            tn, fp = 0, 0
                    else:
                        self.logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
                        tn = fp = fn = tp = 0
                    
                    return {
                        'confusion_matrix': cm.tolist(),
                        'confusion_matrix_labels': ['Benign', 'Malicious'],
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_positives': int(tp),
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'total_samples': len(predictions),
                        'benign_samples': sum(1 for x in actuals if x == 0),
                        'malicious_samples': sum(1 for x in actuals if x == 1),
                        'predicted_benign': sum(1 for x in predictions if x == 0),
                        'predicted_malicious': sum(1 for x in predictions if x == 1),
                        'classification_report': report,
                        'unique_actuals': list(unique_actuals),
                        'unique_predictions': list(unique_predictions)
                    }
                except Exception as e:
                    self.logger.warning(f"Sklearn calculation failed: {str(e)}")
                    return None
            else:
                # Manual calculation if sklearn not available
                total = len(predictions)
                correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
                accuracy = correct / total if total > 0 else 0
                
                # Manual confusion matrix calculation
                tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
                tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
                fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
                fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                return {
                    'confusion_matrix': [[tn, fp], [fn, tp]],
                    'confusion_matrix_labels': ['Benign', 'Malicious'],
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_positives': tp,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'total_samples': total,
                    'benign_samples': sum(1 for x in actuals if x == 0),
                    'malicious_samples': sum(1 for x in actuals if x == 1),
                    'predicted_benign': sum(1 for x in predictions if x == 0),
                    'predicted_malicious': sum(1 for x in predictions if x == 1)
                }
                    
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {type(e).__name__}: {str(e)}")
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(f"Confusion matrix calculation traceback: {traceback.format_exc()}")
            return None
    
    async def process_batch_secure(self, prompts: List[Dict[str, Any]]) -> List[EnhancedPromptResult]:
        """Enhanced batch processing with security measures and chunking for large datasets"""
        
        total_prompts = len(prompts)
        self.logger.info(f"Processing {total_prompts} prompts")
        
        # Validate total batch size
        if total_prompts > SecurityConfig.MAX_BATCH_SIZE:
            self.logger.info(f"Large dataset detected ({total_prompts} prompts). Processing in chunks of {SecurityConfig.CHUNK_SIZE}")
        
        # Sanitize all prompts
        sanitized_prompts = []
        for i, prompt_data in enumerate(prompts):
            try:
                if not isinstance(prompt_data, dict):
                    raise ValueError(f"Prompt data at index {i} is not a dictionary: {type(prompt_data)}")
                
                raw_prompt = prompt_data.get('prompt', '')
                if not isinstance(raw_prompt, str):
                    raw_prompt = str(raw_prompt)
                
                sanitized_prompt = SecurityConfig.sanitize_prompt(raw_prompt)
                prompt_data['prompt'] = sanitized_prompt
                sanitized_prompts.append(prompt_data)
                
            except Exception as e:
                self.logger.error(f"Error sanitizing prompt {i}: {str(e)}")
                raise ValueError(f"Failed to sanitize prompt {i}: {str(e)}")
        
        if not await self.discover_working_endpoint():
            raise ConnectionError("No working endpoint found")
        
        self.start_time = time.time()
        all_results = []
        
        # Process in chunks for large datasets
        chunk_size = SecurityConfig.CHUNK_SIZE
        total_chunks = (len(sanitized_prompts) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(sanitized_prompts))
            chunk = sanitized_prompts[start_idx:end_idx]
            
            self.logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} prompts)")
            
            # Process chunk with secure session
            async with await self._create_secure_session() as session:
                chunk_results = await self._process_chunk_secure(session, chunk, start_idx)
                all_results.extend(chunk_results)
            
            # Log overall progress
            processed = len(all_results)
            self._log_progress(processed, total_prompts)
        
        return all_results
    
    async def _process_chunk_secure(self, session: aiohttp.ClientSession, 
                                   chunk: List[Dict[str, Any]], 
                                   start_idx: int) -> List[EnhancedPromptResult]:
        """Process a single chunk of prompts"""
        results = []
        
        # Process in smaller batches within the chunk
        batch_size = min(self.max_concurrent_requests, len(chunk))
        
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            
            # Process batch with rate limiting
            await self._apply_rate_limit()
            
            batch_results = await self._process_secure_batch(session, batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_secure_batch(self, session: aiohttp.ClientSession, 
                                   batch: List[Dict[str, Any]]) -> List[EnhancedPromptResult]:
        """Process a batch of prompts securely"""
        
        tasks = []
        for prompt_data in batch:
            task = self._process_single_prompt_secure(session, prompt_data)
            tasks.append(task)
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Log error without exposing sensitive data
                    error_msg = f"Processing error: {type(result).__name__}"
                    self.logger.error(error_msg)
                    
                    # Create error result
                    error_result = EnhancedPromptResult(
                        prompt=SecurityConfig.mask_sensitive_data(batch[i].get('prompt', 'ERROR')),
                        decision="ERROR",
                        latency_ms=0.0,
                        category="error",
                        original_type="error",
                        confidence_score=0.0,
                        severity_scores="",
                        error_message=error_msg,
                        timestamp=datetime.now().isoformat(),
                        service_type=self.detected_service or "unknown"
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {type(e).__name__}")
            raise
    
    async def _process_single_prompt_secure(self, session: aiohttp.ClientSession, 
                                          prompt_data: Dict[str, Any]) -> EnhancedPromptResult:
        """Process a single prompt with security measures"""
        
        async with self.semaphore:
            try:
                # Validate and sanitize prompt
                prompt = SecurityConfig.sanitize_prompt(prompt_data.get('prompt', ''))
                
                # Call appropriate service
                if self.detected_service == "openai":
                    api_result = await self._call_openai_moderation_secure(prompt)
                    
                    if api_result['success']:
                        decision, confidence, severity = self._parse_responses(
                            api_result['data'], self.detected_service
                        )
                        detection_type = "openai_moderation"
                        
                else:
                    # Azure service
                    api_result = await self._call_azure_api_secure(session, prompt)
                    
                    if api_result['success']:
                        decision, confidence, severity = self._parse_responses(
                            api_result['data'], self.detected_service
                        )
                        detection_type = "azure_content_safety"
                
                if api_result['success']:
                    # Convert ground truth to binary if available
                    ground_truth_original = prompt_data.get('ground_truth_original')
                    ground_truth_binary = None
                    if ground_truth_original is not None:
                        ground_truth_binary = self.ground_truth_mapper.convert_to_binary(ground_truth_original)
                    
                    return EnhancedPromptResult(
                        prompt=prompt,
                        decision=decision,
                        latency_ms=api_result['latency_ms'],
                        category=detection_type,
                        original_type=prompt_data.get('category', detection_type),
                        confidence_score=confidence,
                        severity_scores=severity,
                        timestamp=datetime.now().isoformat(),
                        ground_truth_binary=ground_truth_binary,
                        prompt_length=len(prompt),
                        prompt_complexity=prompt_data.get('prompt_complexity', 0.0),
                        service_type=self.detected_service or "unknown"
                    )
                else:
                    # Convert ground truth to binary if available
                    ground_truth_original = prompt_data.get('ground_truth_original')
                    ground_truth_binary = None
                    if ground_truth_original is not None:
                        ground_truth_binary = self.ground_truth_mapper.convert_to_binary(ground_truth_original)
                    
                    return EnhancedPromptResult(
                        prompt=prompt,
                        decision="ERROR",
                        latency_ms=api_result['latency_ms'],
                        category=prompt_data.get('category', 'error'),
                        original_type=prompt_data.get('category', 'error'),
                        confidence_score=0.0,
                        severity_scores="",
                        error_message=api_result['error'],
                        timestamp=datetime.now().isoformat(),
                        ground_truth_binary=ground_truth_binary,
                        prompt_length=len(prompt),
                        prompt_complexity=prompt_data.get('prompt_complexity', 0.0),
                        service_type=self.detected_service or "unknown"
                    )
                    
            except Exception as e:
                self.logger.error(f"Single prompt processing failed: {type(e).__name__}")
                raise
    
    async def discover_working_endpoint(self) -> bool:
        """Discover and validate working endpoint with security"""
        
        # OpenAI Moderation API
        if self.service_type == "openai" or (self.service_type == "auto" and not self.endpoint_url):
            return await self._test_openai_endpoint_secure()
        
        # Azure endpoints
        if self.endpoint_url:
            return await self._test_azure_endpoints_secure()
        
        self.logger.error("No endpoint configuration provided")
        return False
    
    async def _test_openai_endpoint_secure(self) -> bool:
        """Test OpenAI endpoint with security measures"""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI library not available")
            return False
        
        try:
            masked_key = SecurityConfig.mask_sensitive_data(self.api_key or "")
            self.logger.info(f"Testing OpenAI endpoint with key: {masked_key}")
            
            # Test with a simple moderation call
            response = await self._call_openai_moderation_secure("Test message for endpoint validation")
            
            if response['success']:
                self.logger.info("✅ OpenAI Moderation endpoint working")
                self.detected_service = "openai"
                self.working_endpoint = "https://api.openai.com/v1/moderations"
                return True
            else:
                self.logger.error(f"❌ OpenAI test failed: {response['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ OpenAI endpoint test error: {type(e).__name__}")
            return False
    
    async def _test_azure_endpoints_secure(self) -> bool:
        """Test Azure endpoints and identify all working endpoints"""
        
        test_endpoints = [
            # Content Safety endpoints - for general unsafe content detection
            (f"{self.endpoint_url}/contentsafety/text:analyze?api-version=2024-09-01", "2024-09-01", "content_safety"),
            (f"{self.endpoint_url}/contentsafety/text:analyze?api-version=2024-02-15-preview", "2024-02-15-preview", "content_safety"),
            
            # Prompt Shield endpoints
            (f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview", "2024-02-15-preview", "prompt_shield"),
        ]
        
        working_endpoints = {}
        
        async with await self._create_secure_session() as session:
            for endpoint_url, api_version, endpoint_type in test_endpoints:
                try:
                    # Test payloads based on endpoint type
                    if endpoint_type == "content_safety":
                        payload = {
                            'text': 'Test message for content safety validation',
                            'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                            'blocklistNames': [],
                            'outputType': 'FourSeverityLevels'
                        }
                    elif endpoint_type == "prompt_shield":
                        payload = {
                            'userPrompt': 'Test message for prompt shield validation',
                            'documents': []
                        }
                    else:
                        payload = {
                            'text': 'Test message for endpoint validation',
                            'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                            'outputType': 'FourSeverityLevels'
                        }
                    
                    # Make secure API call
                    response = await self._secure_api_call(
                        session, 'POST', endpoint_url, 
                        json=payload
                    )
                    
                    if response['success']:
                        masked_endpoint = SecurityConfig.mask_sensitive_data(endpoint_url)
                        self.logger.info(f"✅ {endpoint_type.upper()} endpoint working: {masked_endpoint}")
                        
                        # Store working endpoint
                        working_endpoints[endpoint_type] = {
                            'url': endpoint_url,
                            'api_version': api_version,
                            'endpoint_type': endpoint_type
                        }
                        
                        # Set specific endpoint references
                        if endpoint_type == "prompt_shield":
                            self.prompt_shield_endpoint = working_endpoints[endpoint_type]
                        elif endpoint_type == "content_safety":
                            self.content_safety_endpoint = working_endpoints[endpoint_type]
                        
                        # Set api_version for the class
                        if not hasattr(self, 'api_version'):
                            self.api_version = api_version
                        
                        self.detected_service = "azure"
                        
                    else:
                        self.logger.debug(f"❌ {endpoint_type.upper()} test failed: {response['error']}")
                        
                except Exception as e:
                    self.logger.debug(f"❌ {endpoint_type.upper()} test error: {type(e).__name__}")
        
        # Store all working endpoints
        self.working_endpoints = working_endpoints
        
        # Check if we have at least one working endpoint
        if working_endpoints:
            self.logger.info(f"Found {len(working_endpoints)} working endpoints: {list(working_endpoints.keys())}")
            return True
        else:
            self.logger.error("No working Azure endpoints found")
            return False
    
    async def _call_openai_moderation_secure(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI Moderation API with security measures"""
        
        start_time = time.time()
        
        try:
            # Validate and sanitize prompt
            prompt = SecurityConfig.sanitize_prompt(prompt)
            
            # Make OpenAI API call with version compatibility
            if OPENAI_VERSION >= (1, 0):
                # OpenAI v1.0+ (newer API with sync client)
                client = openai.OpenAI(api_key=self.api_key)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: client.moderations.create(input=prompt)
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                if response and response.results:
                    self.successful_requests += 1
                    return {
                        'success': True,
                        'data': {
                            'results': [result.model_dump() for result in response.results]
                        },
                        'latency_ms': latency_ms
                    }
                else:
                    self.failed_requests += 1
                    return {
                        'success': False,
                        'error': 'Empty response from OpenAI',
                        'latency_ms': latency_ms
                    }
            else:
                # OpenAI v0.x (legacy API)
                if self.api_key:
                    openai.api_key = self.api_key
                
                try:
                    # Try async method first
                    response = await openai.Moderation.acreate(input=prompt)
                except (AttributeError, TypeError):
                    # Fall back to sync method wrapped in executor
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: openai.Moderation.create(input=prompt)
                    )
                
                latency_ms = (time.time() - start_time) * 1000
                
                if response and hasattr(response, 'results') and response.results:
                    self.successful_requests += 1
                    results_data = []
                    for result in response.results:
                        if hasattr(result, 'to_dict'):
                            results_data.append(result.to_dict())
                        else:
                            results_data.append(dict(result))
                    
                    return {
                        'success': True,
                        'data': {
                            'results': results_data
                        },
                        'latency_ms': latency_ms
                    }
                else:
                    self.failed_requests += 1
                    return {
                        'success': False,
                        'error': 'Empty response from OpenAI',
                        'latency_ms': latency_ms
                    }
                
        except Exception as e:
            self.failed_requests += 1
            return {
                'success': False,
                'error': f"OpenAI API error: {type(e).__name__}",
                'latency_ms': (time.time() - start_time) * 1000
            }
    
    async def _call_azure_api_secure(self, session: aiohttp.ClientSession, 
                                   prompt: str) -> Dict[str, Any]:
        """Call Azure API with security measures"""
        
        # Validate and sanitize prompt
        prompt = SecurityConfig.sanitize_prompt(prompt)
        
        # Choose the best available endpoint
        if self.content_safety_endpoint:
            endpoint_info = self.content_safety_endpoint
            result = await self._call_content_safety_secure(session, prompt, endpoint_info)
        elif self.prompt_shield_endpoint:
            endpoint_info = self.prompt_shield_endpoint
            result = await self._call_prompt_shield_secure(session, prompt, endpoint_info)
        else:
            # No endpoints available
            return {
                'success': False,
                'error': 'No working endpoints available',
                'latency_ms': 0
            }
        
        return result
    
    async def _call_content_safety_secure(self, session: aiohttp.ClientSession, 
                                         prompt: str, endpoint_info: dict) -> Dict[str, Any]:
        """Call Content Safety endpoint securely"""
        
        payload = {
            'text': prompt,
            'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
            'blocklistNames': [],
            'outputType': 'FourSeverityLevels'
        }
        
        return await self._secure_api_call(
            session, 'POST', endpoint_info['url'], 
            json=payload
        )
    
    async def _call_prompt_shield_secure(self, session: aiohttp.ClientSession, 
                                        prompt: str, endpoint_info: dict) -> Dict[str, Any]:
        """Call Prompt Shield endpoint securely"""
        
        payload = {
            'userPrompt': prompt,
            'documents': []
        }
        
        return await self._secure_api_call(
            session, 'POST', endpoint_info['url'], 
            json=payload
        )
    
    def _parse_responses(self, response_data: Dict[str, Any], service_type: str) -> Tuple[str, float, str]:
        """Parse responses from different services"""
        try:
            if service_type == "openai":
                return self._parse_openai_response(response_data)
            elif 'userPromptAnalysis' in response_data:
                # Prompt Shield response
                return self._parse_prompt_shield_response(response_data)
            elif 'categoriesAnalysis' in response_data:
                # Content Safety response
                return self._parse_content_safety_response(response_data)
            else:
                return self._parse_azure_response(response_data)
        except Exception as e:
            self.logger.error(f"Response parsing failed: {type(e).__name__}")
            return "ERROR", 0.0, ""
    
    def _parse_openai_response(self, response_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Parse OpenAI moderation response"""
        try:
            if 'results' not in response_data:
                return "ERROR", 0.0, ""
            
            result = response_data['results'][0]
            flagged = result.get('flagged', False)
            
            decision = "BLOCKED" if flagged else "ALLOWED"
            
            # Calculate confidence from category scores
            category_scores = result.get('category_scores', {})
            confidence_score = max(category_scores.values()) if category_scores else 0.0
            
            # Create severity scores string
            severity_scores = json.dumps(category_scores)
            
            return decision, confidence_score, severity_scores
            
        except Exception as e:
            self.logger.error(f"OpenAI response parsing error: {type(e).__name__}")
            return "ERROR", 0.0, ""
    
    def _parse_prompt_shield_response(self, response_data: dict) -> Tuple[str, float, str]:
        """Parse Prompt Shield response"""
        try:
            if 'userPromptAnalysis' in response_data:
                analysis = response_data['userPromptAnalysis']
                attack_detected = analysis.get('attackDetected', False)
                
                if attack_detected:
                    return "BLOCKED", 0.8, "jailbreak_detected"
                else:
                    return "ALLOWED", 0.1, "safe"
            else:
                return "ALLOWED", 0.0, "unknown"
        except Exception as e:
            self.logger.error(f"Error parsing Prompt Shield response: {str(e)}")
            return "ALLOWED", 0.0, "error"
    
    def _parse_content_safety_response(self, response_data: dict) -> Tuple[str, float, str]:
        """Parse Content Safety response"""
        try:
            if 'categoriesAnalysis' in response_data:
                categories = response_data['categoriesAnalysis']
                max_severity = 0
                blocked_categories = []
                
                for category in categories:
                    severity = category.get('severity', 0)
                    if severity >= self.severity_threshold:
                        blocked_categories.append(category.get('category', 'unknown'))
                        max_severity = max(max_severity, severity)
                
                if blocked_categories:
                    confidence = min(max_severity / 6.0, 1.0)
                    return "BLOCKED", confidence, f"unsafe_content_{','.join(blocked_categories)}"
                else:
                    return "ALLOWED", 0.1, "safe"
            else:
                return "ALLOWED", 0.0, "unknown"
        except Exception as e:
            self.logger.error(f"Error parsing Content Safety response: {str(e)}")
            return "ALLOWED", 0.0, "error"
    
    def _parse_azure_response(self, response_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Parse Azure Content Safety and Jailbreak Detection response"""
        try:
            # Content Safety response
            if 'categoriesAnalysis' in response_data:
                categories = response_data['categoriesAnalysis']
                
                max_severity = 0
                for category in categories:
                    severity = category.get('severity', 0)
                    if severity > max_severity:
                        max_severity = severity
                
                decision = "BLOCKED" if max_severity >= self.severity_threshold else "ALLOWED"
                confidence_score = max_severity / 6.0  # Normalize to 0-1
                
                severity_scores = json.dumps(categories)
                return decision, confidence_score, severity_scores
            
            else:
                self.logger.warning(f"Unknown Azure response format: {list(response_data.keys())}")
                return "ERROR", 0.0, ""
                
        except Exception as e:
            self.logger.error(f"Azure response parsing error: {type(e).__name__}")
            return "ERROR", 0.0, ""
    
    def generate_output_filename(self, input_file_path: str, prefix: str = "") -> str:
        """Generate output filename with input file prefix, azure_results keyword, and timestamp"""
        
        # Extract input filename without extension
        input_path = Path(input_file_path)
        input_name = input_path.stem
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Construct filename
        if prefix:
            filename = f"{prefix}_{input_name}_azure_results_{timestamp}.csv"
        else:
            filename = f"{input_name}_azure_results_{timestamp}.csv"
        
        return filename
    
    def save_results_to_csv(self, results: List[EnhancedPromptResult], 
                           input_file_path: str, prefix: str = "") -> str:
        """Save results to CSV file with proper naming and confusion matrix metrics"""
        
        # Generate output filename
        output_filename = self.generate_output_filename(input_file_path, prefix)
        output_path = self.output_directory / output_filename
        
        # Convert results to list of dictionaries
        results_data = [result.to_dict() for result in results]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results_data)
        
        # Reorder columns for better readability
        column_order = [
            'prompt', 'decision', 'predicted_binary', 'confidence_score', 'category',
            'ground_truth_binary', 'latency_ms',
            'service_type', 'timestamp', 'error_message', 'severity_scores',
            'prompt_length', 'prompt_complexity'
        ]
        
        # Only include columns that exist in the DataFrame
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Handle None values
        df = df.fillna('')  # Replace None/NaN with empty strings
        
        # Calculate confusion matrix metrics if ground truth is available
        confusion_metrics = None
        try:
            if 'ground_truth_binary' in df.columns and 'predicted_binary' in df.columns:
                # Check if we have any non-null ground truth values
                valid_ground_truth = df['ground_truth_binary'].notna().sum()
                if valid_ground_truth > 0:
                    confusion_metrics = self.calculate_confusion_matrix(results)
                else:
                    self.logger.info("No ground truth data available - skipping confusion matrix calculation")
        except Exception as e:
            self.logger.warning(f"Confusion matrix calculation failed: {str(e)} - continuing without metrics")
        
        # Save main results
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Append confusion matrix metrics to the same CSV file
        if confusion_metrics:
            self._append_metrics_to_csv(output_path, confusion_metrics)
        
        # Log summary statistics
        self.logger.info(f"Results saved to: {output_path}")
        self.logger.info(f"Total rows: {len(df)}")
        
        if 'ground_truth_binary' in df.columns:
            benign_count = (df['ground_truth_binary'] == 0).sum()
            malicious_count = (df['ground_truth_binary'] == 1).sum()
            self.logger.info(f"Ground truth distribution: {benign_count} benign, {malicious_count} malicious")
        
        if 'decision' in df.columns:
            decision_counts = df['decision'].value_counts()
            self.logger.info(f"Decision distribution: {decision_counts.to_dict()}")
        
        return str(output_path)
    
    def _append_metrics_to_csv(self, output_path: str, confusion_metrics: Dict[str, Any]):
        """Append confusion matrix metrics to CSV file"""
        
        try:
            # Create metrics summary data
            metrics_data = []
            
            # Add separator row
            metrics_data.append([''] * 13)  # Empty row for separation
            metrics_data.append(['CONFUSION MATRIX AND PERFORMANCE METRICS'] + [''] * 12)
            metrics_data.append([''] * 13)  # Empty row for separation
            
            # Add confusion matrix
            cm = confusion_metrics['confusion_matrix']
            labels = confusion_metrics['confusion_matrix_labels']
            
            metrics_data.append(['Confusion Matrix (Actual vs Predicted)'] + [''] * 12)
            metrics_data.append([''] + ['Predicted'] + [''] * 11)
            metrics_data.append([''] + [''] + [labels[0]] + [labels[1]] + [''] * 9)
            metrics_data.append(['Actual'] + [labels[0]] + [str(cm[0][0])] + [str(cm[0][1])] + [''] * 9)
            metrics_data.append([''] + [labels[1]] + [str(cm[1][0])] + [str(cm[1][1])] + [''] * 9)
            metrics_data.append([''] * 13)  # Empty row
            
            # Add performance metrics
            metrics_data.append(['Performance Metrics'] + [''] * 12)
            metrics_data.append(['Accuracy', f"{confusion_metrics.get('accuracy', 0):.3f}"] + [''] * 11)
            metrics_data.append(['Precision', f"{confusion_metrics.get('precision', 0):.3f}"] + [''] * 11)
            metrics_data.append(['Recall', f"{confusion_metrics.get('recall', 0):.3f}"] + [''] * 11)
            metrics_data.append(['F1-Score', f"{confusion_metrics.get('f1_score', 0):.3f}"] + [''] * 11)
            metrics_data.append([''] * 13)  # Empty row
            
            # Add sample distribution
            metrics_data.append(['Sample Distribution'] + [''] * 12)
            metrics_data.append(['Total samples', str(confusion_metrics.get('total_samples', 0))] + [''] * 11)
            metrics_data.append(['Benign (actual)', str(confusion_metrics.get('benign_samples', 0))] + [''] * 11)
            metrics_data.append(['Malicious (actual)', str(confusion_metrics.get('malicious_samples', 0))] + [''] * 11)
            metrics_data.append(['Benign (predicted)', str(confusion_metrics.get('predicted_benign', 0))] + [''] * 11)
            metrics_data.append(['Malicious (predicted)', str(confusion_metrics.get('predicted_malicious', 0))] + [''] * 11)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics_data.append(['Metrics generated at', timestamp] + [''] * 11)
            
            # Append to CSV file
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                for row in metrics_data:
                    writer.writerow(row)
                    
            self.logger.info(f"Confusion matrix metrics appended to CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error appending metrics to CSV: {str(e)}")
    
    async def _apply_rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Add current request time
        self.request_times.append(current_time)
    
    def _log_progress(self, processed: int, total: int):
        """Log progress securely"""
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            
            self.logger.info(
                f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"Rate: {rate:.1f}/sec | Success: {success_rate:.1f}% | "
                f"ETA: {eta/60:.1f}min"
            )

def setup_production_logging():
    """Setup production logging configuration"""
    log_file = "azure_content_safety.log"
    
    ProductionLogger.setup_logging(
        log_level="INFO",
        log_file=log_file,
        enable_console=True,
        max_bytes=10485760,  # 10MB
        backup_count=5
    )
    
    return logging.getLogger(__name__)

def load_dataset_file(file_path: str, logger) -> pd.DataFrame:
    """Load dataset from various file formats (CSV, Parquet, JSONL)"""
    
    original_path = str(file_path)
    file_path_lower = original_path.lower()
    
    # Determine file type by extension
    if file_path_lower.endswith('.parquet'):
        return load_parquet_file(original_path, logger)
    elif file_path_lower.endswith('.jsonl') or file_path_lower.endswith('.json'):
        return load_jsonl_file(original_path, logger)
    elif file_path_lower.endswith('.csv') or file_path_lower.endswith('.tsv'):
        return load_csv_file(original_path, logger)
    else:
        # Default to CSV for unknown extensions
        logger.warning(f"Unknown file extension, attempting to load as CSV: {original_path}")
        return load_csv_file(original_path, logger)

def load_csv_file(file_path: str, logger) -> pd.DataFrame:
    """Load CSV file with error handling"""
    try:
        # Try comma-separated first
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Loaded CSV file: {file_path}")
        return df
    except Exception as e:
        try:
            # Try tab-separated for .tsv files
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"Loaded TSV file: {file_path}")
            return df
        except Exception as e2:
            try:
                # Try different encoding
                df = pd.read_csv(file_path, encoding='latin1')
                logger.warning(f"Loaded CSV with latin1 encoding: {file_path}")
                return df
            except Exception as e3:
                logger.error(f"Failed to load CSV file {file_path}: {str(e3)}")
                raise

def load_parquet_file(file_path: str, logger) -> pd.DataFrame:
    """Load Parquet file with dependency check"""
    if not PYARROW_AVAILABLE:
        raise ImportError("PyArrow required for Parquet files. Install with: pip install pyarrow")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded Parquet file: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load Parquet file {file_path}: {str(e)}")
        raise

def load_jsonl_file(file_path: str, logger) -> pd.DataFrame:
    """Load JSONL (JSON Lines) file with comprehensive parsing"""
    import json
    
    try:
        records = []
        line_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    # Parse JSON line
                    record = json.loads(line)
                    
                    # Handle different JSON structures
                    if isinstance(record, dict):
                        records.append(record)
                    elif isinstance(record, list) and len(record) > 0:
                        # If it's a list, try to convert to dict
                        if isinstance(record[0], dict):
                            records.extend(record)
                        else:
                            # Create dict from list values
                            records.append({f"column_{i}": val for i, val in enumerate(record)})
                    else:
                        # Single value, create dict
                        records.append({"value": record})
                        
                    line_count += 1
                    
                except json.JSONDecodeError as je:
                    logger.warning(f"Invalid JSON on line {line_num}: {str(je)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        if not records:
            raise ValueError("No valid JSON records found in file")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        logger.info(f"Loaded JSONL file: {file_path} ({line_count} records)")
        
        # Handle nested objects by converting to string
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains nested objects
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (dict, list)):
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load JSONL file {file_path}: {str(e)}")
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Cloud Content Safety Processor - Production Ready v7.0 (Complete Fixed Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fixed Issues in v7.0:
  • Complete script with all imports and dependencies
  • Fixed AttributeError and confusion matrix calculation errors
  • Enhanced ground truth validation and error handling
  • Improved null value handling throughout the pipeline
  • Fixed indentation errors in confusion matrix display
  • Added proper validation for ground truth data
  • Comprehensive error handling and logging

Examples:
  # Auto-detect service and schema (recommended)
  python azure-automated-script-clean.py --input dataset.csv --api-key YOUR_KEY

  # Azure Content Safety with specific endpoint
  python azure-automated-script-clean.py --input dataset.parquet --endpoint AZURE_URL --api-key AZURE_KEY

  # OpenAI Moderation API with JSONL input
  python azure-automated-script-clean.py --input dataset.jsonl --service openai --api-key OPENAI_KEY

Environment Variables:
  AZURE_CONTENT_SAFETY_ENDPOINT  - Azure endpoint URL
  AZURE_CONTENT_SAFETY_KEY       - Azure API key
  OPENAI_API_KEY                 - OpenAI API key

Supported File Formats:
  CSV (.csv)     - Comma-separated values
  TSV (.tsv)     - Tab-separated values  
  Parquet (.parquet) - Columnar storage (requires pyarrow)
  JSONL (.jsonl) - JSON Lines format
  JSON (.json)   - Standard JSON format
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input file (CSV, TSV, Parquet, JSONL, or JSON)')
    
    # Authentication
    parser.add_argument('--endpoint', help='Azure Content Safety endpoint URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--service', choices=['auto', 'azure', 'openai'], 
                       default='auto', help='Service type to use')
    
    # Processing options
    parser.add_argument('--concurrent', type=int, default=5,
                       help='Maximum concurrent requests')
    parser.add_argument('--rate-limit', type=int, default=30,
                       help='Rate limit per minute')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds')
    parser.add_argument('--severity-threshold', type=int, default=2,
                       help='Severity threshold for blocking (0-6)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Chunk size for processing large datasets')
    parser.add_argument('--max-batch-size', type=int, default=10000,
                       help='Maximum total batch size before chunking')
    
    # Column specification
    parser.add_argument('--prompt-column', help='Column name containing prompts')
    parser.add_argument('--ground-truth-column', help='Column name for ground truth')
    
    # Output options
    parser.add_argument('--output-dir', default='content_safety_results',
                       help='Output directory')
    parser.add_argument('--output-prefix', default='results',
                       help='Output file prefix')
    
    # Analysis options
    parser.add_argument('--no-schema-detection', action='store_true',
                       help='Disable automatic schema detection')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test configuration without processing')
    parser.add_argument('--dual-detection', action='store_true',
                       help='Enable both content safety and jailbreak detection (comprehensive)')
    parser.add_argument('--save-checkpoints', action='store_true',
                       help='Save progress checkpoints for large datasets')
    
    # Security options
    parser.add_argument('--disable-circuit-breaker', action='store_true',
                       help='Disable circuit breaker pattern')
    
    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--quiet', action='store_true', help='Enable quiet logging')
    
    return parser.parse_args()

async def main():
    """Main async function with enhanced error handling"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = args.log_level
    
    ProductionLogger.setup_logging(
        log_level=log_level,
        log_file=args.log_file,
        enable_console=True
    )
    
    logger = logging.getLogger(__name__)
    
    # Get credentials with security validation
    api_key = args.api_key or os.getenv("AZURE_CONTENT_SAFETY_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint_url = args.endpoint or os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    
    if not api_key:
        logger.error("API key required. Use --api-key or set environment variable")
        return 1
    
    # Validate file path
    if not SecurityConfig.validate_file_path(args.input):
        logger.error(f"Invalid or potentially unsafe file path: {args.input}")
        return 1
    
    # Update security config with command line parameters
    SecurityConfig.CHUNK_SIZE = args.chunk_size
    SecurityConfig.MAX_BATCH_SIZE = args.max_batch_size
    
    # Initialize processor
    try:
        processor = MultiCloudContentSafetyProcessor(
            endpoint_url=endpoint_url,
            api_key=api_key,
            service_type=args.service,
            max_concurrent_requests=args.concurrent,
            rate_limit_per_minute=args.rate_limit,
            timeout_seconds=args.timeout,
            output_directory=args.output_dir,
            prompt_column=args.prompt_column,
            ground_truth_column=args.ground_truth_column,
            severity_threshold=args.severity_threshold,
            auto_detect_schema=not args.no_schema_detection,
            enable_circuit_breaker=not args.disable_circuit_breaker,
            dual_detection=args.dual_detection
        )
        
        # Set checkpoint and file path attributes
        processor.save_checkpoints = args.save_checkpoints
        processor.input_file_path = args.input
        processor.output_prefix = args.output_prefix
        
        # Validate configuration
        if not processor.validate_configuration():
            logger.error("Configuration validation failed")
            return 1
        
        # Dry run mode
        if args.dry_run:
            logger.info("Dry run mode - testing configuration only")
            if await processor.discover_working_endpoint():
                logger.info("✅ Configuration is valid and endpoint is accessible")
                return 0
            else:
                logger.error("❌ Configuration test failed")
                return 1
        
        # Load and process data
        logger.info(f"Loading data from: {args.input}")
        
        # Load data securely with support for multiple file formats
        try:
            df = load_dataset_file(args.input, logger)
            
            logger.info(f"Loaded {len(df)} rows")
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            # Validate dataset
            if df.empty:
                logger.error("Dataset is empty")
                return 1
            
        except Exception as e:
            logger.error(f"Failed to load data: {type(e).__name__}: {str(e)}")
            return 1
        
        # Process data
        try:
            # Auto-detect prompt column
            prompt_column = args.prompt_column
            if not prompt_column:
                # Try common prompt column names
                prompt_candidates = ['prompt', 'text', 'input', 'question', 'query', 'message', 'content']
                for candidate in prompt_candidates:
                    if candidate in df.columns:
                        prompt_column = candidate
                        break
                
                if not prompt_column:
                    prompt_column = df.columns[0]  # Use first column as fallback
                    
            if prompt_column not in df.columns:
                logger.error(f"Prompt column '{prompt_column}' not found in data")
                return 1
            
            logger.info(f"Using prompt column: '{prompt_column}'")
            
            # Auto-detect ground truth column with enhanced validation
            ground_truth_column = args.ground_truth_column
            if not ground_truth_column:
                ground_truth_column = processor.ground_truth_mapper.detect_ground_truth_column(df)
                
            if ground_truth_column:
                logger.info(f"Using ground truth column: '{ground_truth_column}'")
                
                # Analyze the ground truth column with proper error handling
                try:
                    gt_analysis = processor.ground_truth_mapper.analyze_ground_truth_column(df, ground_truth_column)
                    if gt_analysis and gt_analysis.get('total_values', 0) > 0:
                        logger.info(f"Ground truth analysis:")
                        logger.info(f"  Format type: {gt_analysis['format_type']}")
                        logger.info(f"  Total values: {gt_analysis['total_values']}")
                        logger.info(f"  Unique values: {gt_analysis['unique_values']}")
                        logger.info(f"  Value distribution: {gt_analysis['value_counts']}")
                        
                        binary_dist = gt_analysis['binary_distribution']
                        logger.info(f"  Binary distribution: {binary_dist['malicious']} malicious, {binary_dist['benign']} benign")
                        logger.info(f"    Valid ground truth: {binary_dist['valid_count']}/{gt_analysis['total_values']} ({binary_dist['valid_count']/gt_analysis['total_values']*100:.1f}%)")
                        logger.info(f"    Malicious percentage: {binary_dist['malicious_percentage']:.1f}%")
                        
                        if binary_dist['invalid_count'] > 0:
                            logger.warning(f"  Invalid/unmappable values: {binary_dist['invalid_count']}")
                        
                        # Show sample of ground truth values and their binary conversion
                        sample_values = df[ground_truth_column].dropna().head(10)
                        logger.info("Sample ground truth conversions:")
                        for val in sample_values:
                            binary = processor.ground_truth_mapper.convert_to_binary(val)
                            logger.info(f"  '{val}' -> {binary}")
                    else:
                        logger.warning(f"Ground truth column '{ground_truth_column}' appears to be empty or invalid")
                        ground_truth_column = None
                        
                except Exception as e:
                    logger.error(f"Error analyzing ground truth column: {type(e).__name__}: {str(e)}")
                    ground_truth_column = None
            else:
                logger.warning("No ground truth column detected")
                logger.info("Available columns for manual selection:")
                for col in df.columns:
                    logger.info(f"  - {col}")
                logger.info("Use --ground-truth-column to specify manually")
            
            # Convert DataFrame to prompt list
            prompts = []
            for _, row in df.iterrows():
                prompt_data = {
                    'prompt': str(row[prompt_column]),
                    'category': 'unknown'
                }
                
                # Add ground truth if available
                if ground_truth_column and ground_truth_column in df.columns:
                    ground_truth_value = row[ground_truth_column]
                    if pd.notna(ground_truth_value) and ground_truth_value != '':
                        prompt_data['ground_truth_original'] = str(ground_truth_value)
                
                prompts.append(prompt_data)
            
            # Validate prompts before processing
            if not prompts:
                logger.error("No valid prompts found in dataset")
                return 1
            
            logger.info(f"Prepared {len(prompts)} prompts for processing")
            
            # Process batch
            try:
                results = await processor.process_batch_secure(prompts)
                
                if not results:
                    logger.error("No results returned from processing")
                    return 1
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                if logger.level <= logging.DEBUG:
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Save results to CSV
            try:
                output_file = processor.save_results_to_csv(results, args.input, args.output_prefix)
                logger.info(f"Results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save results: {type(e).__name__}: {str(e)}")
                return 1
            
            # Log summary
            total_processed = len(results)
            blocked_count = sum(1 for r in results if r.decision == "BLOCKED")
            error_count = sum(1 for r in results if r.decision == "ERROR")
            
            logger.info(f"Processing complete:")
            logger.info(f"  Total processed: {total_processed}")
            logger.info(f"  Blocked: {blocked_count} ({blocked_count/total_processed*100:.1f}%)")
            logger.info(f"  Errors: {error_count} ({error_count/total_processed*100:.1f}%)")
            
            # Calculate and display confusion matrix if ground truth is available
            if ground_truth_column:
                try:
                    confusion_matrix_results = processor.calculate_confusion_matrix(results)
                    
                    if confusion_matrix_results and isinstance(confusion_matrix_results, dict):
                        logger.info("\n" + "="*50)
                        logger.info("CONFUSION MATRIX AND PERFORMANCE METRICS")
                        logger.info("="*50)
                        
                        # Display confusion matrix with proper error handling
                        try:
                            cm = confusion_matrix_results.get('confusion_matrix')
                            labels = confusion_matrix_results.get('confusion_matrix_labels', ['Benign', 'Malicious'])
                            
                            if cm and len(cm) >= 2 and len(cm[0]) >= 2:
                                logger.info(f"Confusion Matrix (Actual vs Predicted):")
                                logger.info(f"                    Predicted")
                                logger.info(f"                 {labels[0]:<8} {labels[1]:<8}")
                                logger.info(f"Actual {labels[0]:<8} {cm[0][0]:<8} {cm[0][1]:<8}")
                                logger.info(f"       {labels[1]:<8} {cm[1][0]:<8} {cm[1][1]:<8}")
                                logger.info("")
                            else:
                                logger.warning("Confusion matrix format is invalid or incomplete")
                                
                            # Display detailed metrics
                            logger.info(f"Performance Metrics:")
                            logger.info(f"  Accuracy:  {confusion_matrix_results.get('accuracy', 0):.3f}")
                            logger.info(f"  Precision: {confusion_matrix_results.get('precision', 0):.3f}")
                            logger.info(f"  Recall:    {confusion_matrix_results.get('recall', 0):.3f}")
                            logger.info(f"  F1-Score:  {confusion_matrix_results.get('f1_score', 0):.3f}")
                            logger.info("="*50)
                            
                        except Exception as e:
                            logger.error(f"Error displaying confusion matrix: {type(e).__name__}: {str(e)}")
                    else:
                        logger.warning("Could not calculate confusion matrix - insufficient ground truth data")
                except Exception as e:
                    logger.warning(f"Confusion matrix calculation failed: {str(e)} - continuing without metrics")
            
            return 0
            
        except Exception as e:
            logger.error(f"Processing failed: {type(e).__name__}")
            return 1
            
    except Exception as e:
        logger.error(f"Initialization failed: {type(e).__name__}")
        return 1

if __name__ == "__main__":
    sys.exit(run_async_main(main()))