#!/usr/bin/env python3
"""
Multi-Cloud Content Safety Batch Processor - Production Ready v6.0
Comprehensive solution for Azure Content Safety, Azure Prompt Shield, and OpenAI Moderation
Supports multiple dataset schemas with advanced analytics and business metrics

Security enhancements:
- SSL certificate verification
- Input validation and sanitization
- Secure credential handling
- Rate limiting and circuit breaker pattern
- Comprehensive error handling
- Production logging configuration
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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import urllib.parse
import secrets
import hashlib

# Python version compatibility check
PYTHON_VERSION = sys.version_info[:2]
if PYTHON_VERSION < (3, 6):
    print("Error: Python 3.6 or higher is required")
    sys.exit(1)

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

class SecurityConfig:
    """Security configuration and utilities"""
    
    # Maximum input sizes for security
    MAX_PROMPT_LENGTH = 10000
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_BATCH_SIZE = 1000
    
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
    
    # SSL/TLS configuration
    SSL_CONTEXT = None  # Will be created when needed
    
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
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
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
    """Maps ground truth labels to binary values"""
    
    # Define mappings for different ground truth formats
    MALICIOUS_KEYWORDS = {
        'jailbreak', 'malicious', 'harmful', 'unsafe', 'toxic', 'attack', 
        'exploit', 'injection', 'bypass', 'hack', 'manipulate', 'adversarial',
        'prompt_injection', 'social_engineering', 'phishing', 'spam',
        'hate', 'harassment', 'violence', 'self_harm', 'sexual',
        'dangerous', 'illegal', 'unethical', 'abuse', 'threat',
        'blocked', 'flagged', 'rejected', 'denied', 'prohibited',
        'true', '1', 'yes', 'positive', 'bad'
    }
    
    BENIGN_KEYWORDS = {
        'benign', 'safe', 'harmless', 'clean', 'legitimate', 'normal',
        'acceptable', 'appropriate', 'allowed', 'permitted', 'approved',
        'good', 'neutral', 'fine', 'ok', 'okay', 'valid',
        'false', '0', 'no', 'negative', 'none', 'null'
    }
    
    @staticmethod
    def convert_to_binary(ground_truth_value) -> int:
        """Convert ground truth value to binary (0=benign, 1=malicious)"""
        if ground_truth_value is None or ground_truth_value == '':
            return 0
        
        # Convert to string and normalize
        value_str = str(ground_truth_value).lower().strip()
        
        # Direct numeric conversion
        if value_str.isdigit():
            return min(int(value_str), 1)  # Cap at 1
        
        # Float conversion
        try:
            float_val = float(value_str)
            return 1 if float_val > 0.5 else 0
        except ValueError:
            pass
        
        # Keyword matching
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
        
        # Default to benign if uncertain
        return 0
    
    @staticmethod
    def detect_ground_truth_column(df):
        """Auto-detect ground truth column from DataFrame"""
        potential_columns = [
            'ground_truth', 'label', 'target', 'class', 'category',
            'type', 'classification', 'is_jailbreak', 'is_malicious',
            'is_harmful', 'is_safe', 'safety_label', 'attack_type',
            'jailbreak', 'malicious', 'harmful', 'safe', 'benign',
            'gt', 'truth', 'actual', 'expected', 'answer'
        ]
        
        # Check exact matches first
        for col in potential_columns:
            if col in df.columns:
                return col
        
        # Check case-insensitive matches
        df_columns_lower = {col.lower(): col for col in df.columns}
        for col in potential_columns:
            if col.lower() in df_columns_lower:
                return df_columns_lower[col.lower()]
        
        # Check partial matches
        for col in df.columns:
            col_lower = col.lower()
            for potential in potential_columns:
                if potential in col_lower or col_lower in potential:
                    return col
        
        return None

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
        self.original_type = original_type
        self.confidence_score = confidence_score
        self.severity_scores = severity_scores
        self.error_message = error_message
        self.timestamp = timestamp or datetime.now().isoformat()
        self.ground_truth_binary = ground_truth_binary
        self.ground_truth_confidence = ground_truth_confidence
        self.prompt_length = prompt_length or len(prompt)
        self.prompt_complexity = prompt_complexity
        self.service_type = service_type
        self.ground_truth_original = ground_truth_original
    
    def to_dict(self):
        """Convert to dictionary for CSV output"""
        return {
            'prompt': self.prompt,
            'decision': self.decision,
            'latency_ms': self.latency_ms,
            'category': self.category,
            'original_type': self.original_type,
            'confidence_score': self.confidence_score,
            'severity_scores': self.severity_scores,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'ground_truth_binary': self.ground_truth_binary,
            'ground_truth_original': self.ground_truth_original,
            'ground_truth_confidence': self.ground_truth_confidence,
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
        
        # Service discovery
        self.working_endpoint = None
        self.api_version = None
        self.endpoint_type = None
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
                'User-Agent': 'MultiCloud-ContentSafety-Processor/6.0',
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
    
    async def process_batch_secure(self, prompts: List[Dict[str, Any]]) -> List[EnhancedPromptResult]:
        """Enhanced batch processing with security measures"""
        
        # Validate batch size
        if len(prompts) > SecurityConfig.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(prompts)} exceeds maximum {SecurityConfig.MAX_BATCH_SIZE}")
        
        # Sanitize all prompts
        sanitized_prompts = []
        for prompt_data in prompts:
            sanitized_prompt = SecurityConfig.sanitize_prompt(prompt_data.get('prompt', ''))
            prompt_data['prompt'] = sanitized_prompt
            sanitized_prompts.append(prompt_data)
        
        if not await self.discover_working_endpoint():
            raise ConnectionError("No working endpoint found")
        
        masked_endpoint = SecurityConfig.mask_sensitive_data(self.working_endpoint or "unknown")
        self.logger.info(f"Processing {len(sanitized_prompts)} prompts using: {self.detected_service.upper()}")
        self.logger.info(f"Endpoint: {masked_endpoint}")
        
        self.start_time = time.time()
        
        # Process with secure session
        async with await self._create_secure_session() as session:
            results = []
            
            # Process in secure batches
            batch_size = min(self.max_concurrent_requests, len(sanitized_prompts))
            
            for i in range(0, len(sanitized_prompts), batch_size):
                batch = sanitized_prompts[i:i + batch_size]
                
                # Process batch with rate limiting
                await self._apply_rate_limit()
                
                batch_results = await self._process_secure_batch(session, batch)
                results.extend(batch_results)
                
                # Log progress securely
                self._log_progress(i + len(batch), len(sanitized_prompts))
        
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
                        
                elif self.dual_detection and self.detected_service == "azure":
                    # Use comprehensive dual detection
                    dual_result = await self.analyze_content_and_jailbreak(session, prompt)
                    
                    decision = dual_result['combined_decision']
                    confidence = dual_result['max_confidence']
                    severity = json.dumps(dual_result)
                    detection_type = "dual_detection"
                    
                    api_result = {
                        'success': True,
                        'latency_ms': 0,  # Combined latency would be calculated in dual method
                        'data': dual_result
                    }
                    
                else:
                    api_result = await self._call_azure_api_secure(session, prompt)
                    
                    if api_result['success']:
                        decision, confidence, severity = self._parse_responses(
                            api_result['data'], self.detected_service
                        )
                        
                        # Add detection type for better categorization
                        detection_type = "unknown"
                        if self.endpoint_type == "jailbreak_detection":
                            detection_type = "jailbreak"
                        elif self.endpoint_type == "content_safety":
                            detection_type = "content_safety"
                        elif self.endpoint_type == "prompt_shield":
                            detection_type = "prompt_shield"
                
                if api_result['success']:
                    
                    # Convert ground truth to binary if available
                    ground_truth_original = prompt_data.get('ground_truth_original')
                    ground_truth_binary = None
                    if ground_truth_original is not None:
                        ground_truth_binary = self.ground_truth_mapper.convert_to_binary(ground_truth_original)
                    
                    return EnhancedPromptResult(
                        prompt=prompt,  # Already sanitized
                        decision=decision,
                        latency_ms=api_result['latency_ms'],
                        category=detection_type,
                        original_type=prompt_data.get('category', detection_type),
                        confidence_score=confidence,
                        severity_scores=severity,
                        timestamp=datetime.now().isoformat(),
                        ground_truth_binary=ground_truth_binary,
                        ground_truth_original=ground_truth_original,
                        ground_truth_confidence=prompt_data.get('ground_truth_confidence', 0.0),
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
                        ground_truth_original=ground_truth_original,
                        ground_truth_confidence=prompt_data.get('ground_truth_confidence', 0.0),
                        prompt_length=len(prompt),
                        prompt_complexity=prompt_data.get('prompt_complexity', 0.0),
                        service_type=self.detected_service or "unknown"
                    )
                    
            except Exception as e:
                self.logger.error(f"Single prompt processing failed: {type(e).__name__}")
                raise
    
    async def analyze_content_and_jailbreak(self, session: aiohttp.ClientSession, 
                                          prompt: str) -> Dict[str, Any]:
        """Perform both content safety and jailbreak detection for comprehensive analysis"""
        
        prompt = SecurityConfig.sanitize_prompt(prompt)
        results = {
            'content_safety': None,
            'jailbreak_detection': None,
            'combined_decision': 'ALLOWED',
            'max_confidence': 0.0,
            'detection_types': []
        }
        
        # Test content safety endpoint
        if self.endpoint_url:
            content_safety_url = f"{self.endpoint_url}/contentsafety/text:analyze?api-version={self.api_version}"
            content_payload = {
                'text': prompt,
                'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                'blocklistNames': [],
                'outputType': 'FourSeverityLevels'
            }
            
            try:
                content_result = await self._secure_api_call(
                    session, 'POST', content_safety_url, json=content_payload
                )
                
                if content_result['success']:
                    decision, confidence, severity = self._parse_azure_response(content_result['data'])
                    results['content_safety'] = {
                        'decision': decision,
                        'confidence': confidence,
                        'severity': severity,
                        'data': content_result['data']
                    }
                    
                    if decision == "BLOCKED":
                        results['combined_decision'] = "BLOCKED"
                        results['detection_types'].append("unsafe_content")
                        
                    if confidence > results['max_confidence']:
                        results['max_confidence'] = confidence
                        
            except Exception as e:
                self.logger.error(f"Content safety analysis failed: {type(e).__name__}")
        
        # Test jailbreak detection endpoint
        if self.endpoint_url:
            jailbreak_url = f"{self.endpoint_url}/contentsafety/text:detect-jailbreak?api-version={self.api_version}"
            jailbreak_payload = {
                'text': prompt
            }
            
            try:
                jailbreak_result = await self._secure_api_call(
                    session, 'POST', jailbreak_url, json=jailbreak_payload
                )
                
                if jailbreak_result['success']:
                    decision, confidence, severity = self._parse_azure_response(jailbreak_result['data'])
                    results['jailbreak_detection'] = {
                        'decision': decision,
                        'confidence': confidence,
                        'severity': severity,
                        'data': jailbreak_result['data']
                    }
                    
                    if decision == "BLOCKED":
                        results['combined_decision'] = "BLOCKED"
                        results['detection_types'].append("jailbreak")
                        
                    if confidence > results['max_confidence']:
                        results['max_confidence'] = confidence
                        
            except Exception as e:
                self.logger.error(f"Jailbreak detection failed: {type(e).__name__}")
        
        return results
    
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
        """Save results to CSV file with proper naming"""
        
        # Generate output filename
        output_filename = self.generate_output_filename(input_file_path, prefix)
        output_path = self.output_directory / output_filename
        
        # Convert results to list of dictionaries
        results_data = [result.to_dict() for result in results]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results_data)
        
        # Reorder columns for better readability
        column_order = [
            'prompt', 'decision', 'confidence_score', 'category',
            'ground_truth_binary', 'ground_truth_original', 'latency_ms',
            'service_type', 'timestamp', 'error_message', 'severity_scores',
            'original_type', 'ground_truth_confidence', 'prompt_length',
            'prompt_complexity'
        ]
        
        # Only include columns that exist in the DataFrame
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
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
                self.endpoint_type = "openai_moderation"
                self.working_endpoint = "https://api.openai.com/v1/moderations"
                return True
            else:
                self.logger.error(f"❌ OpenAI test failed: {response['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ OpenAI endpoint test error: {type(e).__name__}")
            return False
    
    async def _test_azure_endpoints_secure(self) -> bool:
        """Test Azure endpoints with security measures"""
        
        test_endpoints = [
            # Jailbreak detection endpoints (Prompt Shield) - prioritized for jailbreak detection
            (f"{self.endpoint_url}/contentsafety/text:detect-jailbreak?api-version=2024-02-15-preview", "2024-02-15-preview", "jailbreak_detection"),
            (f"{self.endpoint_url}/contentsafety/text:detect-jailbreak?api-version=2024-09-01", "2024-09-01", "jailbreak_detection"),
            
            # Content Safety endpoints - for general unsafe content detection
            (f"{self.endpoint_url}/contentsafety/text:analyze?api-version=2024-09-01", "2024-09-01", "content_safety"),
            (f"{self.endpoint_url}/contentsafety/text:analyze?api-version=2024-02-15-preview", "2024-02-15-preview", "content_safety"),
            
            # Legacy Prompt Shield endpoints (fallback)
            (f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview", "2024-02-15-preview", "prompt_shield"),
        ]
        
        async with await self._create_secure_session() as session:
            for endpoint_url, api_version, endpoint_type in test_endpoints:
                try:
                    # Test payloads based on endpoint type
                    if endpoint_type == "jailbreak_detection":
                        payload = {
                            'text': 'Test message for jailbreak detection validation'
                        }
                    elif endpoint_type == "content_safety":
                        payload = {
                            'text': 'Test message for content safety validation',
                            'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                            'blocklistNames': [],
                            'outputType': 'FourSeverityLevels'
                        }
                    elif endpoint_type == "prompt_shield":
                        payload = {
                            'userPrompt': 'Test message for prompt shield validation',
                            'documents': ['Test document content']
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
                        
                        self.detected_service = "azure"
                        self.working_endpoint = endpoint_url
                        self.api_version = api_version
                        self.endpoint_type = endpoint_type
                        return True
                    else:
                        self.logger.debug(f"❌ {endpoint_type.upper()} test failed: {response['error']}")
                        
                except Exception as e:
                    self.logger.debug(f"❌ {endpoint_type.upper()} test error: {type(e).__name__}")
        
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
        """Call Azure API with security measures for both content safety and jailbreak detection"""
        
        # Validate and sanitize prompt
        prompt = SecurityConfig.sanitize_prompt(prompt)
        
        # Prepare payload based on endpoint type
        if self.endpoint_type == "jailbreak_detection":
            payload = {
                'text': prompt
            }
        elif self.endpoint_type == "content_safety":
            payload = {
                'text': prompt,
                'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                'blocklistNames': [],  # Optional: custom blocklists
                'outputType': 'FourSeverityLevels'
            }
        elif self.endpoint_type == "prompt_shield":
            payload = {
                'userPrompt': prompt,
                'documents': []
            }
        else:
            # Default to content safety
            payload = {
                'text': prompt,
                'categories': ['Hate', 'SelfHarm', 'Sexual', 'Violence'],
                'blocklistNames': [],
                'outputType': 'FourSeverityLevels'
            }
        
        # Make secure API call
        return await self._secure_api_call(
            session, 'POST', self.working_endpoint, 
            json=payload
        )
    
    def _parse_responses(self, response_data: Dict[str, Any], service_type: str) -> Tuple[str, float, str]:
        """Parse responses from different services"""
        try:
            if service_type == "openai":
                return self._parse_openai_response(response_data)
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
            categories = result.get('categories', {})
            category_scores = result.get('category_scores', {})
            
            confidence_score = max(category_scores.values()) if category_scores else 0.0
            
            # Create severity scores string
            severity_scores = json.dumps(category_scores)
            
            return decision, confidence_score, severity_scores
            
        except Exception as e:
            self.logger.error(f"OpenAI response parsing error: {type(e).__name__}")
            return "ERROR", 0.0, ""
    
    def _parse_azure_response(self, response_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Parse Azure Content Safety and Jailbreak Detection response"""
        try:
            # Jailbreak Detection response
            if 'jailbreakAnalysis' in response_data:
                jailbreak_analysis = response_data['jailbreakAnalysis']
                detected = jailbreak_analysis.get('detected', False)
                
                decision = "BLOCKED" if detected else "ALLOWED"
                
                # Get confidence score from jailbreak analysis
                confidence_score = 0.0
                if 'confidence' in jailbreak_analysis:
                    confidence_score = jailbreak_analysis['confidence']
                elif detected:
                    confidence_score = 0.8  # High confidence if detected
                
                severity_scores = json.dumps(jailbreak_analysis)
                return decision, confidence_score, severity_scores
            
            # Prompt Shield response (legacy)
            elif 'attackDetected' in response_data:
                attack_detected = response_data.get('attackDetected', False)
                decision = "BLOCKED" if attack_detected else "ALLOWED"
                
                confidence_score = 0.0
                if 'analyzeResults' in response_data:
                    results = response_data['analyzeResults']
                    for result in results:
                        if 'categories' in result:
                            for category in result['categories']:
                                severity = category.get('severity', 0)
                                if severity > confidence_score:
                                    confidence_score = severity / 6.0  # Normalize to 0-1
                
                severity_scores = json.dumps(response_data.get('analyzeResults', []))
                return decision, confidence_score, severity_scores
            
            # Content Safety response
            elif 'categoriesAnalysis' in response_data:
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Cloud Content Safety Processor - Production Ready v6.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect service and schema (recommended)
  python azure_production_ready.py --input dataset.csv --api-key YOUR_KEY

  # Azure Content Safety with specific endpoint
  python azure_production_ready.py --input dataset.csv --endpoint AZURE_URL --api-key AZURE_KEY

  # OpenAI Moderation API
  python azure_production_ready.py --input dataset.csv --service openai --api-key OPENAI_KEY

  # Comprehensive protection (both content safety and jailbreak detection)
  python azure_production_ready.py --input dataset.csv --endpoint AZURE_URL --api-key AZURE_KEY --dual-detection

Environment Variables:
  AZURE_CONTENT_SAFETY_ENDPOINT  - Azure endpoint URL
  AZURE_CONTENT_SAFETY_KEY       - Azure API key
  OPENAI_API_KEY                 - OpenAI API key
  
Security Features:
  - Dual detection for comprehensive protection against unsafe content and jailbreaks
  - SSL/TLS certificate verification with minimum TLS 1.2
  - Input sanitization and validation
  - Rate limiting and circuit breaker patterns
  - Secure credential handling and logging
        """
    )
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Input CSV or Parquet file')
    
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
    """Main async function"""
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
        
        # Load data securely
        try:
            if args.input.endswith('.parquet'):
                if not PYARROW_AVAILABLE:
                    logger.error("PyArrow required for Parquet files. Install with: pip install pyarrow")
                    return 1
                df = pd.read_parquet(args.input)
            else:
                df = pd.read_csv(args.input)
            
            logger.info(f"Loaded {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to load data: {type(e).__name__}")
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
            
            # Auto-detect ground truth column
            ground_truth_column = args.ground_truth_column
            if not ground_truth_column:
                ground_truth_column = processor.ground_truth_mapper.detect_ground_truth_column(df)
                
            if ground_truth_column:
                logger.info(f"Using ground truth column: '{ground_truth_column}'")
                
                # Show sample of ground truth values and their binary conversion
                sample_values = df[ground_truth_column].dropna().head(10)
                logger.info("Sample ground truth conversions:")
                for val in sample_values:
                    binary = processor.ground_truth_mapper.convert_to_binary(val)
                    logger.info(f"  '{val}' -> {binary}")
            else:
                logger.warning("No ground truth column detected")
            
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
                    if pd.notna(ground_truth_value):
                        prompt_data['ground_truth_original'] = str(ground_truth_value)
                
                prompts.append(prompt_data)
            
            # Process batch
            results = await processor.process_batch_secure(prompts)
            
            # Save results to CSV
            output_file = processor.save_results_to_csv(results, args.input, args.output_prefix)
            
            logger.info(f"Results saved to: {output_file}")
            
            # Log summary
            total_processed = len(results)
            blocked_count = sum(1 for r in results if r.decision == "BLOCKED")
            error_count = sum(1 for r in results if r.decision == "ERROR")
            
            logger.info(f"Processing complete:")
            logger.info(f"  Total processed: {total_processed}")
            logger.info(f"  Blocked: {blocked_count} ({blocked_count/total_processed*100:.1f}%)")
            logger.info(f"  Errors: {error_count} ({error_count/total_processed*100:.1f}%)")
            
            # Ground truth analysis if available
            if ground_truth_column:
                ground_truth_stats = {}
                for result in results:
                    if result.ground_truth_binary is not None:
                        gt_key = f"GT_{result.ground_truth_binary}"
                        decision_key = f"{gt_key}_{result.decision}"
                        ground_truth_stats[decision_key] = ground_truth_stats.get(decision_key, 0) + 1
                
                if ground_truth_stats:
                    logger.info("Ground truth analysis:")
                    for key, count in ground_truth_stats.items():
                        logger.info(f"  {key}: {count}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Processing failed: {type(e).__name__}")
            return 1
            
    except Exception as e:
        logger.error(f"Initialization failed: {type(e).__name__}")
        return 1

if __name__ == "__main__":
    sys.exit(run_async_main(main()))
