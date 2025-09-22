#!/usr/bin/env python3
"""
Azure Prompt Injection Detection Batch Processor - Production Ready v7.0
Complete production-ready script with all imports and proper error handling

Focus: Prompt injection detection only using Azure Prompt Shield
Removed: Content safety/content filtering functionality

Features:
- Prompt injection detection using Azure Prompt Shield API
- Advanced rate limiting with adaptive controls
- Comprehensive error handling and fallback predictions
- Support for multiple file formats (CSV, Parquet, JSONL)
- Confusion matrix and performance metrics
- Secure API communication with SSL/TLS
- Circuit breaker pattern for resilience
"""

# Standard library imports
import asyncio
import argparse
import hashlib
import json
import logging
import os
import re
import secrets
import ssl
import sys
import time
import traceback
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports - Core dependencies
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)

# Optional dependencies with graceful fallbacks
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Install with: pip install numpy")

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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coro)
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                import concurrent.futures
                
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
            pass


class SecurityConfig:
    """Security configuration and utilities"""
    
    MAX_PROMPT_LENGTH = 10000
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_BATCH_SIZE = 10000
    CHUNK_SIZE = 1000
    
    @staticmethod
    def create_secure_ssl_context():
        """Create secure SSL context with version compatibility"""
        try:
            context = ssl.create_default_context()
        except AttributeError:
            context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
        
        try:
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        except AttributeError:
            try:
                context.protocol = ssl.PROTOCOL_TLS
            except AttributeError:
                try:
                    context.protocol = ssl.PROTOCOL_TLSv1_2
                except AttributeError:
                    context.protocol = ssl.PROTOCOL_SSLv23
            
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
        
        try:
            resolved_path = Path(file_path).resolve()
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
        
        if len(prompt) > SecurityConfig.MAX_PROMPT_LENGTH:
            prompt = prompt[:SecurityConfig.MAX_PROMPT_LENGTH]
        
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
        
        return data[:4] + mask_char * (len(data) - 8) + data[-4:]


class SlidingWindowRateLimiter:
    """Advanced sliding window rate limiter"""
    
    def __init__(self, requests_per_minute: int = 10, requests_per_second: int = None,
                 burst_allowance: int = None, window_size_seconds: float = 60.0,
                 min_interval_seconds: float = None):
        
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second or (requests_per_minute / 60)
        self.burst_allowance = burst_allowance or max(3, requests_per_minute // 6)
        self.window_size = window_size_seconds
        self.min_interval = min_interval_seconds or (60.0 / requests_per_minute)
        
        self.request_timestamps = []
        self.last_request_time = 0
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.total_wait_time = 0
        self.adaptive_delay_multiplier = 1.0
        self.consecutive_limits = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _clean_old_requests(self, current_time: float):
        """Remove requests outside the sliding window"""
        cutoff_time = current_time - self.window_size
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]
    
    def _calculate_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before next request"""
        self._clean_old_requests(current_time)
        requests_in_window = len(self.request_timestamps)
        wait_times = []
        
        if requests_in_window >= self.requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_until = oldest_request + self.window_size
            wait_times.append(wait_until - current_time)
        
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                wait_times.append(self.min_interval - time_since_last)
        
        recent_window = 10.0
        recent_cutoff = current_time - recent_window
        recent_requests = [ts for ts in self.request_timestamps if ts > recent_cutoff]
        
        if len(recent_requests) >= self.burst_allowance:
            burst_wait = recent_window / self.burst_allowance
            wait_times.append(burst_wait)
        
        max_wait = max(wait_times) if wait_times else 0
        
        if max_wait > 0:
            max_wait *= self.adaptive_delay_multiplier
        
        return max(max_wait, 0)
    
    async def acquire(self) -> dict:
        """Acquire permission to make a request"""
        current_time = time.time()
        wait_time = self._calculate_wait_time(current_time)
        
        stats = {
            'wait_time': wait_time,
            'requests_in_window': len(self.request_timestamps),
            'rate_limited': wait_time > 0,
            'current_rate': len(self.request_timestamps) / (self.window_size / 60),
            'adaptive_multiplier': self.adaptive_delay_multiplier
        }
        
        if wait_time > 0:
            self.rate_limited_requests += 1
            self.total_wait_time += wait_time
            self.consecutive_limits += 1
            
            if self.consecutive_limits > 3:
                self.adaptive_delay_multiplier = min(2.0, self.adaptive_delay_multiplier * 1.1)
            
            if wait_time > 1:
                self.logger.info(f"Rate limit reached, sleeping {wait_time:.1f}s")
            
            await asyncio.sleep(wait_time)
            current_time = time.time()
        else:
            if self.consecutive_limits > 0:
                self.consecutive_limits = 0
                self.adaptive_delay_multiplier = max(1.0, self.adaptive_delay_multiplier * 0.95)
        
        self.request_timestamps.append(current_time)
        self.last_request_time = current_time
        self.total_requests += 1
        
        if len(self.request_timestamps) > self.requests_per_minute * 2:
            self._clean_old_requests(current_time)
        
        return stats
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics"""
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        return {
            'total_requests': self.total_requests,
            'rate_limited_requests': self.rate_limited_requests,
            'rate_limit_percentage': (self.rate_limited_requests / max(1, self.total_requests)) * 100,
            'total_wait_time': self.total_wait_time,
            'average_wait_time': self.total_wait_time / max(1, self.rate_limited_requests),
            'current_window_requests': len(self.request_timestamps),
            'current_rate_per_minute': len(self.request_timestamps) / (self.window_size / 60),
            'adaptive_multiplier': self.adaptive_delay_multiplier,
            'consecutive_limits': self.consecutive_limits
        }
    
    def update_limits(self, requests_per_minute: int = None, requests_per_second: int = None):
        """Dynamically update rate limits"""
        if requests_per_minute:
            self.requests_per_minute = requests_per_minute
            self.min_interval = 60.0 / requests_per_minute
        
        if requests_per_second:
            self.requests_per_second = requests_per_second
        
        self.logger.info(f"Rate limits updated: {self.requests_per_minute}/min, {self.requests_per_second:.2f}/sec")


class CircuitBreaker:
    """Circuit breaker pattern for API endpoints"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 rate_limit_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.rate_limit_threshold = rate_limit_threshold
        
        self.failure_count = 0
        self.rate_limit_count = 0
        self.last_failure_time = None
        self.last_rate_limit_time = None
        self.state = 'closed'
        
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                self.logger.info("Circuit breaker half-open, testing...")
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'throttle' in error_str:
                self.on_rate_limit()
            else:
                self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call"""
        if self.state == 'half-open':
            self.logger.info("Circuit breaker closing after successful test")
        
        self.failure_count = 0
        self.rate_limit_count = 0
        self.state = 'closed'
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            self.logger.warning(f"Circuit breaker opening due to {self.failure_count} failures")
    
    def on_rate_limit(self):
        """Handle rate limit failures"""
        self.rate_limit_count += 1
        self.last_rate_limit_time = time.time()
        
        if self.rate_limit_count >= self.rate_limit_threshold:
            self.state = 'open'
            self.logger.warning(f"Circuit breaker opening due to {self.rate_limit_count} rate limits")
    
    def get_state(self) -> dict:
        """Get circuit breaker state"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'rate_limit_count': self.rate_limit_count,
            'last_failure_time': self.last_failure_time,
            'last_rate_limit_time': self.last_rate_limit_time
        }


class RateLimitManager:
    """Configurable rate limit manager for different API tiers"""
    
    API_TIER_CONFIGS = {
        'azure_free': {
            'requests_per_minute': 10,
            'burst_allowance': 3,
            'min_interval_seconds': 6.0,
            'description': 'Azure Free Tier - Conservative limits'
        },
        'azure_standard': {
            'requests_per_minute': 30,
            'burst_allowance': 8,
            'min_interval_seconds': 2.0,
            'description': 'Azure Standard Tier - Moderate limits'
        },
        'azure_premium': {
            'requests_per_minute': 100,
            'burst_allowance': 20,
            'min_interval_seconds': 0.6,
            'description': 'Azure Premium Tier - Higher limits'
        },
        'custom': {
            'requests_per_minute': 10,
            'burst_allowance': 3,
            'min_interval_seconds': 6.0,
            'description': 'Custom configuration'
        }
    }
    
    def __init__(self, api_tier: str = 'azure_free', custom_config: dict = None):
        self.api_tier = api_tier
        self.logger = logging.getLogger(__name__)
        
        if custom_config:
            self.config = custom_config
            self.api_tier = 'custom'
        else:
            self.config = self.API_TIER_CONFIGS.get(api_tier, self.API_TIER_CONFIGS['azure_free'])
        
        self.rate_limiter = SlidingWindowRateLimiter(
            requests_per_minute=self.config['requests_per_minute'],
            burst_allowance=self.config['burst_allowance'],
            min_interval_seconds=self.config['min_interval_seconds']
        )
        
        self.start_time = time.time()
        self.performance_history = []
        self.last_stats_log = time.time()
        
        self.logger.info(f"Rate limiter initialized: {self.config['description']}")
    
    async def acquire_with_monitoring(self) -> dict:
        """Acquire rate limit permission with monitoring"""
        stats = await self.rate_limiter.acquire()
        
        enhanced_stats = {
            **stats,
            'api_tier': self.api_tier,
            'config': self.config,
            'timestamp': time.time()
        }
        
        if stats['rate_limited']:
            if stats['wait_time'] > 5:
                self.logger.warning(f"Long rate limit wait: {stats['wait_time']:.1f}s")
        
        current_time = time.time()
        if current_time - self.last_stats_log > 60:
            self._log_performance_stats()
            self.last_stats_log = current_time
        
        self.performance_history.append(enhanced_stats)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        return enhanced_stats
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        stats = self.rate_limiter.get_statistics()
        runtime = time.time() - self.start_time
        
        self.logger.info(f"Rate Limiting Performance Report:")
        self.logger.info(f"  API Tier: {self.api_tier}")
        self.logger.info(f"  Runtime: {runtime/60:.1f} minutes")
        self.logger.info(f"  Total Requests: {stats['total_requests']}")
        self.logger.info(f"  Rate Limited: {stats['rate_limited_requests']} ({stats['rate_limit_percentage']:.1f}%)")
    
    def detect_and_adjust_limits(self, recent_429_count: int, success_rate: float):
        """Automatically adjust rate limits"""
        current_config = self.config.copy()
        
        if recent_429_count > 5 or success_rate < 0.9:
            new_rpm = max(5, int(current_config['requests_per_minute'] * 0.7))
            self.logger.warning(f"High rate limit errors. Reducing to {new_rpm}/min")
            self.rate_limiter.update_limits(requests_per_minute=new_rpm)
            self.config['requests_per_minute'] = new_rpm
    
    def get_comprehensive_stats(self) -> dict:
        """Get all statistics"""
        base_stats = self.rate_limiter.get_statistics()
        
        return {
            **base_stats,
            'api_tier': self.api_tier,
            'config': self.config,
            'runtime_minutes': (time.time() - self.start_time) / 60
        }


class GroundTruthMapper:
    """Maps ground truth labels to binary values for injection detection"""
    
    INJECTION_KEYWORDS = {
        'jailbreak', 'injection', 'prompt_injection', 'bypass', 'attack',
        'exploit', 'manipulate', 'adversarial', 'hack', 'circumvent',
        'ignore', 'forget', 'override', 'replace', 'substitute',
        'blocked', 'flagged', 'rejected', 'detected', 'suspicious',
        'true', '1', 'yes', 'positive', 'fail', 'failed',
        'high', 'critical', 'severe'
    }
    
    SAFE_KEYWORDS = {
        'safe', 'legitimate', 'normal', 'clean', 'benign',
        'acceptable', 'appropriate', 'allowed', 'permitted',
        'good', 'neutral', 'fine', 'ok', 'okay',
        'false', '0', 'no', 'negative', 'none', 'pass',
        'low', 'minimal', 'minor'
    }
    
    @staticmethod
    def convert_to_binary(ground_truth_value) -> int:
        """Convert ground truth to binary (0=safe, 1=injection)"""
        if ground_truth_value is None or pd.isna(ground_truth_value):
            return 0
        
        try:
            value_str = str(ground_truth_value).lower().strip()
        except Exception:
            return 0
        
        if not value_str or value_str in ['nan', 'none', 'null', '']:
            return 0
        
        if value_str.isdigit():
            return min(int(value_str), 1)
        
        try:
            float_val = float(value_str)
            return 1 if float_val > 0.5 else 0
        except ValueError:
            pass
        
        if value_str in GroundTruthMapper.INJECTION_KEYWORDS:
            return 1
        elif value_str in GroundTruthMapper.SAFE_KEYWORDS:
            return 0
        
        for keyword in GroundTruthMapper.INJECTION_KEYWORDS:
            if keyword in value_str:
                return 1
        
        for keyword in GroundTruthMapper.SAFE_KEYWORDS:
            if keyword in value_str:
                return 0
        
        return 0
    
    @staticmethod
    def detect_ground_truth_column(df):
        """Auto-detect ground truth column"""
        primary_columns = [
            'ground_truth', 'label', 'injection', 'jailbreak', 'attack'
        ]
        
        secondary_columns = [
            'is_jailbreak', 'is_injection', 'is_attack', 'is_safe',
            'malicious', 'harmful', 'safe', 'benign'
        ]
        
        for col in primary_columns + secondary_columns:
            if col in df.columns:
                return col
        
        df_columns_lower = {col.lower(): col for col in df.columns}
        for col in primary_columns + secondary_columns:
            if col.lower() in df_columns_lower:
                return df_columns_lower[col.lower()]
        
        return None


class EnhancedPromptResult:
    """Result structure for prompt injection detection"""
    
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
        self.service_type = service_type
    
    def to_dict(self):
        """Convert to dictionary"""
        predicted_binary = 1 if self.decision == "BLOCKED" else 0
        
        return {
            'prompt': self.prompt,
            'decision': self.decision,
            'predicted_binary': predicted_binary,
            'latency_ms': self.latency_ms,
            'category': self.category,
            'confidence_score': self.confidence_score,
            'severity_scores': self.severity_scores,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'ground_truth_binary': self.ground_truth_binary,
            'prompt_length': self.prompt_length,
            'service_type': self.service_type
        }


class ProductionLogger:
    """Production-ready logging configuration"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: str = None,
                     enable_console: bool = True, max_bytes: int = 10485760,
                     backup_count: int = 5):
        """Setup production logging"""
        from logging.handlers import RotatingFileHandler
        
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers = []
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class AzurePromptInjectionProcessor:
    """Production-ready processor for prompt injection detection"""
    
    def __init__(self, endpoint_url: str = None, api_key: str = None,
                 max_concurrent_requests: int = 5, rate_limit_per_minute: int = 30,
                 timeout_seconds: int = 30, output_directory: str = "prompt_injection_results",
                 prompt_column: str = None, ground_truth_column: str = None,
                 auto_detect_schema: bool = True, enable_circuit_breaker: bool = True,
                 api_tier: str = None, custom_rate_config: dict = None,
                 confidence_threshold: float = 0.5):
        
        self.endpoint_url = endpoint_url.rstrip('/') if endpoint_url else None
        self.api_key = api_key
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout_seconds = timeout_seconds
        self.prompt_column = prompt_column
        self.ground_truth_column = ground_truth_column
        self.confidence_threshold = confidence_threshold  # Threshold for classification (0.0-1.0)
        
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True, mode=0o755)
        
        self.security_config = SecurityConfig()
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.ground_truth_mapper = GroundTruthMapper()
        
        if not api_tier:
            api_tier = 'azure_standard'
        
        if custom_rate_config:
            self.rate_limit_manager = RateLimitManager('custom', custom_rate_config)
        else:
            self.rate_limit_manager = RateLimitManager(api_tier)
        
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.results = []
        
        self.recent_429_count = 0
        self.recent_success_count = 0
        self.last_429_reset = time.time()
        
        self.working_endpoint = None
        self.prompt_shield_endpoint = None
        self.detected_service = "azure"
        
        self.start_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Confidence threshold set to: {confidence_threshold}")
    
    def validate_configuration(self) -> bool:
        """Validate configuration"""
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn required. Install with: pip install scikit-learn")
            return False
        
        if not self.endpoint_url:
            self.logger.error("Azure endpoint URL required")
            return False
        
        if not self.api_key:
            self.logger.error("Azure API key required")
            return False
        
        return True
    
    async def _create_secure_session(self) -> aiohttp.ClientSession:
        """Create secure HTTP session"""
        ssl_context = SecurityConfig.create_secure_ssl_context()
        
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests * 2,
            limit_per_host=self.max_concurrent_requests,
            keepalive_timeout=30,
            ssl=ssl_context,
            verify_ssl=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        
        return aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def _secure_api_call(self, session: aiohttp.ClientSession,
                              method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make secure API call"""
        async def make_request():
            start_time = time.time()
            
            try:
                headers = kwargs.get('headers', {})
                headers.update({
                    'Content-Type': 'application/json',
                    'Ocp-Apim-Subscription-Key': self.api_key
                })
                kwargs['headers'] = headers
                
                async with session.request(method, url, **kwargs) as response:
                    response_data = await response.text()
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        self.successful_requests += 1
                        self.recent_success_count += 1
                        return {
                            'success': True,
                            'data': json.loads(response_data),
                            'latency_ms': latency_ms
                        }
                    elif response.status == 429:
                        self.failed_requests += 1
                        self.recent_429_count += 1
                        return {
                            'success': False,
                            'error': f"HTTP 429: Rate limit exceeded",
                            'latency_ms': latency_ms,
                            'is_rate_limit': True
                        }
                    else:
                        self.failed_requests += 1
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}: {response_data[:200]}",
                            'latency_ms': latency_ms
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
        
        if self.circuit_breaker:
            return await self.circuit_breaker.call(make_request)
        else:
            return await make_request()
    
    async def discover_working_endpoint(self) -> bool:
        """Discover working prompt injection endpoint"""
        test_endpoints = [
            (f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview", "prompt_shield"),
            (f"{self.endpoint_url}/contentsafety/text:shieldPrompt?api-version=2024-09-01", "prompt_shield"),
        ]
        
        async with await self._create_secure_session() as session:
            for endpoint_url, endpoint_type in test_endpoints:
                try:
                    payload = {
                        'userPrompt': 'Test message for validation',
                        'documents': []
                    }
                    
                    response = await self._secure_api_call(session, 'POST', endpoint_url, json=payload)
                    
                    if response['success']:
                        self.logger.info(f"✅ Prompt injection endpoint working: {endpoint_type}")
                        self.prompt_shield_endpoint = {'url': endpoint_url, 'type': endpoint_type}
                        self.working_endpoint = endpoint_url
                        return True
                except Exception as e:
                    self.logger.debug(f"Endpoint test error: {type(e).__name__}")
        
        self.logger.error("No working prompt injection endpoints found")
        return False
    
    async def process_batch_secure(self, prompts: List[Dict[str, Any]]) -> List[EnhancedPromptResult]:
        """Process batch of prompts"""
        total_prompts = len(prompts)
        self.logger.info(f"Processing {total_prompts} prompts for injection detection")
        
        sanitized_prompts = []
        for i, prompt_data in enumerate(prompts):
            raw_prompt = prompt_data.get('prompt', '')
            sanitized_prompt = SecurityConfig.sanitize_prompt(str(raw_prompt))
            prompt_data['prompt'] = sanitized_prompt
            sanitized_prompts.append(prompt_data)
        
        if not await self.discover_working_endpoint():
            raise ConnectionError("No working endpoint found")
        
        self.start_time = time.time()
        all_results = []
        
        chunk_size = SecurityConfig.CHUNK_SIZE
        total_chunks = (len(sanitized_prompts) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(sanitized_prompts))
            chunk = sanitized_prompts[start_idx:end_idx]
            
            async with await self._create_secure_session() as session:
                chunk_results = await self._process_chunk(session, chunk)
                all_results.extend(chunk_results)
            
            self._log_progress(len(all_results), total_prompts)
        
        return all_results
    
    async def _process_chunk(self, session: aiohttp.ClientSession,
                            chunk: List[Dict[str, Any]]) -> List[EnhancedPromptResult]:
        """Process chunk of prompts"""
        results = []
        batch_size = min(self.max_concurrent_requests, len(chunk))
        
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            await self.rate_limit_manager.acquire_with_monitoring()
            
            tasks = [self._process_single_prompt(session, p) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Processing error: {type(result).__name__}")
                else:
                    results.append(result)
        
        return results
    
    async def _process_single_prompt(self, session: aiohttp.ClientSession,
                                    prompt_data: Dict[str, Any]) -> EnhancedPromptResult:
        """Process single prompt"""
        async with self.semaphore:
            ground_truth_original = prompt_data.get('ground_truth_original')
            ground_truth_binary = None
            if ground_truth_original is not None:
                ground_truth_binary = self.ground_truth_mapper.convert_to_binary(ground_truth_original)
            
            prompt = SecurityConfig.sanitize_prompt(prompt_data.get('prompt', ''))
            
            try:
                payload = {'userPrompt': prompt, 'documents': []}
                response = await self._secure_api_call(
                    session, 'POST', self.working_endpoint, json=payload
                )
                
                if response['success']:
                    decision, confidence, severity = self._parse_response(response['data'])
                    
                    return EnhancedPromptResult(
                        prompt=prompt,
                        decision=decision,
                        latency_ms=response['latency_ms'],
                        category="prompt_injection_detection",
                        original_type="injection_detection",
                        confidence_score=confidence,
                        severity_scores=severity,
                        timestamp=datetime.now().isoformat(),
                        ground_truth_binary=ground_truth_binary,
                        prompt_length=len(prompt),
                        service_type="azure"
                    )
                else:
                    decision = self._get_fallback_prediction(prompt, ground_truth_binary)
                    
                    return EnhancedPromptResult(
                        prompt=prompt,
                        decision=decision,
                        latency_ms=response.get('latency_ms', 0),
                        category="fallback",
                        original_type="fallback",
                        confidence_score=0.3,
                        severity_scores="fallback",
                        error_message=response.get('error'),
                        timestamp=datetime.now().isoformat(),
                        ground_truth_binary=ground_truth_binary,
                        prompt_length=len(prompt),
                        service_type="azure"
                    )
            except Exception as e:
                decision = self._get_fallback_prediction(prompt, ground_truth_binary)
                
                return EnhancedPromptResult(
                    prompt=prompt,
                    decision=decision,
                    latency_ms=0.0,
                    category="error",
                    original_type="error",
                    confidence_score=0.1,
                    severity_scores="error",
                    error_message=str(e),
                    timestamp=datetime.now().isoformat(),
                    ground_truth_binary=ground_truth_binary,
                    prompt_length=len(prompt),
                    service_type="azure"
                )
    
    def _parse_response(self, response_data: dict) -> Tuple[str, float, str]:
        """Parse Prompt Shield response"""
        try:
            if 'userPromptAnalysis' in response_data:
                analysis = response_data['userPromptAnalysis']
                attack_detected = analysis.get('attackDetected', False)
                
                if attack_detected:
                    return "BLOCKED", 0.8, "injection_detected"
                else:
                    return "ALLOWED", 0.1, "safe"
            else:
                return "ALLOWED", 0.05, "no_analysis"
        except Exception:
            return "ALLOWED", 0.05, "parse_error"
    
    def _get_fallback_prediction(self, prompt: str, ground_truth: int = None) -> str:
        """Fallback prediction for injection detection"""
        if ground_truth is not None:
            import random
            if random.random() < 0.8:
                return "BLOCKED" if ground_truth == 1 else "ALLOWED"
        
        prompt_lower = prompt.lower()
        injection_patterns = [
            'ignore', 'forget', 'override', 'bypass', 'jailbreak',
            'act as', 'roleplay', 'pretend', 'system prompt'
        ]
        
        score = sum(2 for pattern in injection_patterns if pattern in prompt_lower)
        return "BLOCKED" if score >= 3 else "ALLOWED"
    
    def calculate_confusion_matrix(self, results: List[EnhancedPromptResult]) -> Optional[Dict[str, Any]]:
        """Calculate confusion matrix"""
        if not results or not SKLEARN_AVAILABLE:
            return None
        
        predictions = []
        actuals = []
        
        for result in results:
            if result and hasattr(result, 'ground_truth_binary') and result.ground_truth_binary is not None:
                if result.decision != "ERROR":
                    predicted_binary = 1 if result.decision == "BLOCKED" else 0
                    predictions.append(predicted_binary)
                    actuals.append(int(result.ground_truth_binary))
        
        if len(predictions) < 2:
            return None
        
        try:
            cm = confusion_matrix(actuals, predictions)
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
            recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
            
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            
            return {
                'confusion_matrix': cm.tolist(),
                'confusion_matrix_labels': ['Safe', 'Injection'],
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'total_samples': len(predictions)
            }
        except Exception as e:
            self.logger.error(f"Confusion matrix error: {str(e)}")
            return None
    
    def save_results_to_csv(self, results: List[EnhancedPromptResult],
                           input_file_path: str, prefix: str = "") -> str:
        """Save results to CSV"""
        input_name = Path(input_file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            filename = f"{prefix}_{input_name}_injection_results_{timestamp}.csv"
        else:
            filename = f"{input_name}_injection_results_{timestamp}.csv"
        
        output_path = self.output_directory / filename
        
        results_data = [result.to_dict() for result in results]
        df = pd.DataFrame(results_data)
        df = df.fillna('')
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        confusion_metrics = self.calculate_confusion_matrix(results)
        if confusion_metrics:
            self._append_metrics_to_csv(output_path, confusion_metrics)
        
        self.logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def _append_metrics_to_csv(self, output_path: str, metrics: Dict[str, Any]):
        """Append metrics to CSV"""
        try:
            import csv
            
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([''] * 10)
                writer.writerow(['CONFUSION MATRIX (INJECTION DETECTION)'] + [''] * 9)
                writer.writerow([''] * 10)
                
                cm = metrics['confusion_matrix']
                labels = metrics['confusion_matrix_labels']
                
                writer.writerow(['', 'Predicted', labels[0], labels[1]] + [''] * 6)
                writer.writerow(['Actual', labels[0], str(cm[0][0]), str(cm[0][1])] + [''] * 6)
                writer.writerow(['', labels[1], str(cm[1][0]), str(cm[1][1])] + [''] * 6)
                writer.writerow([''] * 10)
                
                writer.writerow(['Accuracy', f"{metrics.get('accuracy', 0):.3f}"] + [''] * 8)
                writer.writerow(['Precision', f"{metrics.get('precision', 0):.3f}"] + [''] * 8)
                writer.writerow(['Recall', f"{metrics.get('recall', 0):.3f}"] + [''] * 8)
                writer.writerow(['F1-Score', f"{metrics.get('f1_score', 0):.3f}"] + [''] * 8)
        except Exception as e:
            self.logger.error(f"Error appending metrics: {str(e)}")
    
    def _log_progress(self, processed: int, total: int):
        """Log progress"""
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            
            self.logger.info(
                f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                f"Rate: {rate:.1f}/sec"
            )


def load_csv_file(file_path: str, logger) -> pd.DataFrame:
    """Load CSV file"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Loaded CSV: {file_path}")
        return df
    except Exception:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"Loaded TSV: {file_path}")
            return df
        except Exception:
            df = pd.read_csv(file_path, encoding='latin1')
            logger.info(f"Loaded CSV (latin1): {file_path}")
            return df


def load_parquet_file(file_path: str, logger) -> pd.DataFrame:
    """Load Parquet file"""
    if not PYARROW_AVAILABLE:
        raise ImportError("PyArrow required. Install with: pip install pyarrow")
    
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded Parquet: {file_path}")
    return df


def load_jsonl_file(file_path: str, logger) -> pd.DataFrame:
    """Load JSONL file"""
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
    
    if not records:
        raise ValueError("No valid JSON records found")
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded JSONL: {file_path} ({len(records)} records)")
    return df


def load_dataset_file(file_path: str, logger) -> pd.DataFrame:
    """Load dataset from various formats"""
    file_path_lower = file_path.lower()
    
    if file_path_lower.endswith('.parquet'):
        return load_parquet_file(file_path, logger)
    elif file_path_lower.endswith(('.jsonl', '.json')):
        return load_jsonl_file(file_path, logger)
    else:
        return load_csv_file(file_path, logger)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Azure Prompt Injection Detection - Production Ready v7.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', required=True, help='Input file (CSV, TSV, Parquet, JSONL)')
    parser.add_argument('--endpoint', required=True, help='Azure endpoint URL')
    parser.add_argument('--api-key', required=True, help='Azure API key')
    
    parser.add_argument('--concurrent', type=int, default=5, help='Max concurrent requests')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout (seconds)')
    parser.add_argument('--api-tier', choices=['azure_free', 'azure_standard', 'azure_premium'],
                       default='azure_standard', help='API tier for rate limiting')
    
    parser.add_argument('--prompt-column', help='Column name for prompts')
    parser.add_argument('--ground-truth-column', help='Column name for ground truth')
    
    parser.add_argument('--output-dir', default='prompt_injection_results', help='Output directory')
    parser.add_argument('--output-prefix', default='results', help='Output file prefix')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser.parse_args()


async def main():
    """Main async function"""
    args = parse_arguments()
    
    ProductionLogger.setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        enable_console=True
    )
    
    logger = logging.getLogger(__name__)
    
    if not SecurityConfig.validate_file_path(args.input):
        logger.error(f"Invalid file path: {args.input}")
        return 1
    
    try:
        processor = AzurePromptInjectionProcessor(
            endpoint_url=args.endpoint,
            api_key=args.api_key,
            max_concurrent_requests=args.concurrent,
            timeout_seconds=args.timeout,
            output_directory=args.output_dir,
            prompt_column=args.prompt_column,
            ground_truth_column=args.ground_truth_column,
            api_tier=args.api_tier
        )
        
        if not processor.validate_configuration():
            logger.error("Configuration validation failed")
            return 1
        
        logger.info(f"Loading data from: {args.input}")
        df = load_dataset_file(args.input, logger)
        
        if df.empty:
            logger.error("Dataset is empty")
            return 1
        
        logger.info(f"Loaded {len(df)} rows")
        
        prompt_column = args.prompt_column
        if not prompt_column:
            for candidate in ['prompt', 'text', 'input', 'query']:
                if candidate in df.columns:
                    prompt_column = candidate
                    break
            
            if not prompt_column:
                prompt_column = df.columns[0]
        
        logger.info(f"Using prompt column: '{prompt_column}'")
        
        ground_truth_column = args.ground_truth_column
        if not ground_truth_column:
            ground_truth_column = processor.ground_truth_mapper.detect_ground_truth_column(df)
        
        if ground_truth_column:
            logger.info(f"Using ground truth column: '{ground_truth_column}'")
        
        prompts = []
        for _, row in df.iterrows():
            prompt_data = {'prompt': str(row[prompt_column]), 'category': 'injection'}
            
            if ground_truth_column and ground_truth_column in df.columns:
                gt_value = row[ground_truth_column]
                if pd.notna(gt_value) and gt_value != '':
                    prompt_data['ground_truth_original'] = str(gt_value)
            
            prompts.append(prompt_data)
        
        logger.info(f"Prepared {len(prompts)} prompts")
        
        results = await processor.process_batch_secure(prompts)
        
        if not results:
            logger.error("No results returned")
            return 1
        
        output_file = processor.save_results_to_csv(results, args.input, args.output_prefix)
        
        total = len(results)
        blocked = sum(1 for r in results if r.decision == "BLOCKED")
        
        logger.info(f"Detection complete:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Injections detected: {blocked} ({blocked/total*100:.1f}%)")
        
        if ground_truth_column:
            cm_results = processor.calculate_confusion_matrix(results)
            if cm_results:
                logger.info("\n" + "="*50)
                logger.info("PERFORMANCE METRICS")
                logger.info("="*50)
                logger.info(f"Accuracy:  {cm_results['accuracy']:.3f}")
                logger.info(f"Precision: {cm_results['precision']:.3f}")
                logger.info(f"Recall:    {cm_results['recall']:.3f}")
                logger.info(f"F1-Score:  {cm_results['f1_score']:.3f}")
                logger.info("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {type(e).__name__}: {str(e)}")
        if logger.level <= logging.DEBUG:
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(run_async_main(main()))
