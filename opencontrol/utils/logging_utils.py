"""
Logging Utilities for OpenControl.

This module provides comprehensive logging setup and utilities for the
OpenControl system, including structured logging, performance tracking,
and integration with monitoring systems.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import logging.config
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import traceback


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_level: int = logging.INFO):
        """End a named timer and log the duration."""
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.logger.log(log_level, f"Timer '{name}' completed", 
                           extra={'timer_name': name, 'duration_seconds': duration})
            del self.timers[name]
            return duration
        else:
            self.logger.warning(f"Timer '{name}' was not started")
            return None
    
    def log_metric(self, metric_name: str, value: float, unit: str = None):
        """Log a performance metric."""
        extra_data = {'metric_name': metric_name, 'metric_value': value}
        if unit:
            extra_data['metric_unit'] = unit
        
        self.logger.info(f"Metric: {metric_name} = {value}", extra=extra_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        structured: Whether to use structured JSON logging
        console_output: Whether to output to console
        
    Returns:
        Configured root logger
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
    
    # Set up specific loggers
    setup_opencontrol_loggers(numeric_level)
    
    root_logger.info("Logging system initialized", 
                    extra={'log_level': log_level, 'structured': structured, 
                          'log_file': log_file, 'console_output': console_output})
    
    return root_logger


def setup_opencontrol_loggers(log_level: int):
    """Setup specific loggers for OpenControl components."""
    
    # Main OpenControl logger
    opencontrol_logger = logging.getLogger('opencontrol')
    opencontrol_logger.setLevel(log_level)
    
    # Component-specific loggers
    component_loggers = [
        'opencontrol.core',
        'opencontrol.training',
        'opencontrol.control',
        'opencontrol.evaluation',
        'opencontrol.deployment',
        'opencontrol.data'
    ]
    
    for logger_name in component_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
    
    # Set external library log levels
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'opencontrol')
    
    return logging.getLogger(name)


def log_function_call(func):
    """Decorator to log function calls with arguments and timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function entry
        logger.debug(f"Entering {func_name}", 
                    extra={'function': func_name, 'args_count': len(args), 
                          'kwargs_count': len(kwargs)})
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed {func_name}", 
                        extra={'function': func_name, 'duration_seconds': duration, 
                              'status': 'success'})
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log exception
            logger.error(f"Exception in {func_name}: {e}", 
                        extra={'function': func_name, 'duration_seconds': duration, 
                              'status': 'error', 'exception_type': type(e).__name__},
                        exc_info=True)
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls with arguments and timing."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function entry
        logger.debug(f"Entering async {func_name}", 
                    extra={'function': func_name, 'args_count': len(args), 
                          'kwargs_count': len(kwargs), 'async': True})
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed async {func_name}", 
                        extra={'function': func_name, 'duration_seconds': duration, 
                              'status': 'success', 'async': True})
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log exception
            logger.error(f"Exception in async {func_name}: {e}", 
                        extra={'function': func_name, 'duration_seconds': duration, 
                              'status': 'error', 'exception_type': type(e).__name__, 
                              'async': True},
                        exc_info=True)
            raise
    
    return wrapper


class LoggingContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def create_performance_logger(name: str = None) -> PerformanceLogger:
    """Create a performance logger instance."""
    logger = get_logger(name)
    return PerformanceLogger(logger)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG", structured=True)
    
    # Get logger
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test performance logging
    perf_logger = create_performance_logger(__name__)
    perf_logger.start_timer("test_operation")
    time.sleep(0.1)
    perf_logger.end_timer("test_operation")
    perf_logger.log_metric("test_metric", 42.0, "units")
    
    # Test context logging
    with LoggingContext(logger, request_id="12345", user_id="user123"):
        logger.info("Message with context")
    
    # Test function decorator
    @log_function_call
    def test_function(x, y=None):
        return x + (y or 0)
    
    result = test_function(1, y=2)
    logger.info(f"Function result: {result}") 