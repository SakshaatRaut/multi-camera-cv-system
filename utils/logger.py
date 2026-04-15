# Logging Utility
"""
Centralized logging utility
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class SystemLogger:
    """System-wide logger with file and console output"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        if not self._initialized:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True)
            
            self.logger = logging.getLogger('MultiCamCV')
            self.logger.setLevel(log_level)
            
            if not self.logger.handlers:
                # Console handler
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(log_level)
                console_format = logging.Formatter(
                    '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_format)
                self.logger.addHandler(console_handler)
                
                # File handler
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = self.log_dir / f'system_{timestamp}.log'
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - [%(levelname)s] - %(processName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
                
                self._initialized = True
    
    def get_logger(self):
        return self.logger
    
    @staticmethod
    def get_process_logger(name):
        return logging.getLogger(f'MultiCamCV.{name}')


def get_logger(name=None):
    """Get a logger instance"""
    system_logger = SystemLogger()
    if name:
        return SystemLogger.get_process_logger(name)
    return system_logger.get_logger()