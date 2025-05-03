#!/usr/bin/env python3
"""
System Monitoring Module

This module provides utilities for monitoring system resources like memory and CPU usage.
It's designed to be reusable across different models and applications in the Samhail project.
"""

import os
import sys
import time
import threading
import platform
import psutil
from datetime import datetime


class MemoryManager:
    """
    Manages memory usage and provides protection mechanisms against memory overflow.
    """

    def __init__(self, logger, memory_limit_mb=None, memory_limit_percentage=85):
        """
        Initialize memory manager with specified limits.

        Args:
            logger: Logger instance for recording memory events
            memory_limit_mb (int, optional): Explicit memory threshold in MB
            memory_limit_percentage (float): Percentage of system memory to use if threshold not specified
        """
        self.logger = logger
        self.memory_limit_percentage = memory_limit_percentage

        # Get system memory info
        system_memory = psutil.virtual_memory()
        self.total_system_memory_mb = system_memory.total / (1024 * 1024)

        # Calculate memory limit based on input parameters
        if memory_limit_mb:
            self.memory_limit_mb = memory_limit_mb
        else:
            # Use percentage of available memory
            self.memory_limit_mb = int(
                self.total_system_memory_mb * (memory_limit_percentage / 100))

        # Log memory management setup
        self.logger.info("Memory manager initialized", extra={
            "metrics": {
                "total_system_memory_mb": self.total_system_memory_mb,
                "memory_limit_mb": self.memory_limit_mb,
                "memory_limit_percentage": memory_limit_percentage
            }
        })

    def get_current_memory_usage(self):
        """
        Get current process memory usage.

        Returns:
            dict: Memory usage statistics including current, peak, and percentage usage
        """
        # Get memory info for current process
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Calculate memory usage in MB
        current_memory_mb = memory_info.rss / (1024 * 1024)

        # Get system memory info for percentage calculation
        system_memory = psutil.virtual_memory()
        system_used_percentage = system_memory.percent

        # Get peak memory usage if available
        peak_memory_mb = current_memory_mb  # Default to current if peak not available

        # On Linux systems, we can get more accurate peak memory
        if hasattr(memory_info, 'peak'):
            peak_memory_mb = memory_info.peak / (1024 * 1024)

        return {
            "current_mb": current_memory_mb,
            "peak_mb": peak_memory_mb,
            "percent_used": (current_memory_mb / self.total_system_memory_mb) * 100,
            "system_percent_used": system_used_percentage,
            "limit_mb": self.memory_limit_mb
        }

    def check_memory_health(self):
        """
        Check if memory usage is within healthy limits.

        Returns:
            tuple: (is_healthy, memory_usage_dict, warning_message)
        """
        memory_usage = self.get_current_memory_usage()

        # Check if we're approaching the limit (within 90% of limit)
        warning_threshold = 0.9 * self.memory_limit_mb
        danger_threshold = 0.95 * self.memory_limit_mb

        if memory_usage["current_mb"] > danger_threshold:
            message = f"DANGER: Memory usage at {memory_usage['current_mb']:.2f} MB, {(memory_usage['current_mb'] / self.memory_limit_mb) * 100:.1f}% of limit"
            is_healthy = False
        elif memory_usage["current_mb"] > warning_threshold:
            message = f"WARNING: Memory usage at {memory_usage['current_mb']:.2f} MB, {(memory_usage['current_mb'] / self.memory_limit_mb) * 100:.1f}% of limit"
            is_healthy = True
        else:
            message = None
            is_healthy = True

        return is_healthy, memory_usage, message

    def emergency_memory_cleanup(self):
        """
        Attempt to free memory in emergency situations.
        This is called when memory usage exceeds limits.

        Returns:
            bool: True if cleanup was successful in reducing memory usage
        """
        # Log the emergency cleanup attempt
        self.logger.warning("Emergency memory cleanup initiated", extra={
            "metrics": self.get_current_memory_usage()
        })

        print("\033[1;31m⚠️ Emergency memory cleanup initiated\033[0m")

        # Force garbage collection
        import gc
        gc.collect()

        # Sleep briefly to allow system to stabilize
        time.sleep(1)

        # Check if memory improved
        before_cleanup = self.get_current_memory_usage()["current_mb"]
        after_cleanup = self.get_current_memory_usage()["current_mb"]
        memory_freed = before_cleanup - after_cleanup

        self.logger.info("Emergency memory cleanup completed", extra={
            "metrics": {
                "before_cleanup_mb": before_cleanup,
                "after_cleanup_mb": after_cleanup,
                "memory_freed_mb": memory_freed
            }
        })

        # Return True if we freed a significant amount of memory (> 5% of limit)
        significant_improvement = memory_freed > (0.05 * self.memory_limit_mb)
        return significant_improvement


class ResourceMonitor:
    """
    Monitors system resources including memory, CPU, and disk usage.
    Provides methods to log resource usage and warnings.
    """

    def __init__(self, logger, memory_limit_mb=None, memory_limit_percentage=85,
                 monitoring_interval=10.0, warning_threshold=0.8):
        """
        Initialize the resource monitor.

        Args:
            logger: Logger instance for recording resource metrics
            memory_limit_mb (int, optional): Explicit memory threshold in MB
            memory_limit_percentage (float): Percentage of system memory to use if threshold not specified
            monitoring_interval (float): Interval in seconds between monitoring checks
            warning_threshold (float): Threshold (0.0-1.0) of resource usage that triggers warnings
        """
        self.logger = logger
        self.monitoring_interval = monitoring_interval
        self.warning_threshold = warning_threshold

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            logger=logger,
            memory_limit_mb=memory_limit_mb,
            memory_limit_percentage=memory_limit_percentage
        )

        # Initialize monitoring thread as None
        self.monitoring_thread = None
        self.should_monitor = False

        # Track progress of long-running operations
        self.current_operation = None
        self.progress_percent = 0
        self.operation_start_time = None

        # Log initialization
        self.logger.info("ResourceMonitor initialized", extra={
            "metrics": {
                "monitoring_interval": monitoring_interval,
                "warning_threshold": warning_threshold,
                "memory_limit_mb": self.memory_manager.memory_limit_mb
            }
        })

    def get_resource_usage(self):
        """
        Get comprehensive resource usage statistics.

        Returns:
            dict: Resource usage metrics for CPU, memory, disk, and system
        """
        # Get memory usage
        memory_usage = self.memory_manager.get_current_memory_usage()

        # Get CPU usage for this process and system-wide
        process = psutil.Process(os.getpid())
        process_cpu = process.cpu_percent(interval=0.1) / psutil.cpu_count()
        system_cpu = psutil.cpu_percent(interval=0.1)

        # Get disk usage for the current working directory
        disk = psutil.disk_usage(os.getcwd())
        disk_usage = {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent_used": disk.percent
        }

        # Get active threads
        thread_count = threading.active_count()

        # Return consolidated metrics
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": memory_usage,
            "cpu": {
                "process_percent": process_cpu,
                "system_percent": system_cpu,
                "cores": psutil.cpu_count(),
                "logical_cores": psutil.cpu_count(logical=True)
            },
            "disk": disk_usage,
            "threads": thread_count,
            "process_id": os.getpid()
        }

    def print_system_architecture(self):
        """Print system architecture information to console"""
        print("\n\033[1;34m📋 System Information\033[0m")
        print(f"\033[1m🖥️  Platform: {platform.platform()}\033[0m")
        print(f"\033[1m🧠 Processor: {platform.processor()}\033[0m")
        print(
            f"\033[1m💾 Total Memory: {self.memory_manager.total_system_memory_mb / 1024:.2f} GB\033[0m")
        print(f"\033[1m💻 Python Version: {sys.version}\033[0m")
        print(
            f"\033[1m🧮 Memory Limit Set: {self.memory_manager.memory_limit_mb / 1024:.2f} GB\033[0m")

        # Log this information as well
        self.logger.info("System architecture", extra={
            "metrics": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "total_memory_gb": self.memory_manager.total_system_memory_mb / 1024,
                "memory_limit_gb": self.memory_manager.memory_limit_mb / 1024
            }
        })

    def print_resource_usage(self, prefix=""):
        """
        Print current resource usage to console.

        Args:
            prefix (str): Prefix for each printed line (for indentation)
        """
        resources = self.get_resource_usage()
        memory = resources["memory"]
        cpu = resources["cpu"]

        print(
            f"{prefix}\033[1m💾 Memory: {memory['current_mb']:.2f} MB / {memory['limit_mb']:.2f} MB ({memory['percent_used']:.1f}%)\033[0m")
        print(
            f"{prefix}\033[1m🔄 CPU: Process: {cpu['process_percent']:.1f}%, System: {cpu['system_percent']:.1f}%\033[0m")
        print(f"{prefix}\033[1m🧵 Threads: {resources['threads']}\033[0m")

        # Add progress information if available
        if self.current_operation and self.operation_start_time:
            elapsed = time.time() - self.operation_start_time
            if self.progress_percent > 0:
                estimated_total = elapsed / (self.progress_percent / 100)
                remaining = estimated_total - elapsed
                print(
                    f"{prefix}\033[1m⏱️ Operation: {self.current_operation} - {self.progress_percent:.1f}% complete, {remaining:.1f}s remaining\033[0m")
            else:
                print(
                    f"{prefix}\033[1m⏱️ Operation: {self.current_operation} - {elapsed:.1f}s elapsed\033[0m")

    def _monitoring_loop(self):
        """
        Internal monitoring loop that runs in a separate thread.
        Periodically checks resource usage and logs it.
        """
        self.logger.info("Resource monitoring started")

        while self.should_monitor:
            try:
                # Get resource metrics
                resources = self.get_resource_usage()
                memory = resources["memory"]

                # Check if we need to log a warning
                is_healthy, _, warning = self.memory_manager.check_memory_health()

                if warning:
                    # Log warning message with current metrics
                    self.logger.warning(warning, extra={
                        "metrics": resources
                    })

                    # Print warning to console
                    print(f"\033[1;33m⚠️ {warning}\033[0m")

                    # If memory usage is critical, try emergency cleanup
                    if not is_healthy:
                        self.memory_manager.emergency_memory_cleanup()
                else:
                    # Log normal resource usage at regular intervals
                    self.logger.info("Resource usage metrics", extra={
                        "metrics": resources,
                        "operation": self.current_operation,
                        "progress_percent": self.progress_percent
                    })

                # Sleep until next check
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                # Sleep longer on error
                time.sleep(self.monitoring_interval * 2)

    def start(self, operation_name=None):
        """
        Start resource monitoring in a background thread.

        Args:
            operation_name (str, optional): Name of the operation being monitored
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            # Already monitoring
            return

        # Set operation properties
        self.current_operation = operation_name
        self.progress_percent = 0
        self.operation_start_time = time.time()

        # Set flag to enable monitoring
        self.should_monitor = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True  # Make thread daemon so it doesn't block program exit
        )
        self.monitoring_thread.start()

        # Log initial resource usage
        initial_resources = self.get_resource_usage()
        self.logger.info(f"Resource monitoring started for operation: {operation_name}", extra={
            "metrics": initial_resources,
            "operation": operation_name
        })

    def stop(self):
        """
        Stop the resource monitoring thread.
        """
        if not self.monitoring_thread:
            return

        # Signal thread to stop
        self.should_monitor = False

        # Wait for thread to finish (with timeout)
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

        # Log final resource usage
        final_resources = self.get_resource_usage()

        # Calculate duration if we were tracking an operation
        duration = None
        if self.operation_start_time:
            duration = time.time() - self.operation_start_time

        self.logger.info("Resource monitoring stopped", extra={
            "metrics": final_resources,
            "operation": self.current_operation,
            "duration": duration
        })

        # Reset operation tracking
        self.current_operation = None
        self.progress_percent = 0
        self.operation_start_time = None

    def log_progress(self, message, progress_percent=None, operation=None, extra_metrics=None):
        """
        Log progress of an ongoing operation with current resource metrics.

        Args:
            message (str): Progress message to log
            progress_percent (float, optional): Percentage of operation completed (0-100)
            operation (str, optional): Operation name (updates current_operation if provided)
            extra_metrics (dict, optional): Additional metrics to include in the log
        """
        # Update tracking information if provided
        if operation:
            self.current_operation = operation

        if progress_percent is not None:
            self.progress_percent = progress_percent

        # Get current resource usage
        resources = self.get_resource_usage()

        # Merge with extra metrics if provided
        metrics = {"system_resources": resources}
        if extra_metrics:
            metrics.update(extra_metrics)

        # Add progress information
        if self.progress_percent > 0:
            metrics["progress_percent"] = self.progress_percent

        if self.operation_start_time:
            elapsed = time.time() - self.operation_start_time
            metrics["elapsed_time"] = elapsed

            # Calculate ETA if we have progress percentage
            if self.progress_percent > 0:
                estimated_total = elapsed / (self.progress_percent / 100)
                metrics["estimated_total_time"] = estimated_total
                metrics["estimated_remaining_time"] = estimated_total - elapsed

        # Log the progress
        self.logger.info(message, extra={"metrics": metrics})

        # Print progress to console if progress percentage provided
        if progress_percent is not None:
            print(
                f"\033[1m📊 {message} - {progress_percent:.1f}% complete\033[0m")
