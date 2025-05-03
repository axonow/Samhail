#!/usr/bin/env python3
"""
Path Setup Utility Module

This module provides utility functions for setting up Python paths correctly,
particularly for adding the project root to sys.path to ensure imports work correctly
across the project without code duplication.
"""

import os
import sys


def add_project_root_to_path(levels_to_root=0):
    """
    Add the project root directory to Python's sys.path to ensure imports work correctly.

    This function determines the project root based on the caller's location and
    the number of directory levels to traverse upward.

    Args:
        levels_to_root (int): Number of directory levels to go up from the caller's
                              location to reach the project root. Default is 0, which
                              assumes the caller is already at project root level.

    Returns:
        str: The absolute path to the project root that was added to sys.path
    """
    # Get the directory of the caller script
    caller_dir = os.path.dirname(os.path.abspath(
        sys._getframe(1).f_globals['__file__']))

    # Navigate up to the project root based on the specified number of levels
    project_root = caller_dir
    for _ in range(levels_to_root):
        project_root = os.path.dirname(project_root)

    # Only add to sys.path if it's not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root
