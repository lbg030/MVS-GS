"""Compatibility wrapper for the project ConfigParser."""

from droid_slam.base.parse_config import (
    ConfigParser,
    _get_by_path,
    _get_opt_name,
    _set_by_path,
    _update_config,
)

__all__ = [
    "ConfigParser",
    "_update_config",
    "_get_opt_name",
    "_set_by_path",
    "_get_by_path",
]
