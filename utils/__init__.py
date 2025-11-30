"""Utilities package shim.

This module re-exports helpers from the old `simulation.utils` package
and the new `utils.logging` module so existing imports like
`from utils import show_applied_params_banner` continue to work.
"""
# Re-export banner/chart helpers from simulation.utils for compatibility
try:
	from simulation.utils.banner import show_applied_params_banner  # type: ignore
except Exception:
	# Fallback: provide a no-op
	def show_applied_params_banner():
		return False

try:
	from simulation.utils.charts import create_performance_chart  # type: ignore
except Exception:
	create_performance_chart = None

# Export logging helpers from this package
from .logging import setup_logging, get_logger, get_log_file_path, tail_log

__all__ = [
	'show_applied_params_banner',
	'create_performance_chart',
	'setup_logging',
	'get_logger',
	'get_log_file_path',
	'tail_log'
]
