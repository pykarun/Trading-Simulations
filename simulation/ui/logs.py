"""Streamlit page to view application logs.

Provides simple controls to tail the app log, filter by keyword and level,
and download the full log file.
"""
import streamlit as st
from pathlib import Path
import os
import glob
import re

from utils.logging import get_default_log_path, tail_log


def get_sorted_log_files() -> list[str]:
    """Finds all log files in the default log directory and sorts them by timestamp (newest first)."""
    log_dir = os.path.dirname(get_default_log_path())
    
    # Find all files matching 'app.log' or 'app.log.YYYY-MM-DD_HH' or 'app.log.YYYY-MM-DD_HH.gz'
    log_files = glob.glob(os.path.join(log_dir, 'app.log*'))

    # Sort files by parsed timestamp in descending order (newest first)
    # The main log file 'app.log' is considered the most recent.
    def get_file_sort_key(file_path):
        filename = os.path.basename(file_path)
        if filename == 'app.log':
            return float('inf') # Highest priority for the active log file
        
        # Extract timestamp from 'app.log.YYYY-MM-DD_HH[.gz]'
        match = re.search(r'app\.log\.(\d{4}-\d{2}-\d{2}_\d{2})(?:\.gz)?$', filename)
        if match:
            # Convert timestamp to a sortable format (e.g., ISO string or datetime object)
            return match.group(1)
        
        return filename # Fallback for other patterns or unsorted files
    
    # Sort by the custom key, reverse for newest first (inf will be first)
    sorted_files = sorted(log_files, key=get_file_sort_key, reverse=True)
    
    return sorted_files


def render_logs():
    st.header("Application Logs")

    all_log_files = get_sorted_log_files()
    
    if not all_log_files:
        st.info("No log files found.")
        return

    # Default to the most recent log file (which is the first one after sorting)
    selected_log_file_name = st.selectbox(
        "Select a log file", 
        [os.path.basename(f) for f in all_log_files],
        index=0
    )
    
    selected_log_path = next((f for f in all_log_files if os.path.basename(f) == selected_log_file_name), None)

    if not selected_log_path:
        st.error("Selected log file not found.")
        return
    
    st.markdown(f"**Viewing log file:** `{selected_log_path}`")

    cols = st.columns([1, 3, 1])
    with cols[0]:
        lines = st.number_input("Lines", min_value=10, max_value=5000, value=200, step=10)
    with cols[1]:
        keyword = st.text_input("Filter keyword (optional)")
    with cols[2]:
        refresh = st.button("Refresh")

    # Read tail of selected log
    txt = tail_log(log_file_path=selected_log_path, lines=lines)

    if keyword:
        filtered = [l for l in txt.splitlines() if keyword.lower() in l.lower()]
        display_text = "\n".join(filtered[-lines:])
    else:
        display_text = txt

    if not display_text:
        st.info("No log entries found for the selected file (or after filtering).")
    else:
        st.code(display_text, language='')

    # Download full log
    try:
        p = Path(selected_log_path)
        if p.exists():
            with p.open('rb') as f:
                data = f.read()
            st.download_button(label=f'Download {selected_log_file_name}', data=data, file_name=selected_log_file_name)
    except Exception as e:
        st.warning(f"Unable to prepare download for {selected_log_file_name}: {e}")
