"""
Configuration settings for the 8-puzzle solver
"""

import os

# Default directory structure
DEFAULT_INPUT_DIR = 'puzzles'
DEFAULT_OUTPUT_DIR = 'results'

# Default puzzle file
DEFAULT_PUZZLE_FILE = 'easy.txt'

# Search configuration
DEFAULT_SEARCH_TYPE = 'ids'  # iterative deepening search
DEFAULT_HEURISTIC = 'top'   # tiles out of place
DEFAULT_EVAL_TYPE = 'u'     # uniform cost

# Plotting configuration
DEFAULT_PLOT_DPI = 300
DEFAULT_FIGURE_SIZE = (12, 8)

# Other constants
MAX_TO_SOLVE = 10

def get_input_path(filename, input_dir=DEFAULT_INPUT_DIR):
    """Get full path to input file"""
    return os.path.join(input_dir, filename)

def get_output_path(filename, output_dir=DEFAULT_OUTPUT_DIR):
    """Get full path to output file"""
    return os.path.join(output_dir, filename)

def ensure_output_dir(output_dir=DEFAULT_OUTPUT_DIR):
    """Ensure output directory exists"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir