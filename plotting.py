"""
Plotting and testing utilities for the 8-puzzle problem
"""

import matplotlib
import os

# Set backend based on environment - check for GUI availability
if 'DISPLAY' in os.environ:
    # GUI environment detected, try interactive backends
    try:
        import tkinter
        matplotlib.use('TkAgg')
        print("Using TkAgg backend for interactive plots")
    except ImportError:
        try:
            import PyQt5
            matplotlib.use('Qt5Agg')
            print("Using Qt5Agg backend for interactive plots")
        except ImportError:
            matplotlib.use('Agg')
            print("No GUI toolkit available, using Agg backend (file output only)")
else:
    # No display available, use non-interactive backend
    matplotlib.use('Agg')
    print("No display detected, using Agg backend (file output only)")

import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue

from puzzle import Puzzle
from search import SearchNode, run_best_first_search
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_FIGURE_SIZE, DEFAULT_PLOT_DPI


def test_algorithms_and_plot(filename='easy.txt', input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Test all combinations of algorithms (u, g, a) with heuristics (top, torc, md) 
    and create a grouped bar chart showing nodes expanded.
    """
    algorithms = ['u', 'g', 'a']  # uniform-cost, greedy, A*
    heuristics = ['top', 'torc', 'md']  # tiles out of place, tiles out of row/column, manhattan distance
    algorithm_names = ['Uniform-Cost', 'Greedy Best-First', 'A*']
    heuristic_names = ['Tiles Out of Place', 'Tiles Out of Row/Col', 'Manhattan Distance']
    
    # Results storage: [algorithm][heuristic] = nodes_expanded
    results = {}
    
    # Construct full path to input file
    puzzle_path = os.path.join(input_dir, filename)
    
    print(f"Testing all algorithm-heuristic combinations on {puzzle_path}")
    print("=" * 60)
    
    # Read first puzzle from file
    with open(puzzle_path, 'r') as pf:
        puzzle_line = pf.readline().strip()
        
    puzzle_array = [int(i) for i in puzzle_line]
    
    for alg_idx, algorithm in enumerate(algorithms):
        results[algorithm] = {}
        
        for heur_idx, heuristic in enumerate(heuristics):
            print(f"Running {algorithm_names[alg_idx]} with {heuristic_names[heur_idx]}...")
            
            # Create puzzle and search node
            p = Puzzle(puzzle_array)
            
            # Create options object manually
            class Options:
                def __init__(self, search_type, heuristic_func, eval_type):
                    self.search = search_type
                    self.function = heuristic_func
                    self.type = eval_type
                    
            options = Options('bfs', heuristic, algorithm)
            start_node = SearchNode(0, p, '', options)
            
            # Create priority queue and run search
            pq = PriorityQueue()
            pq.put(start_node)
            
            # Run best-first search and capture results
            try:
                nodes_expanded, path_length = run_best_first_search(pq, options)
                if nodes_expanded is not None:
                    results[algorithm][heuristic] = nodes_expanded
                    print(f"  -> Nodes expanded: {nodes_expanded}, Path length: {path_length}")
                else:
                    results[algorithm][heuristic] = float('inf')  # No solution found
                    print(f"  -> No solution found")
            except Exception as e:
                print(f"  -> Error: {e}")
                results[algorithm][heuristic] = float('inf')
    
    # Create grouped bar chart
    print("\nCreating bar chart...")
    
    # Prepare data for plotting
    x = np.arange(len(algorithms))  # Algorithm positions
    width = 0.25  # Width of bars
    
    # Extract data for each heuristic
    heur_data = {}
    for heur in heuristics:
        heur_data[heur] = [results[alg][heur] if results[alg][heur] != float('inf') else 0 
                          for alg in algorithms]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # Create bars for each heuristic
    bars1 = ax.bar(x - width, heur_data['top'], width, label=heuristic_names[0], alpha=0.8)
    bars2 = ax.bar(x, heur_data['torc'], width, label=heuristic_names[1], alpha=0.8)
    bars3 = ax.bar(x + width, heur_data['md'], width, label=heuristic_names[2], alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Search Algorithms')
    ax.set_ylabel('Nodes Expanded')
    ax.set_title(f'Algorithm Performance Comparison on {filename}\n(Nodes Expanded by Algorithm and Heuristic)')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    # Save plot to output directory
    plot_filename = f'algorithm_comparison_{filename.replace(".txt", "")}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(plot_path, dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
    print(f"Plot saved as: {plot_path}")
    
    # Try to show plot (will work in GUI environments)
    backend = matplotlib.get_backend()
    if backend != 'Agg':
        try:
            print("Displaying plot in window...")
            plt.show()
        except Exception as e:
            print(f"Could not display plot window: {e}")
            print("Plot is available as saved file.")
    else:
        print("Non-interactive backend detected. Plot saved to file only.")
    
    # Print summary table
    print("\nSummary Results:")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Top':<10} {'TORC':<10} {'MD':<10}")
    print("-" * 60)
    for alg_idx, alg in enumerate(algorithms):
        top_val = results[alg]['top'] if results[alg]['top'] != float('inf') else 'N/A'
        torc_val = results[alg]['torc'] if results[alg]['torc'] != float('inf') else 'N/A'
        md_val = results[alg]['md'] if results[alg]['md'] != float('inf') else 'N/A'
        print(f"{algorithm_names[alg_idx]:<20} {str(top_val):<10} {str(torc_val):<10} {str(md_val):<10}")
    
    return results