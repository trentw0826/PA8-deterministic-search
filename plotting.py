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
from search import SearchNode, run_best_first_search, run_iterative_search
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_FIGURE_SIZE, DEFAULT_PLOT_DPI


def test_algorithms_and_plot(filename='easy.txt', input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Test all combinations of algorithms (u, g, a, ida) with heuristics (top, torc, md) 
    and create a grouped bar chart showing nodes expanded.
    """
    algorithms = ['u', 'g', 'a', 'ida']  # uniform-cost, greedy, A*, IDA*
    heuristics = ['top', 'torc', 'md']  # tiles out of place, tiles out of row/column, manhattan distance
    algorithm_names = ['Uniform-Cost', 'Greedy Best-First', 'A*', 'IDA*']
    heuristic_names = ['Tiles Out of Place', 'Tiles Out of Row/Col', 'Manhattan Distance']
    
    # Results storage: [algorithm][heuristic] = {'nodes': nodes_expanded, 'length': path_length, 'efficiency': nodes/length}
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
                    
            if algorithm == 'ida':
                # IDA* requires A* evaluation function (f = g + h)
                options = Options('ids', heuristic, 'a')
                start_node = SearchNode(0, p, '', options)
                
                # Run IDA* search
                try:
                    nodes_expanded, path_length = run_iterative_search(start_node)
                    if nodes_expanded is not None and path_length > 0:
                        efficiency = nodes_expanded / path_length
                        results[algorithm][heuristic] = {
                            'nodes': nodes_expanded,
                            'length': path_length,
                            'efficiency': efficiency
                        }
                        print(f"  -> Nodes expanded: {nodes_expanded}, Path length: {path_length}, Efficiency: {efficiency:.2f}")
                    else:
                        results[algorithm][heuristic] = {'nodes': float('inf'), 'length': 0, 'efficiency': float('inf')}
                        print(f"  -> No solution found")
                except Exception as e:
                    print(f"  -> Error: {e}")
                    results[algorithm][heuristic] = {'nodes': float('inf'), 'length': 0, 'efficiency': float('inf')}
            else:
                # Regular best-first search algorithms (u, g, a)
                options = Options('bfs', heuristic, algorithm)
                start_node = SearchNode(0, p, '', options)
                
                # Create priority queue and run search
                pq = PriorityQueue()
                pq.put(start_node)
                
                # Run best-first search and capture results
                try:
                    nodes_expanded, path_length = run_best_first_search(pq, options)
                    if nodes_expanded is not None and path_length > 0:
                        efficiency = nodes_expanded / path_length
                        results[algorithm][heuristic] = {
                            'nodes': nodes_expanded,
                            'length': path_length,
                            'efficiency': efficiency
                        }
                        print(f"  -> Nodes expanded: {nodes_expanded}, Path length: {path_length}, Efficiency: {efficiency:.2f}")
                    else:
                        results[algorithm][heuristic] = {'nodes': float('inf'), 'length': 0, 'efficiency': float('inf')}
                        print(f"  -> No solution found")
                except Exception as e:
                    print(f"  -> Error: {e}")
                    results[algorithm][heuristic] = {'nodes': float('inf'), 'length': 0, 'efficiency': float('inf')}
    
    # Create grouped bar chart
    print("\nCreating bar chart...")
    
    # Prepare data for plotting
    x = np.arange(len(algorithms))  # Algorithm positions
    width = 0.2  # Width of bars (reduced to fit 4 algorithms)
    
    # Extract efficiency data for each heuristic
    heur_data = {}
    for heur in heuristics:
        heur_data[heur] = [results[alg][heur]['efficiency'] if results[alg][heur]['efficiency'] != float('inf') else 0 
                          for alg in algorithms]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # Create bars for each heuristic
    bars1 = ax.bar(x - width, heur_data['top'], width, label=heuristic_names[0], alpha=0.8)
    bars2 = ax.bar(x, heur_data['torc'], width, label=heuristic_names[1], alpha=0.8)
    bars3 = ax.bar(x + width, heur_data['md'], width, label=heuristic_names[2], alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Search Algorithms')
    ax.set_ylabel('Nodes Expanded per Solution Step')
    ax.set_title(f'Algorithm Efficiency Comparison on {filename}\n(Nodes Expanded per Solution Step by Algorithm and Heuristic)')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
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
    
    # Print detailed summary table
    print("\nEfficiency Summary (Nodes Expanded per Solution Step):")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Top':<15} {'TORC':<15} {'MD':<15}")
    print("-" * 80)
    for alg_idx, alg in enumerate(algorithms):
        top_eff = f"{results[alg]['top']['efficiency']:.2f}" if results[alg]['top']['efficiency'] != float('inf') else 'N/A'
        torc_eff = f"{results[alg]['torc']['efficiency']:.2f}" if results[alg]['torc']['efficiency'] != float('inf') else 'N/A'
        md_eff = f"{results[alg]['md']['efficiency']:.2f}" if results[alg]['md']['efficiency'] != float('inf') else 'N/A'
        print(f"{algorithm_names[alg_idx]:<20} {top_eff:<15} {torc_eff:<15} {md_eff:<15}")
    
    print(f"\nDetailed Results for {filename}:")
    print("=" * 80)
    for alg_idx, alg in enumerate(algorithms):
        print(f"\n{algorithm_names[alg_idx]}:")
        for heur_idx, heur in enumerate(heuristics):
            result = results[alg][heur]
            if result['nodes'] != float('inf'):
                print(f"  {heuristic_names[heur_idx]:<25}: {result['nodes']:>4} nodes, {result['length']:>2} steps, {result['efficiency']:>6.2f} efficiency")
            else:
                print(f"  {heuristic_names[heur_idx]:<25}: No solution found")
    
    return results