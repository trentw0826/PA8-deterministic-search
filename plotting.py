"""
Refactored plotting and testing utilities for the 8-puzzle problem.
Clean, modular approach for algorithm comparison and analysis.
"""

import matplotlib
import os

# Set backend based on environment - check for GUI availability
if 'DISPLAY' in os.environ:
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
    matplotlib.use('Agg')
    print("No display detected, using Agg backend (file output only)")

import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
import tracemalloc
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from puzzle import Puzzle
from search import SearchNode, run_best_first_search, run_iterative_search
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_FIGURE_SIZE, DEFAULT_PLOT_DPI

# Clean data structures for results
@dataclass
class PerformanceResult:
    """Clean data structure for algorithm performance results"""
    algorithm: str
    heuristic: str  
    nodes_expanded: int
    solution_length: int
    efficiency: float  # nodes per solution step
    memory_mb: float
    memory_per_step: float
    solved: bool
    
    def __str__(self):
        if not self.solved:
            return f"{self.algorithm} + {self.heuristic}: FAILED"
        return f"{self.algorithm} + {self.heuristic}: {self.nodes_expanded} nodes, {self.solution_length} steps, {self.efficiency:.2f} efficiency"

class AlgorithmTester:
    """Clean, modular algorithm testing system"""
    
    # Algorithm and heuristic configurations
    ALGORITHMS = {
        'u': 'Uniform-Cost',
        'g': 'Greedy Best-First', 
        'a': 'A*',
        'ida': 'IDA*'
    }
    
    HEURISTICS = {
        'top': 'Tiles Out of Place',
        'torc': 'Tiles Out of Row/Col',
        'md': 'Manhattan Distance'
    }
    
    def __init__(self, puzzle_file: str, input_dir: str = DEFAULT_INPUT_DIR):
        """Initialize tester with puzzle file"""
        self.puzzle_file = puzzle_file
        self.input_dir = input_dir
        self.puzzle_array = self._load_puzzle()
        self.results: List[PerformanceResult] = []
    
    def _load_puzzle(self) -> List[int]:
        """Load first puzzle from file"""
        puzzle_path = os.path.join(self.input_dir, self.puzzle_file)
        with open(puzzle_path, 'r') as f:
            puzzle_line = f.readline().strip()
        return [int(i) for i in puzzle_line]


    def _run_single_algorithm(self, algorithm: str, heuristic: str) -> PerformanceResult:
        """Run a single algorithm-heuristic combination with clean memory tracking"""
        # Start memory tracking
        tracemalloc.start()
        
        # Create puzzle and options
        puzzle = Puzzle(self.puzzle_array)
        
        # Create options object
        class Options:
            def __init__(self, search_type, heuristic_func, eval_type):
                self.search = search_type
                self.function = heuristic_func
                self.type = eval_type
        
        try:
            if algorithm == 'ida':
                # IDA* uses iterative deepening with A* evaluation
                options = Options('ids', heuristic, 'a')
                start_node = SearchNode(0, puzzle, '', options)
                nodes_expanded, path_length = run_iterative_search(start_node)
            else:
                # Best-first search algorithms
                options = Options('bfs', heuristic, algorithm)
                start_node = SearchNode(0, puzzle, '', options)
                pq = PriorityQueue()
                pq.put(start_node)
                nodes_expanded, path_length = run_best_first_search(pq, options)
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            # Check if solution was found
            if nodes_expanded is not None and path_length > 0:
                efficiency = nodes_expanded / path_length
                memory_per_step = peak_mb / path_length if path_length > 0 else 0
                
                return PerformanceResult(
                    algorithm=self.ALGORITHMS[algorithm],
                    heuristic=self.HEURISTICS[heuristic],
                    nodes_expanded=nodes_expanded,
                    solution_length=path_length,
                    efficiency=efficiency,
                    memory_mb=peak_mb,
                    memory_per_step=memory_per_step,
                    solved=True
                )
            else:
                tracemalloc.stop()
                return PerformanceResult(
                    algorithm=self.ALGORITHMS[algorithm],
                    heuristic=self.HEURISTICS[heuristic],
                    nodes_expanded=0,
                    solution_length=0,
                    efficiency=float('inf'),
                    memory_mb=0,
                    memory_per_step=0,
                    solved=False
                )
                
        except Exception as e:
            tracemalloc.stop()
            return PerformanceResult(
                algorithm=self.ALGORITHMS[algorithm],
                heuristic=self.HEURISTICS[heuristic],
                nodes_expanded=0,
                solution_length=0,
                efficiency=float('inf'),
                memory_mb=0,
                memory_per_step=0,
                solved=False
            )
    
    def run_all_tests(self) -> List[PerformanceResult]:
        """Run all algorithm-heuristic combinations"""
        print(f"\nTesting all algorithms on {self.puzzle_file}")
        print("=" * 60)
        
        results = []
        
        for algorithm_key in self.ALGORITHMS.keys():
            for heuristic_key in self.HEURISTICS.keys():
                alg_name = self.ALGORITHMS[algorithm_key]
                heur_name = self.HEURISTICS[heuristic_key]
                
                print(f"{alg_name} + {heur_name}...")
                result = self._run_single_algorithm(algorithm_key, heuristic_key)
                results.append(result)
                
                # Clean, simple output
                if result.solved:
                    print(f"   {result.nodes_expanded:>5} nodes | {result.solution_length:>2} steps | {result.efficiency:>6.2f} efficiency | {result.memory_mb:>6.3f}MB")
                else:
                    print(f"   Failed to solve")
                    
        self.results = results
        return results


class ResultAnalyzer:
    """Clean analysis and visualization of performance results"""
    
    def __init__(self, results: List[PerformanceResult], puzzle_file: str):
        self.results = results
        self.puzzle_file = puzzle_file
        self.solved_results = [r for r in results if r.solved]
    
    def print_summary(self):
        """Print clean, organized terminal summary"""
        print(f"\nPERFORMANCE SUMMARY - {self.puzzle_file}")
        print("=" * 70)
        
        if not self.solved_results:
            print("No algorithms successfully solved the puzzle")
            return
            
        # Best performers in each category
        best_efficiency = min(self.solved_results, key=lambda r: r.efficiency)
        best_memory = min(self.solved_results, key=lambda r: r.memory_per_step)
        fewest_nodes = min(self.solved_results, key=lambda r: r.nodes_expanded)
        
        print(f"CHAMPIONS:")
        print(f"   Most Efficient:     {best_efficiency.algorithm} + {best_efficiency.heuristic} ({best_efficiency.efficiency:.1f} nodes/step)")
        print(f"   Memory Efficient:   {best_memory.algorithm} + {best_memory.heuristic} ({best_memory.memory_per_step:.4f} MB/step)")
        print(f"   Fewest Nodes:       {fewest_nodes.algorithm} + {fewest_nodes.heuristic} ({fewest_nodes.nodes_expanded} nodes)")
        
        # Organized results by algorithm
        print(f"\nDETAILED RESULTS:")
        print("-" * 70)
        
        algorithms = {}
        for result in self.solved_results:
            if result.algorithm not in algorithms:
                algorithms[result.algorithm] = []
            algorithms[result.algorithm].append(result)
        
        for alg_name in ['Uniform-Cost', 'Greedy Best-First', 'A*', 'IDA*']:
            if alg_name in algorithms:
                print(f"\n{alg_name}:")
                alg_results = sorted(algorithms[alg_name], key=lambda r: r.efficiency)
                for result in alg_results:
                    print(f"   {result.heuristic:20} | {result.nodes_expanded:>6} nodes | {result.efficiency:>6.1f} eff | {result.memory_per_step:>7.4f} MB/step")
            else:
                print(f"\n{alg_name}: Failed to solve")
    
    def create_visualizations(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        """Create clean, high-relevance visualizations"""
        if not self.solved_results:
            print("Cannot create visualizations - no successful solutions")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comprehensive performance comparison
        self._create_efficiency_comparison(output_dir)
        self._create_memory_comparison(output_dir)
        
    def _create_efficiency_comparison(self, output_dir: str):
        """Create clean efficiency comparison chart"""
        # Organize data by algorithm and heuristic
        alg_order = ['Uniform-Cost', 'Greedy Best-First', 'A*', 'IDA*']
        heur_order = ['Tiles Out of Place', 'Tiles Out of Row/Col', 'Manhattan Distance']
        
        # Create matrix of efficiency values
        efficiency_matrix = []
        for alg in alg_order:
            alg_row = []
            for heur in heur_order:
                # Find result for this combination
                result = next((r for r in self.solved_results if r.algorithm == alg and r.heuristic == heur), None)
                alg_row.append(result.efficiency if result else 0)
            efficiency_matrix.append(alg_row)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(alg_order))
        width = 0.25
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, heur in enumerate(heur_order):
            values = [row[i] for row in efficiency_matrix]
            bars = ax.bar(x + i * width - width, values, width, 
                         label=heur, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Search Algorithms', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nodes Expanded per Solution Step', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Efficiency Comparison\n{self.puzzle_file}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(alg_order)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'efficiency_comparison_{self.puzzle_file.replace(".txt", "")}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
        print(f"Efficiency chart saved: {plot_path}")
        
        # Try to show plot
        self._try_show_plot()
        plt.close()
    
    def _create_memory_comparison(self, output_dir: str):
        """Create clean, readable memory usage comparison chart"""
        # Create a horizontal bar chart showing memory per step, grouped by algorithm
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Organize data by algorithm and heuristic
        alg_order = ['IDA*', 'A*', 'Greedy Best-First', 'Uniform-Cost']  # Best to worst memory efficiency
        heur_order = ['Manhattan Distance', 'Tiles Out of Row/Col', 'Tiles Out of Place']
        heur_short = ['MD', 'TORC', 'TOP']
        
        # Colors for each heuristic
        heur_colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Left plot: Memory per step (horizontal bars)
        y_positions = []
        y_labels = []
        
        for alg_idx, alg in enumerate(alg_order):
            for heur_idx, heur in enumerate(heur_order):
                result = next((r for r in self.solved_results if r.algorithm == alg and r.heuristic == heur), None)
                if result:
                    y_pos = alg_idx * 4 + heur_idx
                    y_positions.append(y_pos)
                    y_labels.append(f"{alg}\n{heur_short[heur_idx]}")
                    
                    bar = ax1.barh(y_pos, result.memory_per_step, 
                                  color=heur_colors[heur_idx], alpha=0.8, height=0.8)
                    
                    # Add value labels
                    ax1.text(result.memory_per_step + max([r.memory_per_step for r in self.solved_results]) * 0.02,
                            y_pos, f'{result.memory_per_step:.4f}',
                            va='center', fontsize=9, fontweight='bold')
        
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(y_labels, fontsize=10)
        ax1.set_xlabel('Memory per Solution Step (MB)', fontsize=12, fontweight='bold')
        ax1.set_title('Memory Efficiency by Algorithm\n', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()  # Best performers at top
        
        # Add algorithm separators
        for i in range(1, len(alg_order)):
            ax1.axhline(y=i*4-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Right plot: Total memory usage (stacked bars)
        alg_memory_data = {}
        for alg in alg_order:
            alg_results = [r for r in self.solved_results if r.algorithm == alg]
            if alg_results:
                alg_memory_data[alg] = [
                    next((r.memory_mb for r in alg_results if r.heuristic == heur), 0)
                    for heur in heur_order
                ]
        
        x = np.arange(len([alg for alg in alg_order if alg in alg_memory_data]))
        width = 0.6
        
        bottom = np.zeros(len(x))
        for heur_idx, heur in enumerate(heur_order):
            values = [alg_memory_data[alg][heur_idx] for alg in alg_order if alg in alg_memory_data]
            bars = ax2.bar(x, values, width, bottom=bottom, 
                          label=heur_short[heur_idx], color=heur_colors[heur_idx], alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, 
                            bottom[i] + value/2,
                            f'{value:.2f}MB', ha='center', va='center', 
                            fontsize=9, fontweight='bold', color='white')
            bottom += values
        
        ax2.set_xlabel('Search Algorithms', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Total Memory Consumption\n', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([alg for alg in alg_order if alg in alg_memory_data], rotation=45)
        ax2.legend(title='Heuristics', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'memory_analysis_{self.puzzle_file.replace(".txt", "")}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
        print(f"Memory analysis saved: {plot_path}")
        
        self._try_show_plot()
        plt.close()
    
    def _try_show_plot(self):
        """Try to display plot if GUI available"""
        backend = matplotlib.get_backend()
        if backend != 'Agg':
            try:
                plt.show()
            except Exception:
                pass


def test_algorithms_and_plot(filename='easy.txt', input_dir=DEFAULT_INPUT_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Refactored main testing function with clean, modular approach
    """
    # Run all tests
    tester = AlgorithmTester(filename, input_dir)
    results = tester.run_all_tests()
    
    # Analyze and display results
    analyzer = ResultAnalyzer(results, filename)
    analyzer.print_summary()
    analyzer.create_visualizations(output_dir)
    
    print(f"\nTesting complete! Results saved to {output_dir}")
    
    return results