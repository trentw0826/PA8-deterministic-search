# 8-Puzzle Search Algorithm Solver

A comprehensive implementation of multiple search algorithms for solving the classic 8-puzzle problem. This project compares the performance of different search strategies (IDA*, Uniform-Cost, Greedy Best-First, and A*) using various heuristic functions.

## Overview

The 8-puzzle is a [sliding puzzle](https://en.wikipedia.org/wiki/Sliding_puzzle) consisting of a 3×3 grid with 8 numbered tiles and one empty space. The goal is to rearrange the tiles from a given initial configuration to reach the solved state:

```
0 1 2
3 4 5
6 7 8
```

Where `0` represents the blank (empty) space.

## Features

- **Multiple Search Algorithms:**

  - IDA* (Iterative Deepening A*)
  - Uniform-Cost Search
  - Greedy Best-First Search
  - A\* Search

- **Four Heuristic Functions:**

  - Tiles Out of Place: Counts tiles not in their goal position
  - Tiles Out of Row/Column: Counts tiles in wrong row plus tiles in wrong column
  - Manhattan Distance: Sum of distances each tile must travel to reach its goal position
  - Manhattan Distance + Linear Conflicts: Enhanced Manhattan Distance with conflict detection

- **Performance Analysis:**

  - Algorithm comparison visualization
  - Statistical analysis of nodes expanded and solution path lengths
  - Grouped bar charts comparing all algorithm-heuristic combinations

- **Modular Architecture:**
  - Clean separation of concerns across multiple modules
  - Configurable parameters through `config.py`
  - Extensible design for adding new algorithms or heuristics

## Usage

### Basic Puzzle Solving

Solve puzzles using the main solver:

```bash
# Basic usage with default settings (IDA*, tiles out of place heuristic, A* evaluation)
python main.py easy.txt

# Specify search algorithm, heuristic, and evaluation function
python main.py medium.txt -s bfs -f md -t a

# Use IDA* with Manhattan Distance heuristic
python main.py hard.txt --search ids --function md --type a

# Use IDA* with the most advanced heuristic (Manhattan Distance + Linear Conflicts)
python main.py hard.txt --search ids --function mdlc --type a

# Compare IDA* vs Basic Iterative Deepening on the same puzzle
python main.py medium.txt --search ids --function md --type a    # IDA*
python main.py medium.txt --search bids --function top --type u  # Basic IDS
```

#### Command Line Arguments

- `filename`: Puzzle file to solve (required)
- `-s, --search`: Search type - `ids` (IDA* - Iterative Deepening A*), `bids` (Basic Iterative Deepening Search), or `bfs` (best-first)
- `-f, --function`: Heuristic function - `top` (tiles out of place), `torc` (tiles out of row/column), `md` (manhattan distance), or `mdlc` (manhattan distance + linear conflicts)
- `-t, --type`: Evaluation function - `u` (uniform cost), `g` (greedy), or `a` (A\*)
- `--input-dir`: Input directory for puzzle files (default: `puzzles`)

**Notes:**

- For optimal IDA* performance, use with evaluation function type `a` (A*) to ensure f(n) = g(n) + h(n).
- Use `bids` to run the basic iterative deepening search for comparison with IDA\*.

### Algorithm Performance Comparison

Run comprehensive performance tests and generate visualization:

```bash
# Test all algorithm combinations on easy puzzles
python test_algorithms.py easy.txt

# Test on different difficulty levels
python test_algorithms.py medium.txt
python test_algorithms.py hard.txt

# List available puzzle files
python test_algorithms.py --list-files

# Specify custom input/output directories
python test_algorithms.py easy.txt --input-dir puzzles --output-dir results
```

This generates:

- Detailed efficiency statistics printed to console (nodes expanded per solution step)
- Grouped bar chart visualization saved as PNG showing algorithm efficiency
- Comparison of search efficiency across all algorithm-heuristic combinations
- Includes IDA* (Iterative Deepening A*) alongside traditional best-first search algorithms
- Both raw performance metrics and normalized efficiency ratios for comprehensive analysis

## Search Strategies

1. **IDA\* (Iterative Deepening A\*)**

   - Complete and optimal search algorithm
   - Uses f-value thresholds (f = g + h) with increasing limits
   - Memory efficient like iterative deepening but with A\* evaluation
   - Combines benefits of A\* optimality with linear memory usage
   - Ideal for problems where memory is limited but optimal solutions are required

2. **Basic Iterative Deepening Search (bids)**

   - Complete and optimal search algorithm (available for comparison)
   - Uses depth limits with increasing values (blind search)
   - Memory efficient but may expand significantly more nodes than IDA\*
   - Provided as an alternative to compare performance with IDA\*
   - Useful for understanding the benefits of informed vs uninformed search

3. **Uniform-Cost Search**

   - Uses f(n) = g(n) where g(n) is the path cost
   - Optimal but may expand many nodes
   - Equivalent to Dijkstra's algorithm

4. **Greedy Best-First Search**

   - Uses f(n) = h(n) where h(n) is the heuristic value
   - Fast but not guaranteed to find optimal solution
   - Focuses purely on heuristic guidance

5. **A\* Search**
   - Uses f(n) = g(n) + h(n)
   - Optimal when heuristic is admissible
   - Best balance of optimality and efficiency

## Heuristic Functions

1. **Tiles Out of Place (top)**

   - Counts number of tiles not in goal position
   - Admissible
   - Simple but effective

2. **Tiles Out of Row/Column (torc)**

   - Counts tiles in wrong row plus tiles in wrong column
   - Admissible
   - More informed than tiles out of place

3. **Manhattan Distance (md)**
   - Sum of distances each tile must travel to reach goal
   - Admissible
   - Well-informed heuristic that often leads to good performance

4. **Manhattan Distance + Linear Conflicts (mdlc)**
   - Manhattan Distance enhanced with linear conflict detection
   - When tiles are in correct row/column but need to pass each other, adds 2 for each conflict
   - Admissible and more informed than basic Manhattan Distance
   - Detects situations where tiles must move out of each other's way
   - Often provides the best performance by accounting for tile interaction patterns

## Puzzle File Format

Puzzle files contain one puzzle per line, where each line is 9 digits representing the 3×3 grid read left-to-right, top-to-bottom:

```
325610748  # Represents:
           # 3 2 5
           # 6 1 0  (0 is the blank space)
           # 7 4 8
```

## Results/Analysis

### Simple Iterative Deepening Results (Covers Task 1.2)

| Difficulty | Puzzle Count | Avg Nodes Expanded | Avg Time (sec) | Avg Solution Length |
| ---------- | ------------ | ------------------ | -------------- | ------------------- |
| Easy       | 10           | 334.1              | 0.0049         | 7.0                 |
| Medium     | 10           | 28,269.0           | 0.1779         | 15.0                |
| Hard       | 10           | 695,113.1          | 5.2693         | 21.0                |
| Random     | 10           | 1,087,364.8        | 7.0974         | 16.4                |
| Worst      | \*           | -                  | -              | -                   |

_\*Worst puzzles were not computed in a reasonable time_

### Analysis of 4 Algorithms across 3 Heuristics (Covers Task 1.5)

After production and analysis of raw search data (nodes expanded, path lengths discovered, memory usage, etc.) some significant patterns emerged. Some include:

- Uniform cost consistently expanded the most nodes per solution step (least efficient) when the solution lengths were smaller but IDA*'s expansions skyrocketed to overtake UCS as lengths increased. This pattern makes sense as UCS expands each node at most once and saves intermediary paths in memory (an overhead that's noticeable when solution lengths are small) whereas IDA* iteratively re-calculates an exponential number of expansions in order to save on memory. 
- UCS' efficiency was not affected by the heuristic given that it relies only on path length to choose expansions
- Greedy search provided the only non-optimal paths but was incredibly quick and very memory efficient for the most informed heuristic
- All algorithms that relied on heuristics demonstrated that manhattan distance was the most informed by consistently leading to solutions quicker
- Both IDA* and Greedy occupied almost no memory compared to the enormous amount taken by UCS. IDA* only ever keeps track of a relatively constant path at once while Greedy consistently found solutions quickly enough to keep path cache low.

![random efficiency results](/results/efficiency_comparison_random.png)
![random memory analysis](/results/memory_analysis_random.png)

### Experience Implementing IDA* (Covers Part 2)

The implementation of IDA* was largely straightforward. Given the existing pattern of iterative searching implemented for IDS, it wasn't too complicated to extend the idea to limit searches by an f-value. The most difficult logic was making sure that nodes that were explored and didn't turn up a solution were properly de-considered from the active path and cost. Additionally, returning the proper f-value (the minimum value that exceeded the current threshold) required some creativity in tracking across multiple recursive calls to the exploration function. In implementing this function and observing the results I learned IDA* can dramatically improve the relationship between finding a valid solution and the time/memory cost for doing so. On a broader level, keeping track of creatively-chosen cached values can bridge the gap between orders of complexity in algorithms processing large amounts of data.

## Configuration

Modify `config.py` to adjust:

- Default directories for input/output files
- Search algorithm and heuristic defaults (current: IDA\*, tiles out of place, uniform cost)
- Maximum number of puzzles to solve
- Plot appearance settings (DPI, figure size)

## Academic Context

This implementation was developed for an AI course programming assignment focused on:

- Understanding different search strategies in node spaces
- Comparing the effectiveness of various heuristics
- Analyzing time and space complexity trade-offs between algorithms/heuristics
- Implementing clean, modular code architecture

## Authors

- Initial code by Chris Archibald (archibald@cs.byu.edu)
- Modified and extended by Trent Welling (tdw57@byu.edu)

## License

This code is for educational purposes as part of an AI course assignment.

## AI Disclaimer

Both testing logic and README partially generated with Claude Sonnet 4.0
