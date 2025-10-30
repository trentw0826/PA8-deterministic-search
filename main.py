"""
Main execution script for the 8-puzzle solver

Code for AI Class Programming Assignment 1
Initial code by Chris Archibald, modified by Trent Welling
archibald@cs.byu.edu, tdw57@byu.edu
"""

import sys
import time
import argparse
from queue import PriorityQueue

from puzzle import Puzzle
from search import SearchNode, run_best_first_search, run_iterative_search
from config import DEFAULT_INPUT_DIR, DEFAULT_SEARCH_TYPE, DEFAULT_HEURISTIC, DEFAULT_EVAL_TYPE, MAX_TO_SOLVE


def get_options(args=sys.argv[1:]):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="8-Puzzle Solver.")
    parser.add_argument('file', metavar='FILENAME', type=str, help="File of puzzles to solve")
    parser.add_argument("-s", '--search', 
                       help="Search type: Options: ids (iterative deepening search) or bfs (best first search)", 
                       default=DEFAULT_SEARCH_TYPE)
    parser.add_argument("-f", '--function', 
                       help="Heuristic function used: Options: top (tiles out of place), torc (tiles out of row/column), or md (manhattan distance)",
                       default=DEFAULT_HEURISTIC)
    parser.add_argument("-t", '--type', 
                       help="Evaluation function type: Options: g (greedy), u (uniform cost), or a (a-star)", 
                       default=DEFAULT_EVAL_TYPE)
    parser.add_argument("--input-dir", 
                       help="Input directory for puzzle files", 
                       default=DEFAULT_INPUT_DIR)
    options = parser.parse_args(args)
    return options


def main():
    """Main function that runs the 8-puzzle solver"""
    # Get command line options
    options = get_options()
    print(options)
    
    # Construct full path to puzzle file
    import os
    puzzle_path = os.path.join(options.input_dir, options.file)
    print('Searching for solutions to puzzles from file: ', puzzle_path)
    
    # Open puzzle file
    pf = open(puzzle_path, 'r')
    
    # Variables to keep track of solving statistics
    num_solved = 0
    exp_num = 0
    tot_time = 0.0
    path_length = 0
    
    for ps in pf.readlines():
        print('Searching to find solution to following puzzle:', ps)
        
        # Create puzzle from file line
        a = [int(i) for i in ps.rstrip()]
        p = Puzzle(a)
        
        # Print the puzzle to the screen
        p.print_puzzle()

        # Create the initial search node corresponding to the given puzzle state
        start_node = SearchNode(0, p, '', options)
                            
        # Create the priority Queue to store the SearchNodes in
        pq = PriorityQueue()
        
        # Insert the initial state into the Queue
        pq.put(start_node)
        
        # Get initial timing info
        start = time.time()
        
        # Run the given search search (each returns number of nodes expanded and the length of the path found)        
        if options.search == 'bfs':
            # Run the best-first searches
            exp, pl = run_best_first_search(pq, options)
        elif options.search == 'ids': 
            # Use this line to run the iterative deepening-search
            exp, pl = run_iterative_search(start_node)
        else:
            print("Search option not valid. Can be bfs or ids")
            sys.exit()

        if exp is None:
            print('PUZZLE NOT SOLVED')
            break
            
        # Keep track of statistics so we can compare search methods
        exp_num += exp
        path_length += pl
        print('Solution path length is : ', pl)
            
        # Calculate Timing info
        end = time.time()
        tot_time += end - start
        num_solved += 1
        
        # Stop after we have solved the specified number of puzzles
        if num_solved >= MAX_TO_SOLVE:
            break
            
    print('Done with solving puzzles.\n\n')
            
    # Print out statistics about this batch
    if num_solved > 0:
        print('Solved', num_solved, 'puzzles from file: ', puzzle_path)
        print('Average nodes expanded: ', float(exp_num) / float(num_solved))
        print('Average search time: ', format(tot_time / num_solved, '.4f'))
        print('Average solution length: ', path_length / num_solved)
    else:
        print('No puzzles were solved')


if __name__ == '__main__':
    main()