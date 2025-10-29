from __future__ import print_function
from queue import PriorityQueue
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np 

# Code for AI Class Programming Assignment 1
# Written by Chris Archibald, modified by Trent Welling
# archibald@cs.byu.edu, tdw57@byu.edu

# GOAL: (0 is the blank tile)
# 0 1 2
# 3 4 5
# 6 7 8

class Puzzle():
    """
    8-puzzle class
    """

    SOLUTION_STATE = [0,1,2,3,4,5,6,7,8]

    def __init__(self, arrangement):
        """
        The state (and arrangement passed in) is a list of length 9, that stores which tile is in each place
        In a solved puzzle, each place number holds the tile of the same number
        i.e. solution is state = [0,1,2,3,4,5,6,7,8]
        """
        #TODO input validation
        self.state = arrangement[:]
        self.blank = None

        for i in range(len(self.state)):
            if self.state[i] == 0:
                self.blank = i

    def print_puzzle(self):
        """
        Print a visual description of the puzzle to the output
        """
        k = 0
        for i in range(3):
            for j in range(3):
                print('', end="") 
                print(self.state[k], end="")
                k = k + 1
            print('')
        
    def get_moves(self):
        """
        The moves correspond to the motion of the tile into the blank space
        it can be U (up), D (down), R (right), or L (left)
        """
        invalid_moves = []
        if self.blank < 3:
            invalid_moves.append('D')
        if self.blank > 5:
            invalid_moves.append('U')
        if self.blank % 3 == 0:
            invalid_moves.append('R')
        if self.blank % 3 == 2:
            invalid_moves.append('L')

        base_moves = ['U', 'D', 'L', 'R']
        valid_moves = []
        for m in base_moves:
            if m not in invalid_moves:
                valid_moves.append(m)

        return valid_moves

    def do_move(self, move):
        """
        Modify the state by performing the given move.
        This assumes that the move is valid
        """
        swapi = 0
        if move == 'U':
            swapi = self.blank + 3
        if move == 'D':
            swapi = self.blank - 3
        if move == 'L':
            swapi = self.blank + 1
        if move == 'R':
            swapi = self.blank - 1

        temp = self.state[swapi]
        self.state[swapi] = self.state[self.blank]
        self.state[self.blank] = temp
        self.blank = swapi

    def undo_move(self, move):
        """
        This modifies the state by undoing the move.  For use in recursive search
        Assumes the move was a valid one
        """
        swapi = 0
        if move == 'D':
            swapi = self.blank + 3
        if move == 'U':
            swapi = self.blank - 3
        if move == 'R':
            swapi = self.blank + 1
        if move == 'L':
            swapi = self.blank - 1

        temp = self.state[swapi]
        self.state[swapi] = self.state[self.blank]
        self.state[self.blank] = temp
        self.blank = swapi
        
    def is_solved(self):
        """
        Returns True if the puzzle is solved, False otherwise
        """
        return self.state == Puzzle.SOLUTION_STATE


    def __repr__(self):
        return "".join([str(i) for i in self.state])

    def id(self):
        """
        Returns the string representation of this puzzle's state. 
        Useful for storing state in a dictionary
        """
        return self.__repr__()

class SearchNode():
    """
    Our search node class
    """
    def __init__(self,cost,puzzle,path,options):
        """
        Initialize all the relevant parts of the search node
        """
        self.cost = cost
        self.puzzle = puzzle
        self.path = path
        self.options = options
        self.h = heuristic(self,self.options)
        self.compute_f_value()
       
    def compute_f_value(self):
        """
        Compute the f-value for this node
        """

        self.h = heuristic(self, self.options)
        self.f_value = 0

        if self.options.type == 'g':
            #greedy search algorithm: f(n) = h(n)
            self.f_value = self.h

        elif self.options.type == 'u':
            #uniform cost search algorithm: f(n) = g(n)
            self.f_value = self.cost

        elif self.options.type == 'a':
            #A* search algorithm: f(n) = g(n) + h(n)
            self.f_value = self.cost + self.h

        else:
            print('Invalid search type (-t) selected: Valid options are g, u, and a')
            sys.exit()


    #Comparison operator 
    def __lt__(self,other):
        """
        Comparison operator so that nodes will be sorted in priority queue based on f-value
        """
        return self.f_value < other.f_value	

def heuristic(node,options):
    """
    This is the function that is called from the SearchNode class to get the heuristic value for a node
    """
    if options.function == 'top':
        return tiles_out_of_place(node.puzzle)
    elif options.function == 'torc':
        return tiles_out_of_row_column(node.puzzle)
    elif options.function == 'md':
        return manhattan_distance_to_goal(node.puzzle)
    else:
        print('Invalid heristic selected. Options are top, torc, and md')
        sys.exit()    

def tiles_out_of_place(puzzle):
    """
    This heuristic counts the number of tiles out of place.    
    """
    #Keep track of the number of tiles out of place
    num_out_of_place = 0
    
    #Cycle through all of the places in the puzzle and see if the right tile is there
    # (We ignore place 0 since that is where the blank tile goes and we shouldn't count it)
    for i in range(1,len(puzzle.state)):
    
        # The tile in place i ( puzzle.state[i] ) should be tile i.  
        # If it isn't increment out of place counter
        # (To compare tile (string) with place (int), we must first convert from string to int as such:
        #  int(puzzle.state[i])
    
        if puzzle.state[i] != i:
            num_out_of_place += 1
 
    return num_out_of_place
    
    
def tiles_out_of_row_column(puzzle):
    """
    This heuristic counts the number of tiles that are in the wrong row, 
    the number of tiles that are in the wrong column
    and returns the sum of these two numbers.
    Remember not to count the blank tile as being out of place, or the heuristic is inadmissible
    """

    wrong_rows = 0
    wrong_cols = 0

    for idx, tile in enumerate(puzzle.state):
        if tile == 0:
            continue
        if get_tile_row(idx) != get_tile_row(tile):
            wrong_rows += 1
        if get_tile_column(idx) != get_tile_column(tile):
            wrong_cols += 1

    return wrong_rows + wrong_cols

    

def manhattan_distance_to_goal(puzzle):
    """
    This heuristic should calculate the sum of all the manhattan distances for each tile to get to 
    its goal position.  Again, make sure not to include the distance from the blank to its goal.
    """
    
    total_distance = 0

    for idx, tile in enumerate(puzzle.state):
        if tile == 0:
            continue
        current_row = get_tile_row(idx)
        current_col = get_tile_column(idx)
        goal_row = get_tile_row(tile)
        goal_col = get_tile_column(tile)
        total_distance += abs(current_row - goal_row) + abs(current_col - goal_col)

    return total_distance

    
def get_tile_row(tile):
    """
    Return the row of the given tile location (Helper function for you to use)
    """
    return int(tile / 3)

def get_tile_column(tile):
    """
    Return the column of the given tile location (Helper function for you to use)
    """
    return tile % 3    
    
def run_iterative_search(start_node):
    """
    This runs an iterative deepening search
    It caps the depth of the search at 40 (no 8-puzzles have solutions this long)
    """
    #Our initial depth limit
    depth_limit = 1
    
    #Maximum depth limit
    max_depth_limit = 40
    
    #Keep track of the total number of nodes we expand
    total_expanded = 0
    
    #Keep trying until our depth limit hits 40
    while depth_limit < max_depth_limit:
        
        #Store visited nodes along the current search path
        visited = dict()
        visited['N'] = 0

        #Mark the initial state as visited
        visited[start_node.puzzle.id()] = True
        
        #Run depth-limited search starting at initial node (which points to initial state)
        path_length = run_dfs(start_node, depth_limit, visited) 
    
        #See how many nodes we expanded on this iteration and add it to our total
        total_expanded += visited['N']
        
        #Check to see if a solution was found
        if path_length is not None:
            #It was! Print out information and return the search stats
            print('Expanded ', total_expanded, 'nodes')
            print('IDS Found solution at depth', depth_limit)
            return total_expanded, path_length
            
        # No solution was found at this depth limit, so increment our depth-limit    
        depth_limit += 1
        
    # No solution was found at any depth-limit, so return None,None (Which signifies no solution found)
    return None, None
    
def run_dfs(node, depth_limit, visited):
    """
    Recursive Depth-Limited Search:  
    
    Check node to see if it is goal, if it is, print solution and return path length
    If not and if depth-limit hasn't been reached, recurse on all children
    """
    visited['N'] = visited['N'] + 1 #Increment our node expansion counter

    # Check to see if this is a goal node
    if node.puzzle.is_solved():
        # It is! Print out solution and return solution length
        print('Iterative Deepening SOLVED THE PUZZLE! SOLUTION = ', node.path)
        return len(node.path)
        
    # Check to see if the depth limit has been reached (number of actions that have been taken)
    if len(node.path) >= depth_limit:
        # It has. Return None, signifying that no path was found
        return None
    
    # Generate successors and recurse on them
    
    # Get the list of moves we can try from this node's state
    moves = node.puzzle.get_moves()
    
    # For each possible move
    for m in moves:
        #Execute the move/action
        node.puzzle.do_move(m)
        node.compute_f_value()


        #Add this move to the node's path
        node.path = node.path + m
        #Add 1 to node's cost
        node.cost = node.cost + 1
        #Check to see if we have already visited this node
        if node.puzzle.id() not in visited:
            #We haven't. Now we will, so add it to visited
            visited[node.puzzle.id()] = True
            
            #Recurse on this new state
            path_length = run_dfs(node, depth_limit, visited)    
            
            #Check to see if a solution was found down this path (return value of None means no)
            if path_length is not None:
                #It was! Return this solution path length to whoever called us
                return path_length
                
            #Remove this state from the visited list.  We only check for duplicates along current search path
            del visited[node.puzzle.id()]

        # That move didn't lead to a solution, so lets try the next one
        # First, though, we need to undo the move (to return puzzle to state before we tried that move)
        node.puzzle.undo_move(m)
        # Remove that last move we tried from the path
        node.path = node.path[0:-1]        
        # Remove 1 from node's cost
        node.cost = node.cost - 1
    
    #Couldn't find a solution here or at any of my successors, so return None
    #This node is not on a solution path under the depth-limit
    return None
        
def run_best_first_search(fringe, options):
    """
    Runs an arbitrary best-first search.  To change which search is run, modify the f-value 
    computation in the search nodes

    fringe is a priority queue of search nodes, ordered by f-values
    """
    #Create our data structure to track visited/expanded states
    visited = dict()
    
    #Variable to tell when we are done
    done = False
                
    #Main search loop.  Keep going as long as we are not done and the FRINGE isn't empty
    while not done and not fringe.empty():
        
        #Get the next SearchNode from the FRINGE
        cur_node = fringe.get()
        
        #Add it to our set of visited/expanded states (join creates a string from the state)
        visited[cur_node.puzzle.id()] = True
        
        #Don't continue if the cost is too much
        if cur_node.cost > 200:
            #None of the puzzles are this long, so we shouldn't continue further on this path
            continue
            
        #Check to see if this node's puzzle state is a goal state
        if cur_node.puzzle.is_solved():
            #It is! We are done, print out details
            done = True
            print('Best-First SOLVED THE PUZZLE: SOLUTION = ', cur_node.path)
            print('Expanded ', len(visited), 'states')
            return len(visited), len(cur_node.path)
            
        else:
            #Generate this SearchNode's successors and add them to the FRINGE
            
            #Get the possible moves (actions) for this state
            moves = cur_node.puzzle.get_moves()
            
            #For each move, do the move, create SearchNode from successor, then add to FRINGE
            for m in moves:
                #Create new puzzle that new node will point to
                np = Puzzle(cur_node.puzzle.state)
                
                #Execute the move/action
                np.do_move(m)
                
                #Add to the FRINGE, as long as we haven't visited that puzzle
                if np.id() not in visited:
                    #Create the new SearchNode
                    new_node = SearchNode(cur_node.cost + 1, np, cur_node.path + m, options)
                    
                    #Add it to the FRINGE, along with its f-value (stored inside the node)
                    fringe.put(new_node)

    #We didn't find a solution
    if not done:
        print('NO SOLUTION FOUND!')
        return None,None
        
def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="8-Puzzle Solver.")
    parser.add_argument('file', metavar='FILENAME', type=str, help="File of puzzles to solve")
    parser.add_argument("-s", '--search', help="Search type: Options: ids (iterative deepening search) or bfs (best first search)", default='ids')
    parser.add_argument("-f", '--function', help="Heuristic function used: Options: top (tiles out of place), torc (tiles out of row/column), or md (manhattan distance)",default='top')
    parser.add_argument("-t", '--type', help="Evaluation function type: Options: g (greedy), u (uniform cost), or a (a-star)", default='u')
    options = parser.parse_args(args)
    return options

if __name__ == '__main__':

    #Get command line options
    options = getOptions()
    print(options)
    print('Searching for solutions to puzzles from file: ', sys.argv[1])
    
    #Open puzzle file
    pf = open(options.file,'r')
    
    #You can modify the maximum number of puzzles to solve if you want to test on more puzzles
    max_to_solve = 40
    
    #Variables to keep track of solving statistics
    num_solved = 0
    exp_num = 0
    tot_time = 0.0
    path_length = 0
    
    for ps in pf.readlines():
        print('Searching to find solution to following puzzle:', ps)
        
        #Create puzzle from file line
        a = [int(i) for i in ps.rstrip()]
        p = Puzzle(a)
        
        #Print the puzzle to the screen
        p.print_puzzle()

        #Create the initial search node corresponding to the given puzzle state
        start_node = SearchNode(0,p,'',options)
                            
        #Create the priority Queue to store the SearchNodes in
        pq = PriorityQueue()
        
        #Insert the initial state into the Queue
        pq.put(start_node)
        
        #Get initial timing info
        start = time.time()
        
        #Run the given search search (each returns number of nodes expanded and the length of the path found)        
        if options.search == 'bfs':
            #Run the best-first searches
            exp, pl = run_best_first_search(pq, options)
        elif options.search == 'ids': 
            #Use this line to run the iterative deepening-search
            exp, pl = run_iterative_search(start_node)
        else:
            print("Search option not valid. Can be bfs or ids")
            sys.exit()

        if exp is None:
            print('PUZZLE NOT SOLVED')
            break
            
        #Keep track of statistics so we can compare search methods
        exp_num += exp
        path_length += pl
        print('Solution path length is : ', pl)
            
        #Calculate Timing info
        end = time.time()
        tot_time += end - start
        num_solved += 1
        
        #Stop after we have solved the specified number of puzzles
        if num_solved >= max_to_solve:
            break
            
    print('Done with solving puzzles.\n\n')
            
    #Print out statistics about this batch
    if num_solved > 0:
        print('Solved', num_solved, 'puzzles from file: ', sys.argv[1])
        print('Average nodes expanded: ', float(exp_num) / float(num_solved))
        print('Average search time: ', tot_time / num_solved)
        print('Average solution length: ', path_length / num_solved)
    else:
        print('No puzzles were solved')

def test_algorithms_and_plot(filename='easy.txt'):
    """
    Test all combinations of algorithms (u, g, a) with heuristics (top, torc, md) 
    and create a grouped bar chart showing nodes expanded.
    """
    algorithms = ['u', 'g', 'a'] 
    heuristics = ['top', 'torc', 'md'] 
    algorithm_names = ['Uniform-Cost', 'Greedy Best-First', 'A*']
    heuristic_names = ['Tiles Out of Place', 'Tiles Out of Row/Col', 'Manhattan Distance']
    
    results = {}
    
    print(f"Testing all algorithm-heuristic combinations on {filename}")
    print("=" * 60)
    
    # Read first puzzle from file
    with open(filename, 'r') as pf:
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
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
    
    # Save plot to file
    plot_filename = f'algorithm_comparison_{filename.replace(".txt", "")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Show plot
    try:
        plt.show()
    except:
        print("Note: Plot display not available in this environment, but saved to file.")
    
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