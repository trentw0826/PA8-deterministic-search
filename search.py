"""
Search algorithms and SearchNode class for the 8-puzzle problem
"""

import sys
from queue import PriorityQueue
from puzzle import Puzzle
from heuristics import heuristic


class SearchNode:
    """
    Our search node class
    """
    
    def __init__(self, cost, puzzle, path, options):
        """
        Initialize all the relevant parts of the search node
        """
        self.cost = cost
        self.puzzle = puzzle
        self.path = path
        self.options = options
        self.h = heuristic(self, self.options)
        self.compute_f_value()
       
    def compute_f_value(self):
        """
        Compute the f-value for this node
        """
        self.h = heuristic(self, self.options)
        self.f_value = 0

        if self.options.type == 'g':
            # greedy search algorithm: f(n) = h(n)
            self.f_value = self.h

        elif self.options.type == 'u':
            # uniform cost search algorithm: f(n) = g(n)
            self.f_value = self.cost

        elif self.options.type == 'a':
            # A* search algorithm: f(n) = g(n) + h(n)
            self.f_value = self.cost + self.h

        else:
            print('Invalid search type (-t) selected: Valid options are g, u, and a')
            sys.exit()

    # Comparison operator 
    def __lt__(self, other):
        """
        Comparison operator so that nodes will be sorted in priority queue based on f-value
        """
        return self.f_value < other.f_value


def run_iterative_search(start_node):
    """
    This runs IDA* (Iterative Deepening A*) search
    It iteratively increases the f-value threshold until a solution is found
    
    Note: For optimal results, this should be used with evaluation function type 'a' (A*)
    to ensure f(n) = g(n) + h(n), which is required for IDA* to work correctly.
    """
    # Initial f-value threshold is the heuristic value of the start node
    threshold = start_node.f_value
    
    # Maximum threshold to prevent infinite loops (arbitrary large value)
    max_threshold = 1000
    
    # Keep track of the total number of nodes we expand
    total_expanded = 0
    
    # Keep trying with increasing thresholds
    while threshold <= max_threshold:
        
        # Store visited nodes along the current search path to avoid cycles
        visited = dict()
        visited['N'] = 0

        # Mark the initial state as visited
        visited[start_node.puzzle.id()] = True
        
        # Run f-value limited search starting at initial node
        result = run_ida_star(start_node, threshold, visited) 
    
        # See how many nodes we expanded on this iteration and add it to our total
        total_expanded += visited['N']
        
        # Check the result
        if isinstance(result, int):
            # Solution found! result is the path length
            print('Expanded ', total_expanded, 'nodes')
            print('IDA* Found solution with threshold', threshold)
            return total_expanded, result
        elif result == float('inf'):
            # No solution exists
            print('No solution found - search space exhausted')
            return None, None
        else:
            # result is the minimum f-value that exceeded the threshold
            # Set this as our new threshold for the next iteration
            threshold = result
        
    # Exceeded maximum threshold, return failure
    print('Exceeded maximum threshold - no solution found')
    return None, None


def run_ida_star(node, threshold, visited):
    """
    Recursive IDA* search with f-value threshold:  
    
    Check node to see if it is goal, if it is, print solution and return path length
    If f-value exceeds threshold, return the f-value for next iteration
    Otherwise, recurse on all children and return minimum f-value that exceeded threshold
    """
    visited['N'] = visited['N'] + 1  # Increment our node expansion counter

    # Check if f-value exceeds the current threshold
    if node.f_value > threshold:
        return node.f_value
        
    # Check to see if this is a goal node
    if node.puzzle.is_solved():
        # It is! Print out solution and return solution length
        print('IDA* SOLVED THE PUZZLE! SOLUTION = ', node.path)
        return len(node.path)
    
    # Track the minimum f-value that exceeded the threshold
    min_threshold = float('inf')
    
    # Generate successors and recurse on them
    
    # Get the list of moves we can try from this node's state
    moves = node.puzzle.get_moves()
    
    # For each possible move
    for m in moves:
        # Execute the move/action
        node.puzzle.do_move(m)
        # Add this move to the node's path
        node.path = node.path + m
        # Add 1 to node's cost
        node.cost = node.cost + 1
        # Recompute f-value with new cost
        node.compute_f_value()

        # Check to see if we have already visited this node (cycle detection)
        if node.puzzle.id() not in visited:
            # We haven't. Now we will, so add it to visited
            visited[node.puzzle.id()] = True
            
            # Recurse on this new state
            result = run_ida_star(node, threshold, visited)    
            
            # Check if a solution was found
            if isinstance(result, int):
                # Solution found! Return the path length
                return result
            
            # Update minimum threshold for next iteration
            if result < min_threshold:
                min_threshold = result
                
            # Remove this state from the visited list (backtrack)
            del visited[node.puzzle.id()]

        # That move didn't lead to a solution, so lets try the next one
        # First, though, we need to undo the move (to return puzzle to state before we tried that move)
        node.puzzle.undo_move(m)
        # Remove that last move we tried from the path
        node.path = node.path[0:-1]        
        # Remove 1 from node's cost
        node.cost = node.cost - 1
        # Recompute f-value with reverted cost
        node.compute_f_value()
    
    # Return the minimum f-value that exceeded the threshold
    return min_threshold


def run_best_first_search(fringe, options):
    """
    Runs an arbitrary best-first search.  To change which search is run, modify the f-value 
    computation in the search nodes

    fringe is a priority queue of search nodes, ordered by f-values
    """
    # Create our data structure to track visited/expanded states
    visited = dict()
    
    # Variable to tell when we are done
    done = False
                
    # Main search loop.  Keep going as long as we are not done and the FRINGE isn't empty
    while not done and not fringe.empty():
        
        # Get the next SearchNode from the FRINGE
        cur_node = fringe.get()
        
        # Add it to our set of visited/expanded states (join creates a string from the state)
        visited[cur_node.puzzle.id()] = True
        
        # Don't continue if the cost is too much
        if cur_node.cost > 200:
            # None of the puzzles are this long, so we shouldn't continue further on this path
            continue
            
        # Check to see if this node's puzzle state is a goal state
        if cur_node.puzzle.is_solved():
            # It is! We are done, print out details
            done = True
            print('Best-First SOLVED THE PUZZLE: SOLUTION = ', cur_node.path)
            print('Expanded ', len(visited), 'states')
            return len(visited), len(cur_node.path)
            
        else:
            # Generate this SearchNode's successors and add them to the FRINGE
            
            # Get the possible moves (actions) for this state
            moves = cur_node.puzzle.get_moves()
            
            # For each move, do the move, create SearchNode from successor, then add to FRINGE
            for m in moves:
                # Create new puzzle that new node will point to
                np = Puzzle(cur_node.puzzle.state)
                
                # Execute the move/action
                np.do_move(m)
                
                # Add to the FRINGE, as long as we haven't visited that puzzle
                if np.id() not in visited:
                    # Create the new SearchNode
                    new_node = SearchNode(cur_node.cost + 1, np, cur_node.path + m, options)
                    
                    # Add it to the FRINGE, along with its f-value (stored inside the node)
                    fringe.put(new_node)

    # We didn't find a solution
    if not done:
        print('NO SOLUTION FOUND!')
        return None, None


def run_basic_iterative_search(start_node):
    """
    This runs the original iterative deepening search (for comparison with IDA*)
    It caps the depth of the search at 40 (no 8-puzzles have solutions this long)
    """
    # Our initial depth limit
    depth_limit = 1
    
    # Maximum depth limit
    max_depth_limit = 40
    
    # Keep track of the total number of nodes we expand
    total_expanded = 0
    
    # Keep trying until our depth limit hits 40
    while depth_limit < max_depth_limit:
        
        # Store visited nodes along the current search path
        visited = dict()
        visited['N'] = 0

        # Mark the initial state as visited
        visited[start_node.puzzle.id()] = True
        
        # Run depth-limited search starting at initial node (which points to initial state)
        path_length = run_dfs(start_node, depth_limit, visited) 
    
        # See how many nodes we expanded on this iteration and add it to our total
        total_expanded += visited['N']
        
        # Check to see if a solution was found
        if path_length is not None:
            # It was! Print out information and return the search stats
            print('Expanded ', total_expanded, 'nodes')
            print('Original IDS Found solution at depth', depth_limit)
            return total_expanded, path_length
            
        # No solution was found at this depth limit, so increment our depth-limit    
        depth_limit += 1
        
    # No solution was found at any depth-limit, so return None,None (Which signifies no solution found)
    return None, None


def run_dfs(node, depth_limit, visited):
    """
    Recursive Depth-Limited Search for basic iterative deepening:  
    
    Check node to see if it is goal, if it is, print solution and return path length
    If not and if depth-limit hasn't been reached, recurse on all children
    """
    visited['N'] = visited['N'] + 1  # Increment our node expansion counter

    # Check to see if this is a goal node
    if node.puzzle.is_solved():
        # It is! Print out solution and return solution length
        print('Original Iterative Deepening SOLVED THE PUZZLE! SOLUTION = ', node.path)
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
        # Execute the move/action
        node.puzzle.do_move(m)
        node.compute_f_value()

        # Add this move to the node's path
        node.path = node.path + m
        # Add 1 to node's cost
        node.cost = node.cost + 1
        # Check to see if we have already visited this node
        if node.puzzle.id() not in visited:
            # We haven't. Now we will, so add it to visited
            visited[node.puzzle.id()] = True
            
            # Recurse on this new state
            path_length = run_dfs(node, depth_limit, visited)    
            
            # Check to see if a solution was found down this path (return value of None means no)
            if path_length is not None:
                # It was! Return this solution path length to whoever called us
                return path_length
                
            # Remove this state from the visited list.  We only check for duplicates along current search path
            del visited[node.puzzle.id()]

        # That move didn't lead to a solution, so lets try the next one
        # First, though, we need to undo the move (to return puzzle to state before we tried that move)
        node.puzzle.undo_move(m)
        # Remove that last move we tried from the path
        node.path = node.path[0:-1]        
        # Remove 1 from node's cost
        node.cost = node.cost - 1
    
    # Couldn't find a solution here or at any of my successors, so return None
    # This node is not on a solution path under the depth-limit
    return None