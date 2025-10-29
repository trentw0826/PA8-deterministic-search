"""
Search algorithms and SearchNode class for the 8-puzzle problem

Code for AI Class Programming Assignment 1
Written by Chris Archibald, modified by Trent Welling
archibald@cs.byu.edu, tdw57@byu.edu
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
    This runs an iterative deepening search
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
    visited['N'] = visited['N'] + 1  # Increment our node expansion counter

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