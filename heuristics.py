"""
Heuristic functions for the 8-puzzle problem

Code for AI Class Programming Assignment 1
Written by Chris Archibald, modified by Trent Welling
archibald@cs.byu.edu, tdw57@byu.edu
"""

import sys


def heuristic(node, options):
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
        print('Invalid heuristic selected. Options are top, torc, and md')
        sys.exit()


def tiles_out_of_place(puzzle):
    """
    This heuristic counts the number of tiles out of place.    
    """
    # Keep track of the number of tiles out of place
    num_out_of_place = 0
    
    # Cycle through all of the places in the puzzle and see if the right tile is there
    # (We ignore place 0 since that is where the blank tile goes and we shouldn't count it)
    for i in range(1, len(puzzle.state)):
        # The tile in place i ( puzzle.state[i] ) should be tile i.  
        # If it isn't increment out of place counter
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