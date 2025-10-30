"""
Heuristic functions for the 8-puzzle problem
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
    elif options.function == 'mdlc':
        return manhattan_distance_with_linear_conflicts(node.puzzle)
    else:
        print('Invalid heuristic selected. Options are top, torc, md, and mdlc')
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


def manhattan_distance_with_linear_conflicts(puzzle):
    """
    Manhattan Distance with Linear Conflicts heuristic.
    This enhances Manhattan distance by detecting when tiles are in the correct row/column
    but need to pass each other to reach their goal positions, adding 2 for each conflict.
    
    A linear conflict occurs when:
    - Two tiles are in their correct row/column
    - Their goal positions are also in that same row/column 
    - They are in reverse order relative to their goals
    
    For each such conflict, we add 2 to the heuristic (minimum moves to resolve).
    """
    # Start with basic Manhattan distance
    md_distance = manhattan_distance_to_goal(puzzle)
    
    # Add linear conflicts
    conflicts = 0
    
    # Check for row conflicts
    for row in range(3):
        conflicts += count_linear_conflicts_in_line(puzzle, row, is_row=True)
    
    # Check for column conflicts  
    for col in range(3):
        conflicts += count_linear_conflicts_in_line(puzzle, col, is_row=False)
    
    return md_distance + 2 * conflicts


def count_linear_conflicts_in_line(puzzle, line_num, is_row=True):
    """
    Count linear conflicts in a specific row or column.
    
    Args:
        puzzle: The puzzle state
        line_num: Row number (0-2) if is_row=True, column number (0-2) if is_row=False
        is_row: True to check a row, False to check a column
    
    Returns:
        Number of linear conflicts in this line
    """
    # Get tiles in this line that belong in this line
    tiles_in_correct_line = []
    
    for pos in range(3):
        if is_row:
            idx = line_num * 3 + pos  # Convert row,col to index
        else:
            idx = pos * 3 + line_num  # Convert row,col to index
            
        tile = puzzle.state[idx]
        
        # Skip blank tile
        if tile == 0:
            continue
            
        # Check if this tile belongs in this row/column
        tile_goal_row = get_tile_row(tile)
        tile_goal_col = get_tile_column(tile)
        
        if is_row and tile_goal_row == line_num:
            tiles_in_correct_line.append((tile, pos))
        elif not is_row and tile_goal_col == line_num:
            tiles_in_correct_line.append((tile, pos))
    
    # Count conflicts among tiles that belong in this line
    conflicts = 0
    
    # For each pair of tiles in the correct line
    for i in range(len(tiles_in_correct_line)):
        for j in range(i + 1, len(tiles_in_correct_line)):
            tile1, pos1 = tiles_in_correct_line[i]
            tile2, pos2 = tiles_in_correct_line[j]
            
            # Get their goal positions in this line
            if is_row:
                goal_pos1 = get_tile_column(tile1)
                goal_pos2 = get_tile_column(tile2)
            else:
                goal_pos1 = get_tile_row(tile1)
                goal_pos2 = get_tile_row(tile2)
            
            # Check if they are in conflicting order
            if (pos1 < pos2 and goal_pos1 > goal_pos2) or (pos1 > pos2 and goal_pos1 < goal_pos2):
                conflicts += 1
    
    return conflicts


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