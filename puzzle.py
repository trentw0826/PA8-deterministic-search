"""
8-Puzzle class and related functionality

Code for AI Class Programming Assignment 1
Written by Chris Archibald, modified by Trent Welling
archibald@cs.byu.edu, tdw57@byu.edu

GOAL: (0 is the blank tile)
0 1 2
3 4 5
6 7 8
"""

class Puzzle:
    """
    8-puzzle class
    """

    SOLUTION_STATE = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(self, arrangement):
        """
        The state (and arrangement passed in) is a list of length 9, that stores which tile is in each place
        In a solved puzzle, each place number holds the tile of the same number
        i.e. solution is state = [0,1,2,3,4,5,6,7,8]
        """
        # TODO input validation
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