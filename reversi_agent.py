"""
This module contains agents that play reversi.

Version 3.0
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2



_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.
        
        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

        self._expanded = 0
    
    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.
        
        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(
                target=self.search, 
                args=(
                    self._color, board, valid_actions, 
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.
        
        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.

        """

        raise NotImplementedError('You will have to implement this.')

class RandomAgent(ReversiAgent):
    """An agent that move randomly."""
    
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

# Each team will implement:
# An evaluation func
# Depth-limited Minimax with Alpha-Beta pruning

# [7 points] The correctness of your implementation
# [2 points] Evaluation Func
# [3 points] Alpha-Beta search
# [1 points] Depth-limited condition
# [1 points] Action ordering (to make pruning more effective)

class AlphaBetaAgent(ReversiAgent):

    def minimax(self, depth, board, valid_action, is_max, alpha, beta):
        env = gym.make('Reversi-v0')
        enemy = 1 if is_max == -1 else -1
        score = np.sum(board == is_max)
        limit = 4

        if depth == limit:
            return score

        self._expanded += 1

        new_board = transition(board, is_max, valid_action)
        next_valids_action = env.get_valid((new_board, enemy))
        next_valids_action = np.array(list(zip(*next_valids_action.nonzero())))

        best_score = alpha if is_max == 1 else beta
        # best_score = score

        for i in next_valids_action:
            action = i
            child_score = self.minimax(depth+1, new_board, action, enemy, alpha, beta)

            if is_max == 1 and best_score < child_score: # mean is max turn
                best_score = child_score
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            elif is_max == -1 and best_score > child_score: # mean is min turn
                best_score = child_score
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            # print(
            #     "\npass: " + str(action) +
            #     " action: " + str(child_action) +
            #     " score: " + str(child_score)
            # )

        return best_score

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):

        try:
            # while True:
            #     pass
            time.sleep(3)

            final_score = 0            # best score of valids_action
            final_action = None        # valids_action that has best score

            for i in valid_actions:    # each valid action
                action = i
                score = self.minimax(0, board, i, color, float('-inf'), float('inf')) # minimax func will return action(state) and score
                # print("action: " + str(i) + " score: " + str(score))
                if final_score < score:
                    final_score = score
                    final_action = action

                # print(
                #     "\npass: " + str(i) +
                #     " action: " + str(action) +
                #     " score: " + str(score)
                # )

            print(self._expanded)

            output_move_row.value = final_action[0]
            output_move_column.value = final_action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

class JadeAgent(ReversiAgent):
       def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        # transition(board, self.player, valid_actions[0])
        try:
            # while True:
            #     pass
            #time.sleep(3)
            final_action = None
            final_score = 0
            nodevalue = np.sum(board == self.player)

            for i in valid_actions:  # each valid action
                action = valid_actions[i]
                #(self, board, player, depth):
                score  = self.getmin(self, board, self.player, 2)       #depth gets decreased by 1 since this is the first level.
                if final_score < score + nodevalue:
                    final_score = score
                    final_action = action

                # print(
                #     "\npass: " + str(i) +
                #     " action: " + str(action) +
                #     " score: " + str(score)
                # )

            output_move_row.value = final_action[0]
            output_move_column.value = final_action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

################################################################################################

    def getmax(self, board, player, depth):
        if self.fullboard(board, player):                                #If the minmax reaches the end of the game.
            return 0
        if depth >= LIMITDEPTH:                                             #If the maximum depth reached.
            return 0
        v = float('-inf')
        valid_actions = self.nextvalid_actions(self, board, player)
        for i in valid_actions:                                              #Loop for all the new valid_moves made from the changed board
            action = valid_actions[i]
            new_board = transition(board, player, action)            #Initialize a new board.
            newplayer = -1 * player                                          #set this to indicate next player's turn
            returnvalue = self.getmin(new_board, newplayer, depth + 1)       #Call getmin to get the least value.

            if(i == 0):                          #Assign value to the first node return to use as a frame.
                v = returnvalue

            if v > returnvalue:                  #Use to find the lowest value out of all the getmins called above.
                v = returnvalue
            nodevalue = np.sum(new_board == self.player)      #The score of this node.
            nodevalue = nodevalue + v                         #Combine the score from this node to the least score from getmin.
        return nodevalue


    def getmin(self, board, player, alpha, depth):
        if self.fullboard(board, player):                            #If the minmax reaches the end of the game.
            return 0
        if depth >= LIMITDEPTH:                                         #If the maximum depth reached.
            return 0
        v = float('-inf')
        valid_actions = self.nextvalid_actions(self, board, player)
        for i in valid_actions:                                         #Loop for all the new valid_moves made from the changed board
            newboard = transition(board, player, i)
            action = valid_actions[i]
            new_board = transition(board, player, action)         # Initialize a new board.
            newplayer = -1 * player                                      # set this to indicate next player's turn
            returnvalue = self.getmax(new_board, newplayer,depth + 1)    # Call getmax to get the highest value.

            if (i == 0):                                # Assign value to the first node return to use as a frame.
                v = returnvalue

            if v < returnvalue:                         # Use to find the lowest value out of all the getmaxs called above.
                v = returnvalue
            nodevalue = np.sum(new_board == self.player)      # The score of this node.
            nodevalue = nodevalue + v                         # Combine the score from this node to the least score from getmin.
        return nodevalue

###############################################################################

    def fullboard(self, board, player):                         #Call to check if the board is already full.
        winner = _ENV.get_winner((board, player))
        if winner is not None:
            return True
        return False

    def nextvalid_actions(self, board, player):                 #Call to find validmove for that node
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

