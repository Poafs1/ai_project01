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
LIMITDEPTH = 3

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

        #ADD HERE////////////////////////////////////////////////////////////////////////////////////
        randidx = random.randint(0, len(valid_actions) - 1)
        random_action = valid_actions[randidx]
        output_move_row.value = random_action[0]
        output_move_column.value = random_action[1]

        position = self.minmax(self, board, self.player, 0)         #Got the best option from minmax
        action = valid_actions[position]
        output_move_row.value = action[0]
        output_move_column.value = action[1]

        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""
    
    def search(
            self, color, board, valid_actions, 
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

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



class AlphaBetaAgent(ReversiAgent):

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        # transition(board, self.player, valid_actions[0])
        try:
            # while True:
            #     pass

            # Calcualte current path score
            # black_score = np.sum(board == 1)
            # white_score = np.sum(board == -1)
            # so score of this turn is eq -> np.sum(board == color)

            # valid_action -> [[2 4], [3 5], [4 2], [5 3]]
            # new_board = transition(current_board, color, action)

            #time.sleep(3)
            env = gym.make('Reversi-v0')

            # ไม่แน่ใจว่า turn ต้องเป็นสีนั้นหรือสีตรงข้าม
            # turn = -1 if color == 1 else 1
            turn = color



            # หา valids action ถัดไปที่สามารถเดินได้
            #valids = env.get_valid((current_board, turn))
            #valids = np.array(list(zip(*valids.nonzero())))
            #print(valids)

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)




        def minmax(self, board, player, depth):                   #The start of the minmax caculations.
            choice = 0
            maxvalue = 0
            for i in valid_actions:
                v = self.getmax(self, board, player, depth)
                if(i == 0):                                       #Set the first move as maximum(Will be changed later if other moves produce better score
                    maxvalue = v
                if(v > maxvalue):                                 #If other move produces better results.
                    maxvalue = v
                    choice = i
            return i

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
                new_board = transition(board, color, action)         # Initialize a new board.
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
