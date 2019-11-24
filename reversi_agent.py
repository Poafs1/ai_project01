"""
This module contains agents that play reversi.
Version 3.0
"""

#THIS CODE IS THE SELECTED CODE TO BE SENT INTO THE COMPETITION FOR THE "ADVANCED JADE" GROUP. 

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2
import sys



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
# [2 points] Evaluation Func -> ok
# [3 points] Alpha-Beta search -> ok
# [1 points] Depth-limited condition -> ok
# [1 points] Action ordering (to make pruning more effective) -> ok

class JADE_IV_Agent(ReversiAgent):

    # [1 points] Action ordering (to make pruning more effective)
    def order(self, state, board, opp_player, player, alpha, beta):    #This function will rearrange the next possible move. 
        list_of_dict = []                                              #will store the score in each move's board from highest score to lowest score.

        best_score = alpha if player == 1 else beta

        for i in state:
            dict = {}
            new_board = transition(board, opp_player, i)
            score = self.evaluate_score(new_board, opp_player, player) #get score from respective move.

            if player == 1 and best_score < score:                                  
                best_score = score
                alpha = max(alpha, score)
                if beta <= alpha:                                      #pruning
                    break
            elif player == -1 and best_score > score:
                best_score = score
                beta = min(beta, score)
                if beta <= alpha:                                      #pruning                   
                    break

            dict["action"] = i
            dict["score"] = score
            list_of_dict.append(dict)                                  #Storing score, move into a list.

        if opp_player == 1:
            sort = sorted(list_of_dict, key=lambda i: i["score"])      #Sort score with their respective move from highest to lowest.
        elif opp_player == -1: # parent is max
            sort = sorted(list_of_dict, key=lambda i: i["score"], reverse=True)

        state = [d["action"].tolist() for d in sort]

        return state

    def next_state(self, board, state, player, opp_player):
        new_board = transition(board, player, state)
        valids = _ENV.get_valid((new_board, opp_player))
        valids = np.array(list(zip(*valids.nonzero())))
        return  new_board, valids

    # [2 points] Evaluation Func
    def evaluate_score(self, board, player, opp_player):                       #return score which in itself is a combination of the board's score from a player's move taken into consideration with
                                                                               #an arbitrary scoreboard.
        
        #The score below was in reference to a paper in the link below:               
        # http://www.csse.uwa.edu.au/cig08/Proceedings/papers/8010.pdf

        open_stage = [                                                                              #open stage encourages the program to take the points near 4 sides, but not near the 4 corners.
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0],
            [0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0],
            [0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0],
            [0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0],
            [0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0],
            [0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],

        ]
        mid_stage = [                                                                               #mid_stage encourages the program to prioritizes 4 corners, and not anywhere that the enermy can go on to take the corners. 
            [6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711],
            [-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813],
            [0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907],
            [-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512],
            [-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512],
            [0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907],
            [-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813],
            [6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711],
        ]
        end_stage = [                                                                               #end_stage encourages similar actions to mid_stage, the difference being that it's used in the state of the game where the game has already slowed down.
            [5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062],
            [-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812],
            [-2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948],
            [-0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007],
            [-0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007],
            [-2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948],
            [-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812],
            [5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062],
        ]
        select_stage = open_stage                                           

        if (board[2][2] == player or board[2][5] == player                  #If the 4 points on the mid-layer of the board is taken, then use mid_stage as an arbitrary extra score.
            or board[5][2] == player or board[5][5] == player):
            select_stage = mid_stage

        if (board[2][2] != 0 and board[2][5] != 0                           #If the 4 points on the mid-layer of the board is not taken, then use end_stage as an arbitrary extra score.
            and board[5][2] != 0 and board[5][5] != 0):
            select_stage == end_stage

        if (board[0][0] == player or board[0][7] == player                  #Also if the 4 points on 4 corners of the board is taken, then use end_stage as an arbitrary extra score.
            or board[7][0] == player or board[7][7] == player):
            select_stage == end_stage

        sum = 0.0
        for i in range(8):
            for j in range(8):
                sum += board[i][j] * select_stage[i][j]                     #Respective move's board score plus arbitrary score from the board will be taken in to the program's considereton for the next move.
        return  sum

        # score = np.sum(board == player)
        # return score.item()

    # [3 points] Alpha-Beta search
    def alpha_beta(self, depth, board, state, player, alpha, beta):         #Minmax simulation function
        opp_player = 1 if player == -1 else -1
        limit = 4                                                           #Set depth limit to 4

        # [1 points] Depth-limited condition
        if depth == limit: return self.evaluate_score(board, player, opp_player), None

        # self._expanded += 1

        best_score = alpha if player == 1 else beta
        best_action = None

        for i in state:
            new_board, new_state = self.next_state(board, i, player, opp_player)                                           #Get a new picture of the new board from the move
            if new_board is None: return self.evaluate_score(board, player, opp_player), None                              #Get the score from that board.
            # sort state here
            order_state = self.order(new_state, new_board, opp_player, player, alpha, beta)                                #Order the moves from highest score to lowest score to make the program run faster.

            child_score, child_action = self.alpha_beta(depth+1, new_board, order_state, opp_player, alpha, beta)          #Get score from recursion

            if player == 1 and best_score < child_score:
                best_score = child_score
                best_action = i
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            elif player == -1 and best_score > child_score:
                best_score = child_score
                best_action = i
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

        return best_score, best_action

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):

        try:
            # while True:
            #     pass
            # time.sleep(3)
            score, action = self.alpha_beta(0, board, valid_actions, color, float('-inf'), float('inf'))

            # print(self._expanded)

            if action is not None:
                output_move_row.value = action[0]
                output_move_column.value = action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
