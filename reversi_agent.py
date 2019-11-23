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
# [1 points] Action ordering (to make pruning more effective)

class KluaAgent(ReversiAgent):
    def ordering(self, valids_action, board, is_max):

        myDict = dict()
        for i in range(len(valids_action)):
            new_board = transition(board, is_max, valids_action[i])
            score = self.evaluateScore(new_board, is_max)
            myDict[i] = score
            # myDict[i].append(valids_action[i])

        print(myDict)
        return valids_action

    def evaluateScore(self, board, turn):
        return np.sum(board == turn)

    def next_state(self, new_board, enemy):
        valids = _ENV.get_valid((new_board, enemy))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def minimax(self, depth, board, valid_action, is_max, alpha, beta):
        enemy = 1 if is_max == -1 else -1
        score = self.evaluateScore(board, is_max)
        limit = 3

        if depth == limit:
            return score

        self._expanded += 1

        new_board = transition(board, is_max, valid_action)
        if new_board is None: return score

        next_valids_action = self.next_state(new_board, enemy)

        # order next_valids_action list
        order_valids_action = self.ordering(next_valids_action, new_board, enemy)

        best_score = alpha if is_max == 1 else beta

        for i in order_valids_action:
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

        return best_score

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):

        try:
            # while True:
            #     pass
            # time.sleep(3)
            final_score = 0            # best score of valids_action
            final_action = None        # valids_action that has best score

            for i in valid_actions:    # each valid action
                action = i
                score = self.minimax(0, board, i, color, float('-inf'), float('inf')) # minimax func will return action(state) and score
                # print("action: " + str(i) + " score: " + str(score))
                if final_score < score:
                    final_score = score
                    final_action = action

            print(self._expanded)

            output_move_row.value = final_action[0]
            output_move_column.value = final_action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class EarthAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.

        # return move
        try:

            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]

            action, value = self.maxValue(board, self.player, float('-inf'), float('inf'), 0)
            output_move_row.value = action[0]
            output_move_column.value = action[1]

        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def maxValue(self, board, player, alpha, beta, depth):
        if self.terminalTest(board, player):
            return None, self.utility(board, player)

        if depth >= 4:
            return None, self.eva(board, player)

        v = float('-inf')
        maxAction = None

        for action in self.actions(board, player):
            newBoard = transition(board, player, action)
            newPlayer = -1 * player
            NO, newV = self.minValue(newBoard, newPlayer, alpha, beta, depth + 1)
            if v < newV:
                v = newV
                maxAction = action

            if v >= beta:
                return maxAction, v

            alpha = max(alpha, v)

        return maxAction, v

    def minValue(self, board, player, alpha, beta, depth):
        if self.terminalTest(board, player):
            return None, self.utility(board, player)

        if depth >= 4:
            return None, self.eva(board, player)

        v = float('inf')
        minAction = None
        for action in self.actions(board, player):
            newBoard = transition(board, player, action)
            newPlayer = -1 * player
            NO, newV = self.maxValue(newBoard, newPlayer, alpha, beta, depth + 1)
            if v > newV:
                v = newV
                maxAction = action

            if v <= alpha:
                return minAction, v

            beta = min(beta, v)

        return minAction, v

    def terminalTest(self, board, player):
        winner = _ENV.get_winner((board, player))
        if winner is not None:
            return True
        return False

    def utility(self, board, player):
        return np.sum(board == self.player)

    def actions(self, board, player):
        valids = _ENV.get_valid((board, player))
        valids = np.array(list(zip(*valids.nonzero())))
        return valids

    def eva(self, board, player):
        for i in board:
            for j in i:
                if j == bg2.BLACK:
                    return np.sum(board == bg2.BLACK)
                elif j == bg2.WHITE:
                    return np.sum(board == bg2.WHITE)
        return self.utility(board, player)

class ViewAgent(ReversiAgent):
    def __index__(self):
        super(self.minimax, self)
        # self.transpositionTable = set()

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        if self._color == 1:
            evaluation, bestAction = self.minimax(board, valid_actions, 4, 0, - sys.maxsize - 100000, sys.maxsize, True)
        else:
            evaluation, bestAction = self.minimax(board, valid_actions, 2, 0, - sys.maxsize - 100000, sys.maxsize, True)
        output_move_row.value = bestAction[0]
        output_move_column.value = bestAction[1]

    def minimax(self, board: np.array, validActions: np.array, depth: int, levelCount: int, alpha: int, beta: int,
                gain: bool):
        if depth == 0:
            return self.evaluateStatistically(board)

        bestAction: np.array = None
        if gain:
            Alpha: int = alpha
            maxevaluation: int = -99999
            player: int = self._color

            for action in validActions:
                newState, newboard = self.createState(board, action, player)
                evaluation = self.minimax(newState, newboard, depth - 1, levelCount + 1, Alpha, beta, not gain)

                if maxevaluation < evaluation:
                    maxevaluation = evaluation

                    if levelCount == 0:
                        bestAction = action

                Alpha = max(Alpha, evaluation)
                if beta <= Alpha:
                    break
            if levelCount != 0:
                return maxevaluation
            else:
                return maxevaluation, bestAction
        else:
            mBeta: int = beta
            minEval: int = sys.maxsize
            player: int = self.getOpponent(self._color)

            for action in validActions:
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions, depth - 1, levelCount + 1, alpha, mBeta, not gain)

                if minEval > evaluation:
                    minEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mBeta = min(mBeta, evaluation)
                if mBeta <= alpha:
                    break
            if levelCount != 0:
                return minEval
            else:
                return minEval, bestAction

    def evaluateStatistically(self, board: np.array):
        countA: int = 0
        countB: int = 0
        evalBoard = np.array(list(zip(*board.nonzero())))

        for row in evalBoard:
            if board[row[0]][row[1]] == self._color:
                countA += 1
            else:
                countB += 1
        return countA - countB

    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1

    def createState(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        newState: np.array = transition(board, player, action)

        Move: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        Move: np.array = np.array(list(zip(*Move.nonzero())))

        return newState, Move

class JadeAgent(ReversiAgent):
    def minimax(self, depth, board, valid_action, is_max, alpha, beta):
        env = gym.make('Reversi-v0')
        enemy = 1 if is_max == -1 else -1
        score = np.sum(board == is_max)
        extra_score = 0  # Will increase depending on the position that the move will produce.
        corner = {[0][0], [0][7], [7][0], [7][7]}  # Create corners with high extra_scores
        sides = {[0][1], [0][2], [0][3], [0][4], [0][5], [0][6], [7][1], [7][2], [7][3], [7][4], [7][5], [7][6], [1][0],
                 [2][0], [3][0], [4][0], [5][0], [6][0], [1][7], [2][7], [3][7], [4][7], [5][7], [6][7]}
        baitcorner = {[2][2], [2][3], [2][4], [2][5], [3][2], [3][5], [4][2], [4][5], [5][2], [5][3], [5][4],
                      [5][5]}  # create a potential bait for the enermy to take.
        limit = 4

        if depth == limit:
            return score, extra_score

        self._expanded += 1

        new_board = transition(board, is_max, valid_action)
        next_valids_action = env.get_valid((new_board, enemy))
        next_valids_action = np.array(list(zip(*next_valids_action.nonzero())))

        best_score = alpha if is_max == 1 else beta
        # best_score = score

        for i in next_valids_action:
            action = i
            for y in corner:  # Calculate extra_Score gained from going into corners.
                if action == corner[y]:
                    extra_score = extra_score + 10
                    break
            for y in sides:  # calculate extra Score gained from going into sides.
                if action == sides[y]:
                    extra_score = extra_score + 5
                    break
            for y in baitcorner:  # calculate extra Score gained from baiting the opponent.
                if action == baitcorner[y]:
                    extra_score = extra_score + 3
                    break
            child_score, child_extra_score = self.minimax(depth + 1, new_board, action, enemy, alpha, beta)
            extra_score = extra_score + child_extra_score
            if is_max == 1 and best_score < child_score:  # mean is max turn
                best_score = child_score
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            elif is_max == -1 and best_score > child_score:  # mean is min turn
                best_score = child_score
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

            # print(
            #     "\npass: " + str(action) +
            #     " action: " + str(child_action) +
            #     " score: " + str(child_score)
            # )

        return best_score, extra_score

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):

        try:
            # while True:
            #     pass
            final_score = 0  # best score of valids_action
            final_action = None  # valids_action that has best score

            for i in valid_actions:  # each valid action
                action = i
                score, extra_score = self.minimax(0, board, i, color, float('-inf'),
                                                  float('inf'))  # minimax func will return action(state) and score
                # print("action: " + str(i) + " score: " + str(score))
                if final_score < score + extra_score:
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