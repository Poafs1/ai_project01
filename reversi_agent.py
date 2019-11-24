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
# [1 points] Action ordering (to make pruning more effective) -> ok

class KluaAgent(ReversiAgent):
    def order(self, state, board, opp_player, player, alpha, beta):
        list_of_dict = []

        best_score = alpha if player == 1 else beta

        for i in state:
            dict = {}
            new_board = transition(board, opp_player, i)
            score = self.evaluate_score(new_board, opp_player, player)

            if player == 1 and best_score < score:
                best_score = score
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            elif player == -1 and best_score > score:
                best_score = score
                beta = min(beta, score)
                if beta <= alpha:
                    break

            dict["action"] = i
            dict["score"] = score
            list_of_dict.append(dict)

        if opp_player == 1:
            sort = sorted(list_of_dict, key=lambda i: i["score"])
        elif opp_player == -1: # parent is max
            sort = sorted(list_of_dict, key=lambda i: i["score"], reverse=True)

        state = [d["action"].tolist() for d in sort]

        return state

    def next_state(self, board, state, player, opp_player):
        new_board = transition(board, player, state)
        valids = _ENV.get_valid((new_board, opp_player))
        valids = np.array(list(zip(*valids.nonzero())))
        return  new_board, valids

    def evaluate_score(self, board, player, opp_player):
        # should know -> count of game turn / depth/ limit ??????
        select_stage = []
        first_stage = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0],
            [0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0],
            [0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0],
            [0, 0.02004, -0.10927, -0.10155, -0.10155, -0.10927, 0.02004, 0],
            [0, 0.05583, 0.10126, -0.10927, -0.10927, 0.10126, 0.05583, 0],
            [0, -0.02231, 0.05583, 0.02004, 0.02004, 0.05583, -0.02231, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],

        ]
        mid_stage = [
            [6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711],
            [-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813],
            [0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907],
            [-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512],
            [-2.00512, -0.18176, 0.62415, 0.10539, 0.10539, 0.62415, -0.18176, -2.00512],
            [0.33907, -1.87550, 1.06939, 0.62415, 0.62415, 1.06939, -1.87550, 0.33907],
            [-3.32813, -1.52928, -1.87550, -0.18176, -0.18176, -1.87550, -1.52928, -3.32813],
            [6.32711, -3.32813, 0.33907, -2.00512, -2.00512, 0.33907, -3.32813, 6.32711],
        ]
        end_stage = [
            [5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062],
            [-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812],
            [-2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948],
            [-0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007],
            [-0.59007, -2.01723, -1.07055, 0.73486, 0.73486, -1.07055, -2.01723, -0.59007],
            [-2.58948, -2.16084, 0.49062, -1.07055, -1.07055, 0.49062, -2.16084, -2.58948],
            [-0.17812, 0.96804, -2.16084, -2.01723, -2.01723, -2.16084, 0.96804, -0.17812],
            [5.50062, -0.17812, -2.58948, -0.59007, -0.59007, -2.58948, -0.17812, 5.50062],
        ]



        # sum = 0.0
        # for i in range(8):
        #     for j in range(8):
        #         sum += board[i][j] * score_board[i][j]
        # return  sum
        
        score = np.sum(board == player)
        return score.item()

    def alpha_beta(self, depth, board, state, player, alpha, beta):
        opp_player = 1 if player == -1 else -1
        limit = 4

        if depth == limit: return self.evaluate_score(board, player, opp_player), None

        # self._expanded += 1

        best_score = alpha if player == 1 else beta
        best_action = None

        for i in state:
            new_board, new_state = self.next_state(board, i, player, opp_player)
            if new_board is None: return self.evaluate_score(board, player, opp_player), None
            # sort state here
            order_state = self.order(new_state, new_board, opp_player, player, alpha, beta)

            child_score, child_action = self.alpha_beta(depth+1, new_board, order_state, opp_player, alpha, beta)

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

class QewAgent(ReversiAgent):
    def __index__(self):
        super(QewAgent, self)

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        try:
            # 1900, -1900 is the number of probability state
            evaluation, best_state = self.maxinf(board, valid_actions, 4, 0, -1830, 1830, True)
            # time.sleep(2)
            # evaluation, best_state = self._max(board,valid_actions,4,0,-sys.maxsize - 1,sys.maxsize,True)
            if best_state is not None:
                output_move_row.value = best_state[0]
                output_move_column.value = best_state[1]
            # time.sleep(3)

        except Exception as e:
            # time.sleep(7)
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')

            traceback.print_tb(e.__traceback__)

    def maxinf(self, board: np.array, validactions: np.array, depth: int, level: int, alpha: int, beta: int,
               gain: bool):
        if depth == 0:
            return self.evaluation(board)

        bestMove: np.array = None
        maxAlpha: int = alpha
        maxEvaluation = -1830
        player = self._color
        for move in validactions:
            newboard, newaction = self.createState(board, move, player)
            newmove = self.mininf(newboard, newaction, depth - 1, level + 1, maxAlpha, beta, not gain)
            if maxEvaluation < newmove:
                maxEvaluation = newmove
                if level == 0:
                    bestMove = move

            maxAlpha = max(maxAlpha, maxEvaluation)
            if maxAlpha >= beta:
                break
        if level != 0:
            return maxEvaluation
        else:
            return maxEvaluation, bestMove

    def mininf(self, board: np.array, validactions: np.array, depth: int, level: int, alpha: int, beta: int,
               gain: bool):
        if depth == 0:
            return self.evaluation(board)

        bestMove: np.array = None
        minBeta: int = beta
        minEvaluation = 1830
        player: int = self.getOpponent(self._color)

        for move in validactions:
            newboard, newaction = self.createState(board, move, player)
            newmove = self.maxinf(newboard, newaction, depth - 1, level + 1, alpha, minBeta, not gain)
            if minEvaluation > newmove:
                minEvaluation = newmove
                if level == 0:
                    bestMove = move

            minBeta = min(minBeta, minEvaluation)
            if alpha >= minBeta:
                break
        if level != 0:
            return minEvaluation
        else:
            return minEvaluation, bestMove

    def evaluation(self, board: np.array):
        countA: int = 0
        countB: int = 0
        evaluationBoard = np.array(list(zip(*board.nonzero())))
        for i in evaluationBoard:
            if board[i[0]][i[1]] == self._color:
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
        validMoves: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves: np.array = np.array(list(zip(*validMoves.nonzero())))
        return newState, validMoves

class PloyRandomAgent(ReversiAgent):
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        try:
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)

    def Min(self, board: np.array, valid_actions: np.array, depth: int, level: int, alpha: float, beta: float,
            gain: bool, validactions):

        if depth == 0:
            count1: int = 0
            count2: int = 0
            evl_board = np.array(list(zip(*board.nonzero())))

            for row in evl_board:
                if board[row[0]][row[1]] == self._color:
                    count1 += 1
                else:
                    count2 += 1
            return count1 - count2
        MinBeta: int = beta
        min_evl = float('inf')
        player: int = self.getOpponent(self._color)
        thebest: np.array = None
        for Actions in validactions:
            newboard, newaction = self.createState(board, Actions, player)
            newmove = self.Max_value(newboard, newaction, depth - 1, level + 1, alpha, MinBeta, not gain)
        if min_evl > newmove:
            min_evl = newmove

            if level == 0:
                thebest = Actions

        MinBeta = min(MinBeta, newmove)
        if MinBeta <= alpha:
            return -1
        if level != 0:
            return min_evl
        else:
            return min_evl, thebest


    def Max(self, board: np.array, validactions: np.array, depth: int, level: int, alpha: float, beta: float, gain: bool):
        if depth == 0:
            count1: int = 0
            count2: int = 0
            evl_board = np.array(list(zip(*board.nonzero())))
            for row in evl_board:
                if board[row[0]][row[1]] == self._color:
                    count1 += 1
                else:
                    count2 += 1
            return count1 - count2

        thebest: np.array = None
        MaxAlpha: int = alpha
        max_evl = float('-inf')
        for Actions in validactions:
            newboard, newaction = self.createState(board, Actions, player)
            newmove = self.Min_value(newboard, newaction, depth - 1, level + 1, MaxAlpha, beta, not gain)
            if max_evl < newmove:
                max_evl = newmove

                if level == 0:
                    thebest = Actions
            MaxAlpha = max(MaxAlpha, max_evl)
            if beta <= MaxAlpha:
                break
        if level != 0:
            return max_evl
        else:
            max_evl, thebest


    @staticmethod
    def getOpponent(player: int):
        if player == 1:
            return -1
        else:
            return 1


    def createState(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        newState: np.array = transition(board, player, action)
        validMoves: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves: np.array = np.array(list(zip(*validMoves.nonzero())))
        return newState, validMoves
