import math
import random
import time

import util
from featureExtractors import *
from game import Agent, Directions, Actions

"""
This first two agents were implemented to help us understand how the codebase worked, based on labs
available at https://cs.brynmawr.edu/Courses/cs372/fall2017/Lab1.pdf

The actual solution for the Reinforcement Learning will be at the remaining classes
"""


class RandomAgent(Agent):
    """
    The random agent just calls a possible solution randomly
    """

    def __init__(self):
        super().__init__()
        self.__location = None

    def getAction(self, state):
        actions_available = state.getLegalPacmanActions()
        del actions_available[-1]
        print("Type of {}. Actions available: {}".format(type(actions_available), actions_available))
        action = random.choice(actions_available)
        print("Picked action: ", action)
        if action == Directions.WEST:
            return Directions.WEST
        elif action == Directions.SOUTH:
            return Directions.SOUTH
        elif action == Directions.EAST:
            return Directions.EAST
        elif action == Directions.NORTH:
            return Directions.NORTH
        else:
            return Directions.STOP


class ReflexAgent(Agent):
    """
    The reflex agent calls one of possible direction if they have food
    """

    def __init__(self):
        super().__init__()
        self.__location = None

    def choose_random_direction(self, actions_available):
        action = random.choice(actions_available)
        print("Picked action randomly: ", action)
        if action == Directions.WEST:
            return Directions.WEST
        elif action == Directions.SOUTH:
            return Directions.SOUTH
        elif action == Directions.EAST:
            return Directions.EAST
        elif action == Directions.NORTH:
            return Directions.NORTH
        else:
            return Directions.STOP

    def getAction(self, state):
        location = state.getPacmanPosition()
        print('Location: {}', location)
        actions_available = state.getLegalPacmanActions()
        del actions_available[-1]

        for action in actions_available:
            if action == Directions.WEST:
                if state.hasFood(location[0] - 1, location[1]):
                    return Directions.WEST
            elif action == Directions.SOUTH:
                if state.hasFood(location[0], location[1] - 1):
                    return Directions.SOUTH
            elif action == Directions.EAST:
                if state.hasFood(location[0] + 1, location[1]):
                    return Directions.EAST
            elif action == Directions.NORTH:
                if state.hasFood(location[0], location[1] + 1):
                    return Directions.NORTH

        return self.choose_random_direction(actions_available)
