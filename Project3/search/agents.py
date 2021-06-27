from game import Agent
from game import Directions
import random


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
        self.__location = None

    def getAction(self, state):
        actions_available = state.getLegalPacmanActions()
        del actions_available[-1]
        print(f"Type of {type(actions_available)}. Actions available: {actions_available}")
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
        print(f'Location: {location}')
        actions_available = state.getLegalPacmanActions()
        del actions_available[-1]

        for action in actions_available:
            if action == Directions.WEST:
                if state.hasFood(location[0]-1, location[1]):
                    return Directions.WEST
            elif action == Directions.SOUTH:
                if state.hasFood(location[0], location[1]-1):
                    return Directions.SOUTH
            elif action == Directions.EAST:
                if state.hasFood(location[0]+1, location[1]):
                    return Directions.EAST
            elif action == Directions.NORTH:
                if state.hasFood(location[0], location[1]+1):
                    return Directions.NORTH

        return self.choose_random_direction(actions_available)
