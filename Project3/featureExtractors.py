import util
from game import Directions, Actions

"""
In this file, we implement the class feature extractor, which retrieves
the information for all pacman game states
"""


class FeatureExtractor:
    """
    Returns basic game features for Pacman
        - distance of nearest food
        - if a ghost is too close
    """

    def __init__(self):
        pass

    def is_food_close(self, pacman_future_location, food, walls):
        food_available = [(pacman_future_location[0], pacman_future_location[1], 0)]
        visited = set()

        while food_available:
            pacman_future_location_x, pacman_future_location_y, dist = food_available.pop(0)
            if (pacman_future_location_x, pacman_future_location_y) in visited:
                continue
            visited.add((pacman_future_location_x, pacman_future_location_y))
            # return if we find food at this position
            if food[pacman_future_location_x][pacman_future_location_y]:
                return dist
            # if not, we will search the neighbours positions
            nbrs = Actions.getLegalNeighbors((pacman_future_location_x, pacman_future_location_y), walls)
            for nbr_x, nbr_y in nbrs:
                food_available.append((nbr_x, nbr_y, dist + 1))
        # return None if no more food is found
        return None

    def map_manhattan_distances(self, pacman_position, ghosts):
        return map(lambda g: util.manhattanDistance(pacman_position, g.getPosition()), ghosts)

    def getFeatures(self, state, action):
        """
            extract the following information:
                grid of food
                wall locations and
                get the ghost locations
        """
        available_food = state.getFood()
        walls_positions = state.getWalls()
        ghosts_positions = state.getGhostPositions()

        capsules_left = len(state.getCapsules())
        scared_ghost = []
        active_ghost = []

        features = util.Counter()

        for ghost in state.getGhostStates():
            if not ghost.scaredTimer:
                active_ghost.append(ghost)
            else:
                scared_ghost.append(ghost)

        pacman_location = state.getPacmanPosition()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(ghost, walls_positions) for ghost in ghosts_positions)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and available_food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.is_food_close((next_x, next_y), available_food, walls_positions)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls_positions.width * walls_positions.height)

        if scared_ghost:
            distance_to_closest_scared_ghost = min(self.map_manhattan_distances(pacman_location, scared_ghost))
            if active_ghost:
                distance_to_closest_active_ghost = min(self.map_manhattan_distances(pacman_location, active_ghost))
            else:
                distance_to_closest_active_ghost = 10

            features["capsules"] = capsules_left

            if distance_to_closest_scared_ghost <= 8 and distance_to_closest_active_ghost >= 2:
                features["#-of-ghosts-1-step-away"] = 0
                features["eats-food"] = 0.0

        features.divideAll(10.0)
        return features
