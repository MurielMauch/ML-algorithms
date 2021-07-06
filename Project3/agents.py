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


"""
The functions available from now on are the ones used to implement the Q-Values Reinforcement Agent
"""


class ReinforcementAgent(Agent):
    """
    The Reinforcement Agent estimates Q-Values (as well as policies) from experience
    """

    def __init__(self, actions_available=None, num_training=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        Instantiates the Reinforcement Agent
        
        Keyword Arguments:
            actions_available: function that takes a state and return the legal actions
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            num_training: number of training episodes
        """
        if actions_available is None:
            actions_available = lambda state: state.getLegalActions()
        self.actions_available = actions_available
        self.episodes_so_far = 0
        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.num_training = int(num_training)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def update(self, state, action, next_state, reward):
        pass  # not implemented here

    def getLegalActions(self, state):
        """
          Get the actions available for a given state
        """
        return self.actions_available(state)

    def observeTransition(self, state, action, next_state, delta_reward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episode_rewards += delta_reward
        self.update(state, action, next_state, delta_reward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
          Making sure we have the default values set
        """
        self.last_state = None
        self.last_action = None
        self.episode_rewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodes_so_far < self.num_training:
            self.accum_train_rewards += self.episode_rewards
        else:
            self.accum_test_rewards += self.episode_rewards
        self.episodes_so_far += 1
        if self.episodes_so_far >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def isInTraining(self):
        return self.episodes_so_far < self.num_training

    def isInTesting(self):
        return not self.isInTraining()

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.last_state = state
        self.last_action = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            Called right after the last action
        """
        if not self.last_state is None:
            reward = state.getScore() - self.last_state.getScore()
            self.observeTransition(self.last_state, self.last_action, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodes_so_far == 0:
            print("Beginning {} episodes of Training".format(self.num_training))

    def final(self, state):
        """
          Ends the game
        """
        delta_reward = state.getScore() - self.last_state.getScore()
        self.observeTransition(self.last_state, self.last_action, state, delta_reward)
        self.stopEpisode()

        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.last_window_accum_rewards = 0.0
        self.last_window_accum_rewards += state.getScore()

        num_eps_update = 100
        if self.episodes_so_far % num_eps_update == 0:
            print("Reinforcement Learning Status:")
            window_avg = self.last_window_accum_rewards / float(num_eps_update)
            if self.episodes_so_far <= self.num_training:
                train_avg = self.accum_train_rewards / float(self.episodes_so_far)
                print("\tCompleted {} out of {} training episodes".format(self.episodes_so_far, self.num_training))
                print("\tAverage Rewards over all training: {}".format(train_avg))
            else:
                test_avg = float(self.accum_test_rewards) / (self.episodes_so_far - self.num_training)
                print("\nCompleted {} test episodes".format(self.episodes_so_far - self.num_training))
                print("\nAverage Rewards over testing: {}".format(test_avg))
            print("\nAverage Rewards for last {} episodes: {}".format(num_eps_update, window_avg))
            print("\nEpisode took {} seconds".format(time.time() - self.episodeStartTime))
            self.last_window_accum_rewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodes_so_far == self.num_training:
            msg = 'Training Done (turning off epsilon and alpha)'
            print("{}\n{}".format(msg, '-' * len(msg)))


class QLearningAgent(ReinforcementAgent):
    """
      This is the Q-Learning Agent
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

        self.q_values = util.Counter()
        print("ALPHA: {}".format(self.alpha))
        print("DISCOUNT: {}".format(self.discount))
        print("EXPLORATION: {}".format(self.epsilon))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen a state or (state,action) tuple
        """
        return self.q_values[(state, action)]

    def getValue(self, state):
        """
          Returns max_action Q(state,action) where the max is over legal actions. 
          Return 0.0 if final state
        """
        possible_state_q_values = util.Counter()
        for action in self.getLegalActions(state):
            possible_state_q_values[action] = self.getQValue(state, action)

        return possible_state_q_values[possible_state_q_values.argMax()]

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  
          Returns none if final state
        """
        possible_state_q_values = util.Counter()
        possible_actions = self.getLegalActions(state)
        if len(possible_actions) == 0:
            return None

        for action in possible_actions:
            possible_state_q_values[action] = self.getQValue(state, action)

        if possible_state_q_values.totalCount() == 0:
            return random.choice(possible_actions)
        else:
            return possible_state_q_values.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  
          With probability self.epsilon >> take a random action 
          otherwise >> take the best policy action 
          None if terminal state
        """
        legal_actions = self.getLegalActions(state)
        action = None
        if len(legal_actions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legal_actions)
            else:
                action = self.getPolicy(state)

        return action

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a state = action => nextState and reward transition.
          Here is where we update our Q-Value
        """
        print("State: {}. Action: {}. NextState: {}. Reward: {}".format(state, action, next_state, reward))
        print("QVALUE: {}".format(self.getQValue(state, action)))
        print("VALUE: {}".format(self.getValue(next_state)))
        self.q_values[(state, action)] = self.getQValue(state, action) + self.alpha * (
                reward + self.discount * self.getValue(next_state) - self.getQValue(state, action))


class ApproximateQAgent(QLearningAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        self.featExtractor = util.lookup(extractor, globals())()

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # Pacman initial position
        QLearningAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qValue = 0.0
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            qValue += (self.weights[key] * features[key])
        return qValue

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        possible_state_q_values = []
        for act in self.getLegalActions(state):
            possible_state_q_values.append(self.getQValue(state, act))
        for key in features.keys():
            self.weights[key] += self.alpha * (reward + self.discount * (
                    (1 - self.epsilon) * self.getValue(next_state) + (self.epsilon / len(possible_state_q_values)) * (
                sum(possible_state_q_values))) - self.getQValue(state, action)) * features[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        QLearningAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
