import math
import random
import time

import util
import utils
from featureExtractors import FeatureExtractor
from game import Agent, Directions, Actions

"""
The functions available from now on are the ones used to implement the Q-Values Reinforcement Agent
"""


class ReinforcementAgent(Agent):
    """
    The Reinforcement Agent estimates Q-Values (as well as policies) from experience
    """

    def __init__(self, actions_available=None, num_training=10, epsilon=0.5, alpha=0.5, gamma=1):
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

        self.episodes_so_far = 0

        self.number_of_actions_taken = 0
        self.actions_available = actions_available

        self.num_training = int(num_training)
        self.epsilon = float(epsilon)  # exploration rate
        self.alpha = float(alpha)  # learning rate
        self.discount = float(gamma)  # discount factor

        self.last_state = None
        self.last_action = None

        self.accum_train_score = 0.0
        self.accum_test_score = 0.0
        self.episode_score = 0.0

    def update(self, state, action, next_state, reward):
        pass  # not implemented here

    def getLegalActions(self, state):
        # Get the actions available for a given state
        return self.actions_available(state)

    def observeTransition(self, state, action, next_state, delta_reward):
        """
            Called to inform the agent that a transition has been observed.
            We will call self.update with the same arguments
        """
        self.episode_score += delta_reward
        self.update(state, action, next_state, delta_reward)

    def startEpisode(self):
        self.last_state = None
        self.last_action = None
        self.episode_score = 0.0
        self.number_of_actions_taken = 0

    def stopEpisode(self):
        print("Ending episode: {}".format(self.episodes_so_far + 1))
        print('Score: {}'.format(self.episode_score))
        message = 'Number of actions taken: {}'.format(self.number_of_actions_taken)
        print(message)
        print("-" * len(message))

        utils.set_globals(self.episodes_so_far, self.number_of_actions_taken, self.episode_score)

        if self.episodes_so_far < self.num_training:
            self.accum_train_score += self.episode_score
        else:
            self.accum_test_score += self.episode_score
        self.episodes_so_far += 1
        if self.episodes_so_far >= self.num_training:
            # now we remove the parameters
            self.epsilon = 0.0
            self.alpha = 0.0

    def doAction(self, state, action):
        self.number_of_actions_taken += 1
        self.last_state = state
        self.last_action = action

    def observationFunction(self, state):
        # Called right after the last action
        if not self.last_state is None:
            reward = state.getScore() - self.last_state.getScore()
            self.observeTransition(self.last_state, self.last_action, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodes_so_far == 0:
            message = "Beginning {} episodes of Training".format(self.num_training)
            print('-' * len(message))
            print(message)
            print('-' * len(message))

    def final(self, state):
        # finishes the game
        delta_reward = state.getScore() - self.last_state.getScore()
        self.observeTransition(self.last_state, self.last_action, state, delta_reward)
        self.stopEpisode()

        if self.episodes_so_far == self.num_training:
            msg = 'Finished training'
            print("{}\n{}".format(msg, '-' * len(msg)))


class QLearningAgent(ReinforcementAgent):
    """
      This is the Q-Learning Agent
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

        self.q_values = util.Counter()
        print("alpha/learning rate: {}".format(self.alpha))
        print("gamma/discount: {}".format(self.discount))
        print("epsilon/exploration: {}".format(self.epsilon))

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
    """

    def __init__(self, extractor='FeatureExtractor', epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        self.featExtractor = FeatureExtractor()

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # Pacman initial position
        QLearningAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        # return Q(state,action) = w * featureVector
        q_value = 0.0
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            q_value += (self.weights[key] * features[key])
        return q_value

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
        # Should update your weights based on transition
        # we get the features based on the current state and action, ignoring the past
        # this allows us to have a huge performance advantage
        features = self.featExtractor.getFeatures(state, action)
        possible_state_q_values = []

        for act in self.getLegalActions(state):
            possible_state_q_values.append(self.getQValue(state, act))
        for key in features.keys():
            self.weights[key] += self.alpha * (reward + self.discount * (
                    (1 - self.epsilon) * self.getValue(next_state) + (self.epsilon / len(possible_state_q_values)) * (
                sum(possible_state_q_values))) - self.getQValue(state, action)) * features[key]

    def final(self, state):
        # called to end the game
        QLearningAgent.final(self, state)
