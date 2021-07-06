from game import Agent, Directions, Actions
from featureExtractors import *
import random,util,time, math

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

    def update(self, state, action, nextState, reward):
        """
            This class will call this function, which you write, after
            observing a transition and reward
        """
        util.raiseNotDefined()

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actions_available(state)

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodes_so_far < self.num_training:
              self.accum_train_rewards += self.episodeRewards
        else:
              self.accum_test_rewards += self.episodeRewards
        self.episodes_so_far += 1
        if self.episodes_so_far >= self.num_training:
          # Take off the training wheels
          self.epsilon = 0.0    # no exploration
          self.alpha = 0.0      # no learning

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

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodes_so_far == 0:
            print("Beginning {} episodes of Training", self.num_training)

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodes_so_far % NUM_EPS_UPDATE == 0:
            print("Reinforcement Learning Status:")
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodes_so_far <= self.num_training:
                trainAvg = self.accum_train_rewards / float(self.episodes_so_far)
                print("\tCompleted {} out of {} training episodes".format(self.episodes_so_far, self.num_training))
                print("\tAverage Rewards over all training: {}", trainAvg)
            else:
                testAvg = float(self.accum_test_rewards) / (self.episodes_so_far - self.num_training)
                print("\nCompleted {} test episodes", self.episodes_so_far - self.num_training)
                print("\nAverage Rewards over testing: {}", testAvg)
            print("\nAverage Rewards for last {} episodes: {}".format(NUM_EPS_UPDATE, windowAvg))
            print("\nEpisode took {} seconds", time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
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

        self.qValues = util.Counter()
        print("ALPHA: {}".format(self.alpha))
        print("DISCOUNT: {}".format(self.discount))
        print("EXPLORATION: {}".format(self.epsilon))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        possibleStateQValues = util.Counter()
        for action in self.getLegalActions(state):
            possibleStateQValues[action] = self.getQValue(state, action)

        return possibleStateQValues[possibleStateQValues.argMax()]

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleStateQValues = util.Counter()
        possibleActions = self.getLegalActions(state)
        if len(possibleActions) == 0:
            return None

        for action in possibleActions:
            possibleStateQValues[action] = self.getQValue(state, action)

        if possibleStateQValues.totalCount() == 0:
            return random.choice(possibleActions)
        else:
            return possibleStateQValues.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        print("State: {}. Action: {}. NextState: {}. Reward: {}".format(state, action, nextState, reward))
        print("QVALUE: {}", self.getQValue(state, action))
        print("VALUE: {}", self.getValue(nextState))
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
                    reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
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

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        possibleStateQValues = []
        for act in self.getLegalActions(state):
            possibleStateQValues.append(self.getQValue(state, act))
        for key in features.keys():
            self.weights[key] += self.alpha * (reward + self.discount * (
                        (1 - self.epsilon) * self.getValue(nextState) + (self.epsilon / len(possibleStateQValues)) * (
                    sum(possibleStateQValues))) - self.getQValue(state, action)) * features[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
