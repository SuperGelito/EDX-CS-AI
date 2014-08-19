# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)


        self.values = util.Counter()
        if 'epsilon' in args:
            self.epsilon = args['epsilon']
        else:
            self.epsilon = 0
        if 'gamma' in args:
            self.gamma = args['gamma']
        else:
            self.gamma = 0

        if 'alpha' in args:
            self.alpha = args['alpha']
        else:
            self.alpha = 0



    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        valState = self.values[state]
        ret = 0.0
        if valState != 0.0:
            ret = valState[action]
        return ret


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        ret = 0.0
        legalActions = self.getLegalActions(state)
        qValLegalActions = util.Counter()
        for legalAction in legalActions:
            qValLegalAction = self.getQValue(state, legalAction)
            qValLegalActions[legalAction] = qValLegalAction
        if len(qValLegalActions) > 0:
            ret = qValLegalActions[qValLegalActions.argMax()]
        return ret

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        ret = None
        legalActions = self.getLegalActions(state)
        qValLegalActions = util.Counter()
        for legalAction in legalActions:
            qValLegalAction = self.getQValue(state, legalAction)
            qValLegalActions[legalAction] = qValLegalAction
        if len(qValLegalActions) > 0:
            ret = qValLegalActions.argMax()
        return ret

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
        qValLegalActions = util.Counter()
        for legalAction in legalActions:
            qValLegalAction = self.getQValue(state, legalAction)
            qValLegalActions[legalAction] = qValLegalAction

        action = None
        if len(qValLegalActions) > 0:
            action = qValLegalActions.argMax()
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        valState = self.values[state]
        qValToUpdate = 0.0
        if valState != 0.0:
            qValToUpdate = valState[action]
        qValMaxNextState = 0.0
        valNextState = self.values[nextState]
        if valNextState != 0.0:
            qValMaxNextState = valNextState[valNextState.argMax()]
        qValToUpdate += self.alpha * ((reward + self.gamma * qValMaxNextState) - qValToUpdate)
        # qValToUpdate += (1 - self.alpha) * qValToUpdate + self.alpha * (reward + self.gamma * qValMaxNextState)
        if valState == 0.0:
            valState = util.Counter()
            for la in self.getLegalActions(state):
                valState[la] = 0.0
        valState[action] = qValToUpdate
        self.values[state] = valState

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
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
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        sumValFeatures = 0.0
        for key in features:
            w = self.weights[key]
            f = features[key]
            sumValFeatures += w * f
        return sumValFeatures

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        nextLegalActions = self.getLegalActions(nextState)
        nextActions = util.Counter()
        for nextLegalAction in nextLegalActions:
            nextActions[nextLegalAction] = self.getQValue(nextState, nextLegalAction)

        difference = (reward + self.gamma * nextActions[nextActions.argMax()]) - self.getQValue(state, action)

        features = self.featExtractor.getFeatures(state, action)
        for key in features:
            w = self.weights[key]
            f = features[key]
            w += self.alpha * difference * f
            self.weights[key] = w

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            for key in self.weights:
                print 'Game Ended'
                print 'Feature %s Weight %s' % (key, self.weights[key])
            pass
