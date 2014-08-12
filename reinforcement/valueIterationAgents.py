# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        #Values of the status
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.valuesTemp = self.values.copy()

        for i in range(0, iterations, 1):
            states = mdp.getStates()
            for state in states:
                if not mdp.isTerminal(state):
                    actions = mdp.getPossibleActions(state)
                    actVals = []
                    for action in actions:
                        qVal = self.computeQValueFromValues(state, action)
                        actVals.append(qVal)
                    self.valuesTemp[state] = max(actVals)
            self.values = self.valuesTemp.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        # Apply Bellman Equation to get QValue
        tempValues = self.values
        possibleStates = self.mdp.getTransitionStatesAndProbs(state, action)
        val = 0
        for posState in possibleStates:
            nextState = posState[0]
            prob = posState[1]
            rew = self.mdp.getReward(state, action, nextState)
            val += prob * (rew + (self.discount * tempValues[nextState]))
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        retAction = None
        valAct = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            val = self.computeQValueFromValues(state, action)
            valAct[action] = val
        retAction = valAct.argMax()
        return retAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
