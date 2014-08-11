# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        ghostsPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.getWalls();
        capsules = successorGameState.getCapsules();
        maxEval = 10000

        #1! Calculate pellets points to substract from max Eval 100 importance
        foodPath = newFood.asList()
        foodPoints = len(foodPath) * 100

        #2! Calculate distance from food 1 importance 5 if there are walls
        minDistance = 0
        nearestFood = None
        for f in foodPath:
            d = util.manhattanDistance(newPos,f)
            if f[0] == newPos[0]:
                for i in range(newPos[1],f[1]):
                     if walls[newPos[0]][i] == True:
                        d = d+4
            if f[1] == newPos[1]:
                for i in  range(newPos[0],f[0]):
                    if walls[i][newPos[1]] == True:
                        d = d+4
            if minDistance == 0 or d < minDistance:
                minDistance = d
                nearestFood = f
        
        #override the manhatan distance with Maze distance
        import searchAgents
        if len(foodPath) > 0:
            minDistance = searchAgents.mazeDistance(newPos,nearestFood,successorGameState)

        import math
        #Calculate ghots (convert to negative) importance 9000
        
        gPoints = 0
        i=0
        for g in ghostsPositions:
            #Using BFS only when problems of near ghosts
            gpoint = 0
            securityDistance = 3
            multiplier = 1
            if newScaredTimes[i] != 0:
                securityDistance = securityDistance * 3
                multiplier = (-1/3)
            distance = util.manhattanDistance(newPos,g)
            if distance < securityDistance:
                gNew = (int(g[0]),int(g[1]))
                distance = searchAgents.mazeDistance(newPos,gNew,successorGameState)
                if distance <= securityDistance:
                    if distance == 0:
                        distance = 0.5
                    gpoint = 3000 * (securityDistance/distance)
                    gpoint = gpoint * multiplier
            gPoints = gPoints + gpoint
            i=i+1
        ret = maxEval -gPoints -foodPoints -minDistance
        print "Position:{0}, Ghost Point:{1}, FoodPoint:{2}, Distance points:{3}, Total:{4}".format(newPos,gPoints,foodPoints,minDistance,ret)
        return ret

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return MinimaxAgent.miniMaxSearch(self,gameState,0,1)[0]

    def miniMaxSearch(self,gameState,currentIndex,currentDepth):
        import copy
        #Check if we need to reset the depth and index
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth = currentDepth + 1
        #Legal actions of the current state
        actions = gameState.getLegalActions(currentIndex)
        #Default min Agent
        MaxAgent = False
        #If agent is Pacman then is a max agent
        if currentIndex == 0:
            MaxAgent = True
        #This is the current option not initialized, we will return this as action
        currentOption = None
        #Go over legal actions
        for ac in actions:
            #Create the next state based on current action
            sGameState = gameState.generateSuccessor(currentIndex, ac)
            #if we are in max depth we evaluate the state
            childOption = None
            if (currentDepth == self.depth and currentIndex == gameState.getNumAgents() - 1) or sGameState.isWin() or sGameState.isLose():
                childOption = (ac, self.evaluationFunction(sGameState))
            else:
                #if depth is not maximum we set the value based on its children then we evaluate the branch, this will make the same option
                childOption = MinimaxAgent.miniMaxSearch(self,sGameState,currentIndex+1,copy.copy(currentDepth))
            if childOption!=None:
                acVal = (ac,childOption[1])
                select = False
                if currentOption == None:
                    select = True
                else:
                    if (MaxAgent and childOption[1] > currentOption[1]) or (not MaxAgent and childOption[1] < currentOption[1]):
                        select = True
                if select:
                    currentOption = acVal
        return currentOption


                                                
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return AlphaBetaAgent.alphaBetaSearch(self,gameState,0,1,-100000,100000)[0]

    def alphaBetaSearch(self,gameState,currentIndex,currentDepth,alpha,beta):
        import copy
        #Check if we need to reset the depth and index
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth = currentDepth + 1
        #Legal actions of the current state
        actions = gameState.getLegalActions(currentIndex)
        #Default min Agent
        MaxAgent = False
        #If agent is Pacman then is a max agent
        if currentIndex == 0:
            MaxAgent = True
        #This is the current option not initialized, we will return this as action
        currentOption = None
        if(MaxAgent):
            currentOption = (None,-100000)
        else:
            currentOption = (None,100000)
        #Go over legal actions
        for ac in actions:
            #Create the next state based on current action
            sGameState = gameState.generateSuccessor(currentIndex, ac)
            #if we are in max depth we evaluate the state
            childOption = None
            if (MaxAgent and currentOption[1] < beta) or (not MaxAgent and currentOption[1] > alpha) :
                if (currentDepth == self.depth and currentIndex == gameState.getNumAgents() - 1) or sGameState.isWin() or sGameState.isLose():
                    childOption = (ac, self.evaluationFunction(sGameState))
                else:
                    #if depth is not maximum we set the value based on its children then we evaluate the branch, this will make the same option
                    childOption = AlphaBetaAgent.alphaBetaSearch(self,sGameState,currentIndex+1,copy.copy(currentDepth),copy.copy(alpha),copy.copy(beta))
            if childOption!=None:
                acVal = (ac,childOption[1])
                select = False
                if (MaxAgent and childOption[1] > currentOption[1]):
                    currentOption = acVal
                    alpha = childOption[1]
                elif (not MaxAgent and childOption[1] < currentOption[1]):
                    currentOption = acVal
                    beta = childOption[1]
        return currentOption
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return ExpectimaxAgent.expectiMaxSearch(self,gameState,0,1)[0]

    def expectiMaxSearch(self,gameState,currentIndex,currentDepth):
        import copy
        #Check if we need to reset the depth and index
        if currentIndex >= gameState.getNumAgents():
            currentIndex = 0
            currentDepth = currentDepth + 1
        #Legal actions of the current state
        actions = gameState.getLegalActions(currentIndex)
        #Default min Agent
        MaxAgent = False
        #If agent is Pacman then is a max agent
        if currentIndex == 0:
            MaxAgent = True
        #This is the current option not initialized, we will return this as action
        currentOption = None
        sumVal = 0.0
        #Go over legal actions
        for ac in actions:
            #Create the next state based on current action
            sGameState = gameState.generateSuccessor(currentIndex, ac)
            #if we are in max depth we evaluate the state
            childOption = None
            if (currentDepth == self.depth and currentIndex == gameState.getNumAgents() - 1) or sGameState.isWin() or sGameState.isLose():
                childOption = (ac, self.evaluationFunction(sGameState))
            else:
                #if depth is not maximum we set the value based on its children then we evaluate the branch, this will make the same option
                childOption = ExpectimaxAgent.expectiMaxSearch(self,sGameState,currentIndex+1,copy.copy(currentDepth))
            if childOption!=None:
                acVal = (ac,childOption[1])
                select = False
                if currentOption == None:
                    select = True
                else:
                    if MaxAgent and childOption[1] > currentOption[1]: 
                        select = True
                if  not MaxAgent:
                    select = True
                    sumVal = sumVal + childOption[1]
                if select:
                    currentOption = acVal
        if not MaxAgent:
            currentOption = (currentOption[0], sumVal / float(len(actions)))
        return currentOption

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
     # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    ghostsPositions = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls();
    capsules = currentGameState.getCapsules();
    maxEval = 10000

    #1! Calculate pellets points to substract from max Eval 100 importance
    foodPath = newFood.asList()
    foodPoints = len(foodPath) * 100

    #2! Calculate distance from food 1 importance 5 if there are walls
    minDistance = 0
    nearestFood = None
    for f in foodPath:
        d = util.manhattanDistance(newPos,f)
        if f[0] == newPos[0]:
            for i in range(newPos[1],f[1]):
                    if walls[newPos[0]][i] == True:
                        d = d+4
        if f[1] == newPos[1]:
            for i in  range(newPos[0],f[0]):
                if walls[i][newPos[1]] == True:
                    d = d+4
        if minDistance == 0 or d < minDistance:
            minDistance = d
            nearestFood = f
        
    #override the manhatan distance with Maze distance
    import searchAgents
    if len(foodPath) > 0:
        minDistance = searchAgents.mazeDistance(newPos,nearestFood,currentGameState)

    import math
    #Calculate ghots (convert to negative) importance 9000
        
    gPoints = 0
    i=0
    for g in ghostsPositions:
        #Using BFS only when problems of near ghosts
        gpoint = 0
        securityDistance = 3
        multiplier = 1
        if newScaredTimes[i] != 0:
            securityDistance = securityDistance * 3
            multiplier = (-1/3)
        distance = util.manhattanDistance(newPos,g)
        if distance < securityDistance:
            gNew = (int(g[0]),int(g[1]))
            distance = searchAgents.mazeDistance(newPos,gNew,currentGameState)
            if distance <= securityDistance:
                if distance == 0:
                    distance = 0.5
                gpoint = 3000 * (securityDistance/distance)
                gpoint = gpoint * multiplier
        gPoints = gPoints + gpoint
        i=i+1
    ret = maxEval -gPoints -foodPoints -minDistance
    print "Position:{0}, Ghost Point:{1}, FoodPoint:{2}, Distance points:{3}, Total:{4}".format(newPos,gPoints,foodPoints,minDistance,ret)
    return ret
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

