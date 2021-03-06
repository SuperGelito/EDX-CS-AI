# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    ret = []
    state = problem.getStartState()
    closed = set()
    closed.add(state)
    st = util.Stack()
    for succ in problem.getSuccessors(state):
        st.push(succ)
    path = depthFirstSearchAlg(problem,st,[],closed)
    if path != False:
        for a in path:
           ret.append(a[1])
    return ret

def depthFirstSearchAlg(problem,fringe,path,closed):
    route = list(path)
    if fringe.isEmpty():
        return False
    for item in fringe.list:
        if item[0] not in closed:
            closed.add(item[0])
            routeitem = list(route) 
            routeitem.append(item)
            if problem.isGoalState(item[0]):
                return routeitem
            else:
                st = util.Stack()
                for succ in problem.getSuccessors(item[0]):
                    st.push(succ)
                ret = depthFirstSearchAlg(problem,st,routeitem,closed)
                if ret != False:
                    return ret
    return False

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    ret = []
    state = problem.getStartState()
    closed = set()
    closed.add(state)
    st = util.Stack()
    pts = []
    for succ in problem.getSuccessors(state):
        st.push(succ)
        pathElem = []
        pathElem.append(succ)
        pts.append(pathElem)
    path = breadthFirstSearchAlg(problem,st,pts,closed)
    if path != False:
        for a in path:
           ret.append(a[1])
    return ret

def breadthFirstSearchAlg(problem,fringe,paths,closed):
    if fringe.isEmpty():
        return False
    st = util.Stack()
    pt = []
    for i in range(0,len(fringe.list)):
        item = fringe.list[i]
        if item[0] not in closed:
            closed.add(item[0])
            if problem.isGoalState(item[0]):
                return paths[i]
            else:
                for succ in problem.getSuccessors(item[0]):
                    st.push(succ)
                    routeitem = list(paths[i]) 
                    routeitem.append(succ)
                    pt.append(routeitem)
    ret = breadthFirstSearchAlg(problem,st,pt,closed)
    if ret != False:
        return ret
    return False

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
   
    ret = []
    state = problem.getStartState()
    closed = set()
    closed.add(state)
    st = util.PriorityQueue()
    pts = {}
    for succ in problem.getSuccessors(state):
        st.push(succ,succ[2])
        pathElem = []
        pathElem.append(succ)
        pts[succ] = pathElem
    path = uniformSearchAlg(problem,st,pts,closed)
    if path != False:
        for a in path:
           ret.append(a[1])
    return ret

def uniformSearchAlg(problem,fringe,paths,closed):
    if fringe.isEmpty():
        return False
    while not fringe.isEmpty():
        node = fringe.pop()
        if node[0] not in closed:
            closed.add(node[0]) 
            nodePath = paths[node]
            if problem.isGoalState(node[0]):
                return nodePath
            else:
                for succ in problem.getSuccessors(node[0]):
                    if succ not in closed:
                        sucP = (succ[0],succ[1],succ[2] + node[2])
                        fringe.push(sucP,sucP[2])
                        nodePathChild = list(nodePath)
                        nodePathChild.append(sucP)
                        paths[sucP] = nodePathChild
                ret = uniformSearchAlg(problem,fringe,paths,closed)
                return ret
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    ret = []
    state = problem.getStartState()
    closed = set()
    closed.add(state)
    st = util.PriorityQueue()
    pts = {}
    succs = list(problem.getSuccessors(state))
    for succ in succs:
        h = heuristic(succ[0],problem)
        #sucP = (succ[0],succ[1],succ[2],h)
        sucP = (succ[0],succ[1],succ[2])
        st.push(sucP,sucP[2] + h)
        pathElem = []
        pathElem.append(succ)
        pts[succ] = pathElem
    path = aStarSearchAlg(problem,st,pts,closed,heuristic)
    aStarAdmisibleHeuristic(path)
    if path != False:
        for a in path:
           ret.append(a[1])
    return ret
    
def aStarSearchAlg(problem,fringe,paths,closed,heuristic):
    if fringe.isEmpty():
        return False
    while not fringe.isEmpty():
        node = fringe.pop()
        if node[0] not in closed:
            closed.add(node[0]) 
            nodePath = paths[node]
            if problem.isGoalState(node[0]):
                return nodePath
            else:
                for succ in problem.getSuccessors(node[0]):
                    if succ not in closed:
                        h = heuristic(succ[0],problem)
                        sucP = (succ[0],succ[1],succ[2] + node[2],h)
                        fringe.push(sucP,sucP[2] + h)
                        nodePathChild = list(nodePath)
                        nodePathChild.append(sucP)
                        paths[sucP] = nodePathChild
                ret = aStarSearchAlg(problem,fringe,paths,closed,heuristic)
                return ret
    return False

def aStarAdmisibleHeuristic(path):
    import math
    valid = True
    for i in range(0,len(path)):
        node = path[i]
        if len(node) > 3: 
            heur = node[3]
            costTotal = 0
            currentCost = 0
            for j in range(len(path)-1,i-1,-1):
                cost = path[j][2]
                if currentCost > 0:
                    costTotal = costTotal + math.fabs(currentCost - cost)
                currentCost = cost
            if heur > costTotal:
                valid = False
            print 'Position x:{},y:{},Cost to Goal {},Heuristic {}'.format(node[0][0][0],node[0][0][1],costTotal,heur)
            if not valid:
                print 'Heuristic invalid --> Position x:{},y:{},Cost to Goal {},Heuristic {}'.format(node[0][0][0],node[0][0][1],costTotal,heur)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
