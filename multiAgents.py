# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        availableFood = currentGameState.getFood().asList()
        minDist2Food = min([manhattanDistance(newPos, f) for f in availableFood]) if availableFood else 0
        dist2Ghost = manhattanDistance(newPos, newGhostStates[0].getPosition()) if newGhostStates else 0
        #dist2Capsule = manhattanDistance(newPos, successorGameState.getCapsules())
        if dist2Ghost > 1:
            return successorGameState.getScore() - minDist2Food 
        return -100000
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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
    def __getValue(self, gameState: GameState, level: int, treeDepth: int, numAgents: int):
        if level == treeDepth or (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        agentIndex = level % numAgents
        legalActions = gameState.getLegalActions(agentIndex)
        nextStates = [gameState.generateSuccessor(agentIndex, a) for a in legalActions]
        nextStateValues = [self.__getValue(nextState, level + 1, treeDepth, numAgents) for nextState in nextStates]
        if agentIndex == 0:
            # max node: Agent
            return max(nextStateValues)
        else:
            # min node: Adversary
            return min(nextStateValues)
        


    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        treeDepth = self.depth * numAgents
        legalActions = gameState.getLegalActions()
        nextStates = [gameState.generateSuccessor(0, a) for a in legalActions]
        nextStateValues = []
        nextStateValues = [self.__getValue(nextState, 1, treeDepth, numAgents) for nextState in nextStates]
        maxValue = max(nextStateValues)
        actionIndex = nextStateValues.index(maxValue)
        #actionIndex = random.choice([i for i, val in enumerate(nextStateValues) if val == maxValue])
        return legalActions[actionIndex]      

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def __getValue(self, gameState: GameState, alpha, beta, level: int, treeDepth: int, numAgents: int):
        if level == treeDepth or (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        agentIndex = level % numAgents
        legalActions = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            # max node: Agent
            v = -1000000
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, self.__getValue(nextState, alpha, beta, level + 1, treeDepth, numAgents))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:
            # min node: Adversary
            v = 1000000
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                v = min(v, self.__getValue(nextState, alpha, beta, level + 1, treeDepth, numAgents))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        treeDepth = self.depth * numAgents
        legalActions = gameState.getLegalActions()
        nextStates = [gameState.generateSuccessor(0, a) for a in legalActions]
        maxValue = -1000000
        actionIndex = -1
        for i, nextState in enumerate(nextStates):
            value = self.__getValue(nextState, maxValue, 1000000, 1, treeDepth, numAgents)
            if value > maxValue:
                maxValue = value
                actionIndex = i
        return legalActions[actionIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def __getValue(self, gameState: GameState, level: int, treeDepth: int, numAgents: int):
        if level == treeDepth or (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        agentIndex = level % numAgents
        legalActions = gameState.getLegalActions(agentIndex)
        nextStates = [gameState.generateSuccessor(agentIndex, a) for a in legalActions]
        nextStateValues = [self.__getValue(nextState, level + 1, treeDepth, numAgents) for nextState in nextStates]
        if agentIndex == 0:
            # max node: Agent
            return max(nextStateValues)
        else:
            # min node: Adversary
            return sum(nextStateValues) / len(nextStateValues)
        
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        treeDepth = self.depth * numAgents
        legalActions = gameState.getLegalActions()
        nextStates = [gameState.generateSuccessor(0, a) for a in legalActions]
        nextStateValues = []
        nextStateValues = [self.__getValue(nextState, 1, treeDepth, numAgents) for nextState in nextStates]
        maxValue = max(nextStateValues)
        actionIndex = nextStateValues.index(maxValue)
        #actionIndex = random.choice([i for i, val in enumerate(nextStateValues) if val == maxValue])
        return legalActions[actionIndex]     

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsule = currentGameState.getCapsules()
    # food score
    dist_f = min([(manhattanDistance(pos, f), f) for f in food]) if food else (0, None)
    food_score = 0.05 * -dist_f[0] + 0.95 * -len(food)
    # capsule score
    dist_c = min([(manhattanDistance(pos, c), c) for c in capsule]) if capsule else (0, None)
    capsule_score = 0.05 * -dist_c[0] + 0.95 * -len(capsule)
    return currentGameState.getScore() + 100 * food_score + 500 * capsule_score
# Abbreviation
better = betterEvaluationFunction
