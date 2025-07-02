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
from game import Directions, Actions
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        foodList = newFood.asList()
        currentFoodList = currentGameState.getFood().asList()
        
        if foodList:
            closestFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 20.0 / (closestFoodDistance + 1)
        
        if len(foodList) < len(currentFoodList):
            score += 200
            
        if len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()):
            score += 300
        
        minGhostDistance = float('inf')
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            minGhostDistance = min(minGhostDistance, ghostDistance)
            
            if newScaredTimes[i] > 0:
                score += 150.0 / (ghostDistance + 1)
            else: 
                if ghostDistance <= 1:
                    score -= 500 
                elif ghostDistance == 2:
                    score -= 50 
        
        if minGhostDistance >= 3:
            if foodList:
                closestFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
                score += 30.0 / (closestFoodDistance + 1) 
        
        currentPos = currentGameState.getPacmanPosition()
        if action != Directions.STOP:
            actions = currentGameState.getLegalActions()
            if len(actions) > 2:
                actionVector = Actions.directionToVector(action)
                newPosFromAction = (currentPos[0] + actionVector[0], currentPos[1] + actionVector[1])
                if newPosFromAction != newPos:
                    pass
                else:
                    score -= 5
        
        if action == Directions.STOP and minGhostDistance > 2:
            score -= 100
        elif action == Directions.STOP:
            score -= 10 
            
        return score

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
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                maxEval = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    eval_score = minimax(successor, nextDepth, nextAgent)
                    maxEval = max(maxEval, eval_score)
                return maxEval
            else:
                minEval = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    eval_score = minimax(successor, nextDepth, nextAgent)
                    minEval = min(minEval, eval_score)
                return minEval

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

