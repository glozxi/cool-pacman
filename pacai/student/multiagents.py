import random
import math

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

from pacai.core.distance import manhattan
from pacai.core.actions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        nonScaredGhostPos = [
            state.getPosition() for state in newGhostStates if not state.isScared()]
        if nonScaredGhostPos:
            distToGhost = min(manhattan(
                newPosition, tuple(map(int, pos))) for pos in nonScaredGhostPos)
            if distToGhost == 0:
                distToGhost = 0.0001
        else:
            distToGhost = math.inf
        foodList = currentGameState.getFood().asList()
        succFoodNum = len(successorGameState.getFood().asList())
        distToFood = min(manhattan(newPosition, foodPos) for foodPos in foodList)
        return -2.0 / distToGhost - 2 * distToFood - 4 * succFoodNum

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        
        def maxValue(state, depth):
            if depth == self._treeDepth or not state.getLegalActions(0):
                return self.getEvaluationFunction()(state)
            max = -math.inf
            for action in state.getLegalActions(0):
                if action == Directions.STOP:
                    continue
                val = minValue(state.generateSuccessor(0, action), 1, depth)
                if val > max:
                    max = val
            return max
        
        def minValue(state, ghostNumber, depth):
            if depth == self._treeDepth or not state.getLegalActions(ghostNumber):
                return self.getEvaluationFunction()(state)
            min = math.inf
            for action in state.getLegalActions(ghostNumber):
                totalGhosts = state.getNumAgents() - 1
                if ghostNumber == totalGhosts:
                    val = maxValue(
                        state.generateSuccessor(ghostNumber, action), depth + 1)
                else:
                    val = minValue(
                        state.generateSuccessor(ghostNumber, action), ghostNumber + 1, depth)
                if val < min:
                    min = val
            return min
            
        max = -math.inf
        maxAct = None
        for action in state.getLegalActions(0):
            if action == Directions.STOP:
                continue
            val = minValue(state.generateSuccessor(0, action), 1, 0)
            if val > max:
                max = val
                maxAct = action

        return maxAct

    def getTreeDepth(self):
        return super().getTreeDepth()
    
    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def getAction(self, state):
        def maxValue(state, depth, alpha, beta):
            if depth == self._treeDepth or not state.getLegalActions(0):
                return self.getEvaluationFunction()(state)
            v = -math.inf
            for action in state.getLegalActions(0):
                if action == Directions.STOP:
                    continue
                v = max(v, minValue(
                        state.generateSuccessor(0, action), 1, depth, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def minValue(state, ghostNumber, depth, alpha, beta):
            if depth == self._treeDepth or not state.getLegalActions(ghostNumber):
                return self.getEvaluationFunction()(state)
            v = math.inf
            for action in state.getLegalActions(ghostNumber):
                totalGhosts = state.getNumAgents() - 1
                if ghostNumber == totalGhosts:
                    v = min(v, maxValue(
                        state.generateSuccessor(
                            ghostNumber, action), depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(
                        state.generateSuccessor(
                            ghostNumber, action), ghostNumber + 1, depth, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v
            
        maxVal = -math.inf
        maxAct = None
        for action in state.getLegalActions(0):
            if action == Directions.STOP:
                continue
            v = maxValue(state.generateSuccessor(0, action), 0, -math.inf, math.inf)
            if v > maxVal:
                maxVal = v
                maxAct = action
        return maxAct

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
    def getTreeDepth(self):
        return super().getTreeDepth()
    
    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        def maxValue(state, depth):
            if depth == self._treeDepth or not state.getLegalActions(0):
                return self.getEvaluationFunction()(state)
            v = -math.inf
            for action in state.getLegalActions(0):
                if action == Directions.STOP:
                    continue
                v = max(v, expValue(
                        state.generateSuccessor(0, action), 1, depth))
            return v
        
        def expValue(state, ghostNumber, depth):
            legalActions = state.getLegalActions(ghostNumber)
            if depth == self._treeDepth or not legalActions:
                return self.getEvaluationFunction()(state)
            expVal = 0
            for action in legalActions:
                totalGhosts = state.getNumAgents() - 1
                if ghostNumber == totalGhosts:
                    v = maxValue(
                        state.generateSuccessor(
                            ghostNumber, action), depth + 1)
                else:
                    v = expValue(
                        state.generateSuccessor(
                            ghostNumber, action), ghostNumber + 1, depth)
                expVal += v / float(len(legalActions))
            return expVal
            
        maxVal = -math.inf
        maxAct = None
        for action in state.getLegalActions(0):
            if action == Directions.STOP:
                continue
            v = maxValue(state.generateSuccessor(0, action), 0)
            if v > maxVal:
                maxVal = v
                maxAct = action
        return maxAct

    def getTreeDepth(self):
        return super().getTreeDepth()
    
    def getEvaluationFunction(self):
        return super().getEvaluationFunction()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: I wanted pacman to avoid ghosts so I used the negative reciprocal of the nearest
    distance to a non-scared ghost. It becomes more negative when the ghost is nearer. I multiplied
    it by a large number because I did not want pacman to die.

    I used negative of the nearest distance to a scared ghost because they give a lot of points when
    pacman eats them.

    I used negative of the nearest distance to food because I wanted pacman to go nearer to the food
    to eat them.

    I used negative of the number of food left because I wanted pacman to eat food so fewer remains.

    I used a large negative of the number of capsules left because I wanted pacman to eat capsules
    and get a lot of points.

    I used the score of the state so that pacman does things that increases the score. When pacman
    goes near something that it can eat to increase the score, it will eat it.

    """

    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    nonScaredGhostPos = [
        state.getPosition() for state in ghostStates if not state.isScared()]
    distToNormalGhost = math.inf
    if nonScaredGhostPos:
        distToNormalGhost = min(manhattan(
            position, tuple(map(int, pos))) for pos in nonScaredGhostPos)
        if distToNormalGhost == 0:
            distToNormalGhost = 0.000001

    scaredGhostPos = [
        state.getPosition() for state in ghostStates if state.isScared()]
    distToScaredGhost = 0
    if scaredGhostPos:
        distToScaredGhost = min(manhattan(
            position, tuple(map(int, pos))) for pos in scaredGhostPos)

    foodList = currentGameState.getFood().asList()
    foodNum = currentGameState.getNumFood()
    distToFood = 0
    if foodNum:
        distToFood = min(manhattan(position, foodPos) for foodPos in foodList)

    capsuleNum = currentGameState.getNumCapsules()

    return (-100.0 / distToNormalGhost - 5 * distToScaredGhost
            - 1.5 * distToFood - 2 * foodNum - 20 * capsuleNum + currentGameState.getScore())

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
