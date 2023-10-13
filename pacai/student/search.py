"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # *** Your Code Here ***
    stack = Stack()
    stack.push(problem.startingState())
    visited = set()
    parent = {}

    while not stack.isEmpty():
        currState = stack.pop()
        if problem.isGoal(currState):
            res = []
            while True:
                # is start state
                if (currState not in parent):
                    break
                action = parent[currState][1]
                currState = parent[currState][0]
                res.insert(0, action)
            return res
        if currState not in visited:
            visited.add(currState)
            for (successorState, action, cost) in problem.successorStates(currState):
                if (successorState in visited):
                    continue
                parent[successorState] = (currState, action)
                stack.push(successorState)
    return []

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    queue = Queue()
    queue.push(problem.startingState())
    visited = set()
    parent = {}

    while not queue.isEmpty():
        currState = queue.pop()
        if (problem.isGoal(currState)):
            res = []
            while True:
                # is start state
                if (currState not in parent):
                    break
                action = parent[currState][1]
                currState = parent[currState][0]
                res.insert(0, action)
            return res
        if currState not in visited:
            visited.add(currState)
            for (successorState, action, cost) in problem.successorStates(currState):
                if (successorState in visited):
                    continue
                parent[successorState] = (currState, action)
                queue.push(successorState)
    return []

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()
