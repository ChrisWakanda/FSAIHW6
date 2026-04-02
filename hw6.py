import numpy as np
from enum import Enum

def solve_mdp():
    """
    Solve the 4x3 grid MDP using Value Iteration, TD Learning, and Q-Learning.
    Returns:
    utilities_vi: numpy array of shape (11,) - utilities from value iteration
    policy: numpy array of shape (11,) - optimal policy from value iteration
    utilities_td: numpy array of shape (11,) - utilities from TD learning
    q_values: numpy array of shape (11, 4) - Q-values from Q-learning
    """
    #-----------------DO NOT MODIFY BELOW-----------------------#
    # Define states and actions
    states = [(0,0), (1,0), (2,0), (3,0), (0,1), (2,1), (3,1),
    (0,2), (1,2), (2,2), (3,2)]
    actions = ["NORTH", "EAST", "WEST", "SOUTH"]
    # Terminal states
    terminal_states = {(3,1): -1, (3,2): +1}
    #-----------------DO NOT MODIFY ABOVE-----------------------#
    
    # Mapping string directons to (dx, dy) grid movements
    class Directions(Enum):
        NORTH=(0,1)
        SOUTH=(0, -1)
        WEST=(-1, 0)
        EAST=(1, 0)
    
    # handles movement and detects wall collisions
    def nextState(curr, direction):
        oldx, oldy = curr
        dirx, diry = Directions[direction].value
        nextStatex=oldx+dirx
        nextstatey=oldy+diry
        if nextStatex < 0 or nextStatex > 3 or nextstatey < 0 or nextstatey > 2 or (nextStatex == 1 and nextstatey == 1):
            return curr # return current state if invalid
        return (oldx+dirx, oldy+diry)

    utilities_vi=np.zeros(len(states))
    store={}
    for index, state in enumerate(states):
        store[state]=index
        
    # The dictionary/map that stores the probabilties of the action shift
    incline = {
        "NORTH": [("NORTH", 0.8), ("EAST", 0.1), ("WEST", 0.1)],
        "SOUTH": [("SOUTH", 0.8), ("EAST", 0.1), ("WEST", 0.1)],
        "EAST":  [("EAST", 0.8), ("NORTH", 0.1), ("SOUTH", 0.1)],
        "WEST":  [("WEST", 0.8), ("NORTH", 0.1), ("SOUTH", 0.1)]
    }

    # PART A - Value Iteration

    policy=np.zeros(len(states), dtype=int)

    for _ in range(100):
        # Must update based on the previous iteration's board state
        clonedUtils=np.copy(utilities_vi)
        for state in states:
            if state in terminal_states:
                clonedUtils[store[state]]=terminal_states[state]
                continue
            baction=None
            bval=float('-inf')
            
            #finds the action with the highest posible ultility value
            for action in actions:
                possibleVal=0
                for dir, prob in incline[action]:
                    nexts=nextState(state, dir)
                    possibleVal += prob*utilities_vi[store[nexts]]
                possibleVal*=0.9
                if possibleVal>bval:
                    bval=possibleVal
                    baction=action
            clonedUtils[store[state]]=bval
            policy[store[state]]=actions.index(baction)
        utilities_vi=clonedUtils

    # PART B - TD Learning

    learningRate = 0.02
    gamma = 0.9
    utilities_td=np.zeros(len(states))
    
    for _ in range(5000):
        # dropping the agent anywhere for maximium exploration
        curr = states[np.random.choice(len(states))]

        while curr not in terminal_states:
            nextAction=actions[policy[store[curr]]]
            possibleDirections=[]
            probabs=[]
            for d, prob in incline[nextAction]:
                possibleDirections.append(d)
                probabs.append(prob)
                
            # random choice to see where the actual direction is and use that to find next state.
            realDirection=np.random.choice(possibleDirections, p=probabs)
            nexts=nextState(curr, realDirection)

            # TD update rule
            previousUtility = utilities_td[store[curr]]
            nextUtility = utilities_td[store[nexts]]

            if nexts in terminal_states:
                reward = terminal_states[nexts]
            else:
                reward = 0.0

            td_error=(reward + (gamma * nextUtility)) - previousUtility
            utilities_td[store[curr]]=previousUtility + (learningRate * td_error)

            curr=nexts

    # Hardcoded the terminal states in the finl array so it prints correctly
    for t_state in terminal_states:
        utilities_td[store[t_state]] = terminal_states[t_state]

    # PART C - Q Learning

    q_values = np.zeros((len(states), 4))

    for _ in range(5000):
        curr = states[np.random.choice(len(states))]
        while curr not in terminal_states:
            # pick a random action to learn the environment
            action = np.random.choice(actions)
            possibleDirections = []
            probabs = []
            for d, prob in incline[action]:
                possibleDirections.append(d)
                probabs.append(prob)
            realDirection = np.random.choice(possibleDirections, p=probabs)
            nexts = nextState(curr, realDirection)

            currColumn = actions.index(action)
            priorQVal = q_values[store[curr], currColumn]

            # look ahead at the best possible future action
            maxNextQ = np.max(q_values[store[nexts]])
            
            if nexts in terminal_states:
                reward = terminal_states[nexts]
            else:
                reward = 0.0

            tdError = (reward + (gamma * maxNextQ)) - priorQVal
            q_values[store[curr], currColumn] = priorQVal + (learningRate * tdError)

            curr = nexts

    for t_state in terminal_states:
        for a in actions:
            q_values[store[t_state], actions.index(a)] = terminal_states[t_state]

    # rounding
    utilities_vi=np.round(utilities_vi, 3)
    utilities_td=np.round(utilities_td, 3)
    q_values=np.round(q_values, 3)

    return utilities_vi, policy, utilities_td, q_values