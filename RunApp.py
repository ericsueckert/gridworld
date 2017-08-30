'''

Eric Eckert
'''

import MDP, Grid

def GW_Values_string(V_dict):

    #Remove "DEAD" state so we can sort values
    temp = {}
    temp.update(V_dict)
    del temp["DEAD"]

    grid = []
    s = "["
    rowl = []

    for i in range(3):
        for j in range(4):
            rowl.append(s)

        grid.append(rowl)
        rowl = []

    for state in temp:
        row = 2-state[1]
        col = state[0]
        val = "%.3f" % temp[state]
        grid[row][col] += val

    s = ""
    for row in grid:
        for val in row:
            s += val + "]\t"
        s += "\n\n"
    #print(grid)
    return s

def GW_QValues_string(Q_dict):

    #Remove "DEAD" state so we can sort values
    temp = {}
    temp.update(Q_dict)
    # Remove all dead states
    deadstates = [sa for sa in temp.keys() if sa[0] == "DEAD"]
    for deadstate in deadstates:
        if deadstate in temp:
            del temp[deadstate]

    grid = []
    s = "["
    rowl = []

    for i in range(3):
        for j in range(4):
            rowl.append(s)

        grid.append(rowl)
        rowl = []

    for state in temp:
        #print(state)

        row = 2-state[0][1]
        col = state[0][0]
        #If the grid value has been updated, skip this key state,action
        if not grid[row][col] == "[":
            continue
        salist = []
        #Find all state,action combinations for this state
        salist = [sa for sa in temp.keys() if sa[0] == state[0]]
        qlist = [temp[sa][0] for sa in salist]
        #print(qlist)

        #Add the highest q value to the grid
        val = "%.3f" % max(qlist)
        grid[row][col] += val

    s = ""
    for row in grid:
        for val in row:
            s += val + "]\t"
        s += "\n\n"
    #print(grid)
    return s

def GW_Policy_string(optPolicy):
    temp = {}
    temp.update(optPolicy)
    del temp["DEAD"]

    grid = []
    s = "["
    rowl = []

    for i in range(3):
        for j in range(4):
            rowl.append(s)

        grid.append(rowl)
        rowl = []

    for state in temp:
        row = 2-state[1]
        col = state[0]

        val = temp[state]
        grid[row][col] += val

    s = ""
    for row in grid:
        for val in row:
            s += val + "]\t"
        s += "\n\n"
    #print(grid)
    return s

def test():
    '''Create the MDP, then run an episode of random actions for 10 steps.'''
    grid_MDP = MDP.MDP()
    grid_MDP.register_start_state((0,0))
    grid_MDP.register_actions(Grid.ACTIONS)
    grid_MDP.register_operators(Grid.OPERATORS)
    grid_MDP.register_transition_function(Grid.T)
    grid_MDP.register_reward_function(Grid.R)
    #grid_MDP.random_episode(100)

    # Uncomment the following, when you are ready...

    #grid_MDP.valueIteration( 0.9, 7)
    #print("GW values: ")
    #print(GW_Values_string(grid_MDP.V))

    grid_MDP.QLearning( 0.5, 50, 0.5)
    print(GW_QValues_string(grid_MDP.Q))
    grid_MDP.extractPolicy()
    print(GW_Policy_string(grid_MDP.optPolicy))

test()
