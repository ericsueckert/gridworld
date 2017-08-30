'''MDP.py
S. Tanimoto, May 2016, 2017.

Provides representations for Markov Decision Processes, plus
functionality for running the transitions.

The transition function should be a function of three arguments:
T(s, a, sp), where s and sp are states and a is an action.
The reward function should also be a function of the three same
arguments.  However, its return value is not a probability but
a numeric reward value -- any real number.

operators:  state-space search objects consisting of a precondition
 and deterministic state-transformation function.
 We assume these are in the "QUIET" format used in earlier assignments.

actions:  objects (for us just Python strings) that are 
 stochastically mapped into operators at runtime according 
 to the Transition function.

Eric Eckert 1338722
CSE 415 SP17
Tanimoto


'''
import random
import itertools
from operator import itemgetter

REPORTING = True

class MDP:
    def __init__(self):
        self.known_states = set()
        self.succ = {} # hash of adjacency lists by state.

    def register_start_state(self, start_state):
        self.start_state = start_state
        self.known_states.add(start_state)

    def register_actions(self, action_list):
        self.actions = action_list

    def register_operators(self, op_list):
        self.ops = op_list

    def register_transition_function(self, transition_function):
        self.T = transition_function

    def register_reward_function(self, reward_function):
        self.R = reward_function

    def state_neighbors(self, state):
        '''Return a list of the successors of state.  First check
           in the hash self.succ for these.  If there is no list for
           this state, then construct and save it.
           And then return the neighbors.'''
        neighbors = self.succ.get(state, False)
        if neighbors==False:
            neighbors = [op.apply(state) for op in self.ops if op.is_applicable(state)]
            self.succ[state]=neighbors
            self.known_states.update(neighbors)
        return neighbors

    def random_episode(self, nsteps):
        self.current_state = self.start_state
        self.known_states = set()
        self.known_states.add(self.current_state)
        self.current_reward = 0.0
        for i in range(nsteps):
            self.take_action(random.choice(self.actions))
            if self.current_state == 'DEAD':
                print('Terminating at DEAD state.')
                break
        if REPORTING: print("Done with "+str(i)+" of random exploration.")

    def take_action(self, a):
        s = self.current_state
        neighbors = self.state_neighbors(s)
        threshold = 0.0
        rnd = random.uniform(0.0, 1.0)
        r = self.R(s,a,s)
        for sp in neighbors:
            threshold += self.T(s, a, sp)
            if threshold>rnd:
                r = self.R(s, a, sp)
                s = sp
                break
        self.current_state = s
        self.known_states.add(self.current_state)
        if REPORTING: print("After action "+a+", moving to state "+str(self.current_state)+\
                            "; reward is "+str(r))

    def generateAllStates(self):
        #print("Exploring all states...")
        UNEXPLORED = []
        UNEXPLORED += self.known_states
        #print (UNEXPLORED)
        while UNEXPLORED:
            S = UNEXPLORED[0]
            del UNEXPLORED[0]
            self.known_states.add(S)
            current_known = set()
            current_known.update(self.known_states)

            #Check all neighbors
            neighbors = self.state_neighbors(S)
            for neighbor in neighbors:

                if not (neighbor in UNEXPLORED) and not (neighbor in current_known):
                    UNEXPLORED.append(neighbor)
            self.known_states.update(current_known)


    def valueIteration(self, discount, iterations):
        self.generateAllStates()
        self.V = {s:0 for s in self.known_states}
        new_V = {}

        #Perform specified number of iterations
        for i in range(iterations):
            new_V = {}
            #print("===========================================")
            #print("Iteration ", i)

            #Calculate new V value for every state
            for s in self.V:
                #print("-------------------------------------------")
                #print("Calculating state ", str(s))
                #For this state look at all possible actions
                max_q = -10000000000
                #Find all neighbors
                neighbors = []
                neighbors += self.state_neighbors(s);
                #print(neighbors)
                #Add self to neighbors (in case it runs into wall)
                neighbors.append(s)

                for a in self.actions:
                    q = 0
                    #print("Action: ", a)
                    #Look at each neighbor and calculate the total Q value for Q(s,a)
                    for sp in neighbors:
                        probability = self.T(s, a, sp)
                        #print("Probability of moving to ", str(sp), ": ", probability)
                        if probability == 0:
                            continue
                        reward = self.R(s, a, sp)
                        #print("V value of ", str(sp), ": ", self.V[sp])
                        sp_val = probability * (reward + (discount * self.V[sp]))
                        q += sp_val
                        #print("Value at dest: ", self.V[sp])
                        #print("Value for moving to ", str(sp), ": ", sp_val)

                    #print("Q value: ", q)
                    #Check if q value for this action is the max
                    max_q = max(q, max_q)

                #set new V value of state
                new_V[s] = max_q
                #print("new V value: ", max_q)
            #Update all V values
            self.V = {}
            self.V = new_V


    def QLearning(self, discount, nEpisodes, epsilon):
        self.generateAllStates()
        Qkeys = itertools.product(self.known_states, self.actions)
        #Q(s,a) is mapped to (Qval, count)
        #Start at 0 for all state action pairs
        self.Q = {k:(0, 0) for k in Qkeys}

        for i in range(nEpisodes):
            print("=========================")
            print("Episode ", i)
            self.current_state = self.start_state
            #Perform an episode by repeatedly making moves and calculating Q values
            #Until the player dies
            while not self.current_state == "DEAD":
                s = self.current_state
                actionvals = []
                #Find all Q(s,a) values for the current state
                for act in self.actions:
                    Qval = self.Q[(s,act)][0]
                    actionvals.append((Qval,(s,act)))

                rnd = random.uniform(0.0, 1.0)
                #find the highest q value in sa list
                #Most of the time find the optimal action and take it (1-epsilon)
                if rnd > epsilon:
                    #print("taking optimal action")
                    optimalq = max(actionvals, key=itemgetter(0))[0]
                    #print(optimalq)
                    optimalactionvals = []
                    #Find all sa combinations with this q value
                    for sa in actionvals:
                        if sa[0] == optimalq:
                            optimalactionvals.append(sa)
                    #print(optimalactionvals)
                    actionvals = optimalactionvals
                #else: print("taking random action")
                #print(actionvals)

                # randomly select from optimal sa in list
                #If optimal choice was taken, this will be
                #optimal choice
                a = random.choice(actionvals)[1][1]

                #Grab current values
                q = self.Q[(s,a)]
                q = (q[0], q[1] + 1)
                count = q[1]
                #Take action to find sp
                self.take_action(a)

                sp = self.current_state

                reward = self.R(s, a, sp)
                #find max Q value in sp
                Qsapvals = []
                for act in self.actions:
                    Qvalp = self.Q[(sp, act)][0]
                    Qsapvals.append(Qvalp)

                #Calculate Q value for current state/action
                qp = (1-1/count) * q[0] + (1/count) * (reward + (discount * max(Qsapvals)))
                #print("New Q value: ", qp)
                #print("New count: ", count)

                #Set new Q value for state action pair
                self.Q[(s,a)] = (qp, count)


    def extractPolicy(self):
        optimalq = {}
        for s in self.known_states:
            actionvals = []
            #Find all state action pairs for current state
            for act in self.actions:
                Qval = self.Q[(s,act)][0]
                actionvals.append((Qval,(s,act)))
            #Find optimal sa pair and map it
            optimalsa = max(actionvals, key=itemgetter(0))
            #print(optimalsa)
            optimalq[s] = optimalsa[1][1]
            #print(optimalq[s])

        self.optPolicy = optimalq
