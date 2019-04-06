
def isValid(state):
    """check whether the state is valid"""
    return state != 'w'

def isTerm(state):
    return state == 'e'

def show(values):
    for i in range(3):
        for j in range(4):
            if (i, j) in values:
                print('%8.2f' % values[(i, j)], end='')
            else:
                print(' ' * 7 + 'w', end='')
        print()

def showPolicy(policy):
    l = ['^', '>', 'v', '<']
    for i in range(3):
        for j in range(4):
            if (i, j) in policy:
                print(l[policy[(i, j)]], end=' ')
            else:
                print('w ', end='')
        print()

class MDP:
    """Implement basic model"""
    def __init__(self, maze, reward):
        self.states = set()
        self.terminal = set()
        self.reward = reward
        r, c = len(maze), len(maze[0])

        for i in range(r):
            for j in range(c):
                if isValid(maze[i][j]):
                    self.states.add((i, j))
                if isTerm(maze[i][j]):
                    self.terminal.add((i, j))

    def getStates(self):
        return self.states
        
    def getPosibleActions(self, state):
        if state not in self.states:
            return []
        actions = [
            (state[0] - 1, state[1]),
            (state[0], state[1] + 1),
            (state[0] + 1, state[1]),
            (state[0], state[1] - 1)
        ]
        return actions


    def getTransitions(self, state, action):
        res = []
        delta1 = action[1] - state[1]
        delta2 = action[0] - state[0]
        west = (state[0] - delta1, state[1] - delta2)
        east = (state[0] + delta1, state[1] + delta2)
        if action in self.states:
            res.append((action, 0.8))
        else:
            res.append((state, 0.8))
        if west in self.states:
            res.append((west, 0.1))
        else:
            res.append((state, 0.1))
        if east in self.states:
            res.append((east, 0.1))
        else:
            res.append((state, 0.1))
        return res


    def getReward(self, state, action, nextState):
        return self.reward[state[0]][state[1]]

    def isTerminal(self, state):
        return state in self.terminal

class IterationAgent:
    """
        Common interface of VI and PI
    """
    def __init__(self, mdp, discount=0.9, iterations=5):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}

    def computeQvalue(self, state, action):
        """
            Compute Qvalue from Values 
            according to Bellman Equation.
        """
        transitions = self.mdp.getTransitions(state, action)
        res = 0
        for nextState, prop in transitions:
            res += prop * (self.mdp.getReward(state, action, nextState) +
                            self.discount * self.values[nextState])
        return res

    def getAction(self, state):
        """
            Get best action according to the values.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPosibleActions(state)

        maxA = None
        maxQValue = - float('inf')
        for a in actions:
            tmp = self.computeQvalue(state, a)
            if tmp > maxQValue:
                maxQValue = tmp
                maxA = a

        return maxA

    def getPolicy(self):
        raise NotImplementedError

class ValueIterationAgent(IterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=5):
        super(ValueIterationAgent, self).__init__(mdp, discount, iterations)
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}
        self.qvalues = {}
        states = self.mdp.getStates()
        for s in states:
            self.values[s] = 0      # initial values

        for i in range(self.iterations):
            tmpValues = {}
            for state in states:
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPosibleActions(state)
                    qvalues = [self.computeQvalue(state, a) for a in actions]
                    tmpValues[state] = max(qvalues)
                    self.qvalues[state] = qvalues
                else:
                    tmpValues[state] = self.mdp.getReward(state, state, state)
            self.values = tmpValues

    def getPolicy(self):
        # 0 - up / 1 - right / 2 - down / 3 - left
        policy = {}
        for s in self.qvalues.keys():
            policy[s] = self.qvalues[s].index(self.values[s])
        return policy


class PolicyIterationAgent(IterationAgent):
    def __init__(self, mdp, discount=0.9, iterations=5):
        super(PolicyIterationAgent, self).__init__(mdp, discount, iterations)
        self.pi = {}
        self.qvalues = {}
        self.lastQvalues = {}
        states = self.mdp.getStates()
        for s in states:
            self.qvalues[s] = [0, 0, 0, 0]
            self.values[s] = 0
            self.pi[s] = 0
        self.lastQvalues = self.qvalues
        for i in range(self.iterations):
            tmpValues = {}
            for state in self.values.keys():                # update qvalue
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPosibleActions(state)
                    tmpValues[state] = self.qvalues[state][self.pi[state]]
                else:
                    tmpValues[state] = self.mdp.getReward(state, state, state)
            self.values = tmpValues
            self.lastQvalues = self.qvalues             # remmember qvalues
            self.qvalues = {}
            for state in self.pi.keys():                # update Pi
                actions = self.mdp.getPosibleActions(state)
                qvalues = [self.computeQvalue(state, a) for a in actions]
                self.qvalues[state] = qvalues
                self.pi[state] = qvalues.index(max(qvalues))
        self.qvalues = self.lastQvalues

    def getPolicy(self):
        return self.pi
        
if __name__ == '__main__':
    maze = [
        '000e',
        '0w0e',
        '0000'
    ]
    reward = [
        [-0.03, -0.03, -0.03, 1],
        [-0.03, -0.03, -0.03, -1],
        [-0.03, -0.03, -0.03, -0.03]
    ]
    mdp = MDP(maze, reward)
    # agent = ValueIterationAgent(mdp, 0.9, 5)
    
    agent = PolicyIterationAgent(mdp, 0.9, 100)

    show(agent.values)
    print('Policy:')
    showPolicy(agent.getPolicy())