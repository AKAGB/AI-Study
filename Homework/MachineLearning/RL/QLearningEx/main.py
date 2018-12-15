import numpy as np

class RL:
    def __init__(self, R, gamma=0.8, alpha=0.5, k=5):
        """
        Initialize the parameters
        """
        self.gamma = gamma 
        self.R = R
        self.Q = np.zeros(R.shape)
        self.k = k
        self.alpha = alpha

    def restart(self):
        """
        Reset Q matrix
        """
        self.Q = np.zeros(self.R.shape)

    def getActions(self, state):
        """
        Return the actions which the current state can perform.
        """
        return np.where(self.R[state] != -1)[0]

    def compute(self, state, next_state):
        """
        Compute the formula to update Q[state, action]
        """
        old = self.Q[state, next_state]
        predict = self.R[state, next_state] + self.gamma * np.max(self.Q[next_state]) 
        self.Q[state, next_state] = old + self.alpha * (predict - old)

    def loop(self, limit=1000):
        states = range(self.R.shape[0])
        for i in range(limit):
            state = np.random.choice(states)
            # This code can be editted.
            for j in range(self.k):
                actions = self.getActions(state)
                next_state = np.random.choice(actions)
                self.compute(state, next_state)
                state = next_state

    def getPolicy(self, state, goal, limit=1000):
        """
        Get Policy by Q matrix.
        """
        policy = [state]
        cnt = 0
        while cnt < limit and state != goal:
            state = np.argmax(self.Q[state])
            policy.append(state)
            cnt += 1
        if cnt >= limit:
            print('Iteration reaches the upper limit.')
        return policy

if __name__ == '__main__':
    gamma = 0.8
    R = np.array([  [-1, -1, -1, -1, 0, -1], 
                    [-1, -1, -1, 0, -1, 100],
                    [-1, -1, -1, 0, -1, -1],
                    [-1, 0, 0, -1, 0, -1],
                    [0, -1, -1, 0, -1, 100],
                    [-1, 0, -1, -1, 0, 100]])
    machine = RL(R, gamma)
    machine.loop(100)
    policy = machine.getPolicy(2, 5)
    print('Q matrix:')
    print((machine.Q * 100 / np.max(machine.Q)).astype(np.int32))
    print('Policy:\n', policy)