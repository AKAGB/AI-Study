import json
import random
class Bot(object):
    """
    The Bot class that applies the Qlearning logic to Flappy bird game
    After every iteration (iteration = 1 game that ends with the bird dying) updates Q values
    After every DUMPING_N iterations, dumps the Q values to the local JSON file
    """

    def __init__(self):
        self.gameCNT = 0  # Game count of current run, incremented after every death
        self.DUMPING_N = 25  # Number of iterations to dump Q values to JSON after
        self.discount = 0.9
        self.r = {0: 1, 1: -1500}  # Reward function
        self.lr = 0.3
        self.load_qvalues()
        self.last_state = "420_240_0"
        self.last_action = 0
        self.moves = []

    def load_qvalues(self):
        """
        Load q values from a JSON file
        """
        self.qvalues = {}
        try:
            fil = open("qvalues.json", "r")
        except IOError:
            return
        self.qvalues = json.load(fil)
        fil.close()

    def act(self, xdif, ydif, vel):
        """
        Chooses the best action with respect to the current state - Chooses 0 (don't flap) to tie-break
        """
        state = self.map_state(xdif, ydif, vel)

        self.moves.append(
            (self.last_state, self.last_action, state)
        )  # Add the experience to the history

        if len(self.moves) > 1001:
            del self.moves[0]

        self.last_state = state  # Update the last_state with the current state

        if self.qvalues[state][0] >= self.qvalues[state][1]:
            self.last_action = 0
            return 0
        else:
            self.last_action = 1
            return 1

    def update_scores(self, dump_qvalues = True):
        """
        Update qvalues via iterating over experiences
        """
        history = list(reversed(self.moves))

        # Flag if the bird died in the top pipe
        high_death_flag = True if int(history[0][2].split("_")[1]) >= 100  else False


        # if high_death_flag:
        #     print('High')
        # else:
        #     print('Low')
        # print(history[0][2].split("_"))
        # Q-learning score updates
        #You code here
        t = 1
        cnt = 1
        # miu = len(history) // 100 + 1
        for s, a, s_ in history:
            reward = self.r[0]
            reward /= cnt
            cnt += 1
            if cnt > 1000:
                break
            if t:
                if high_death_flag:
                    
                    if a == 1:
                        # reward = self.r[1] * abs(int(s.split('_')[2])) / 10 + 100 * miu
                        reward = self.r[1]
                        t -= 1
                else:
                    if a == 0:
                        # reward = self.r[1] * abs(int(s.split('_')[2])) / 10 + 100 * miu
                        reward = self.r[1]
                        t -= 1
            
            # if (high_death_flag and a == 1) or (not high_death_flag and a == 0):
            #     reward *= -1
            

            self.qvalues[s][a] += self.lr * (reward + self.discount * max(self.qvalues[s_]) - self.qvalues[s][a])
            

        self.gameCNT += 1  # increase game count
        if dump_qvalues:
            self.dump_qvalues()  # Dump q values (if game count % DUMPING_N == 0)
        self.moves = []  # clear history after updating strategies

    def map_state(self, xdif, ydif, vel):
        """
        Map the (xdif, ydif, vel) to the respective state, with regards to the grids
        The state is a string, "xdif_ydif_vel"

        X -> [-40,-30...120] U [140, 210 ... 420]
        Y -> [-300, -290 ... 160] U [180, 240 ... 420]
        """
        if xdif < 140:
            xdif = int(xdif) - (int(xdif) % 10)
        else:
            xdif = int(xdif) - (int(xdif) % 70)

        if ydif < 180:
            ydif = int(ydif) - (int(ydif) % 10)
        else:
            ydif = int(ydif) - (int(ydif) % 60)

        return str(int(xdif)) + "_" + str(int(ydif)) + "_" + str(vel)

    def dump_qvalues(self, force = False):
        """
        Dump the qvalues to the JSON file
        """
        if self.gameCNT % self.DUMPING_N == 0 or force:
            fil = open("qvalues.json", "w")
            json.dump(self.qvalues, fil)
            fil.close()
            print("Q-values updated on local file.")
