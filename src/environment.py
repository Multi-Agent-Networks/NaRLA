import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self):
        self.fig     = None
        self.states  = []
        self.rewards = []

        # FOLLOWING GYM CONVENTION
        class Obs: pass
        self.observation_space = Obs()
        self.observation_space.shape = (2,)

        self.action_space = Obs()
        self.action_space.n = 2

    def reset(self):
        self.count   = 0
        self.states  = []
        self.rewards = []

        # clear the plot
        if self.fig is not None: self.ax.clear()

        return self.sample()

    def sample(self):
        self.x = np.random.uniform(size=(2))
        self.states.append(self.x)
        return self.x

    def step(self, action):
        x = self.x
        R = 1

        #   (x2 - x1) * (datay  - y1) - (y2 - y1) * (datax  - x1)
        C = (1. - 0.) * (x[1] - 0.) - (1. - 0.) * (x[0] - 0.)
        C = C > 0

        if action != C:
            R = -1; self.count += 1

        # stops after 25 episodes
        done = self.count == 25

        self.rewards.append(R)

        return self.sample(), R, done, None

    def render(self):
        X = np.array(self.states)
        C = ((1. - 0.) * (X[:,1] - 0.) - (1. - 0.) * (X[:,0] - 0.)) > 0

        # figure setup
        if self.fig is None:
            self.fig   = plt.figure()
            self.ax    = self.fig.add_subplot(111)

        self.ax.scatter(X[1:,0],X[1:,1],c=self.rewards)

        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
