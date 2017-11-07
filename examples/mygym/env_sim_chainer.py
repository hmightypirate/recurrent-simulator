import chainer
import warnings
import gym
from chainerrl import env
from chainerrl import spaces

import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.color import gray2rgb
import collections

import numpy as np
from collections import deque
from env_gym_chainer import GymEnvironment
import recurrent_env


try:
    import cv2

    def imresize(img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

except Exception:
    from PIL import Image

    warnings.warn(
        'Since cv2 is not available PIL will be used instead to resize images.'
        ' This might affect the resulting images.')

    def imresize(img, size):
        return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))


UNROLL_STEPS = 600

class SimEnvironment(GymEnvironment):
    """ Simulated environment

    preprocesss screens and holds it onto a screen
    buffer of size agent_history_length from
    which the environment state is constructed

    """

    def __init__(self, sample_env, res_width, res_height,
                 agent_history_length,
                 render=False,
                 # Extra parameters for simulated environment
                 optimizer=None,
                 unroll_dep=15,
                 obs_dep=5,
                 pred_dep=10, batch_size=16,
                 warm_up_steps=60000,
                 buffer_size=1e6,
                 num_output_channels=1024, training_steps=30, loadto=None,
                 savesim=None, chain_period=100,
                 save_steps=30000):
        """ Initialization stuff

        Parameters:
        -----------
        env: gym environment
        res_width: resized width
        res_height: resized height
        agent_history_length: buffer length


        """

        self.chain_period = chain_period

        super(SimEnvironment, self).__init__(
            env=sample_env,
            res_width=res_width,
            res_height=res_height,
            agent_history_length=agent_history_length,
            render=render)

        # Pick actions from the environment
        num_actions = self.env.action_space.n

        # Instance of the simulated environment
        self.sim_env = recurrent_env.Recurrent_Env(
            optimizer=optimizer, width=res_width, height=res_height,
            num_actions=num_actions,
            unroll_dep=unroll_dep,
            obs_dep=obs_dep,
            pred_dep=pred_dep,
            batch_size=batch_size,
            warm_up_steps=warm_up_steps,
            buffer_size=buffer_size,
            num_output_channels=num_output_channels,
            training_steps=training_steps, outdir=savesim,
            save_steps=save_steps)

        # load previous checkpoint (if provided)
        if loadto is not None:
            self.sim_env.load(loadto)

        self.viewer_sim = None

    def reset(self):
        state = super(SimEnvironment, self).reset()

        # Reset internal state of the simulator
        self.sim_env.reset_state()
        self.last_orig_screen = self.current_screen()
        self.last_sim_screen = self.last_orig_screen

        return state

    def step(self, action):

        # Pick prev screens
        my_screen = self.last_sim_screen
        my_orig_screen = self.last_orig_screen

        # Calculate next step in the real environemtn
        super(SimEnvironment, self).step(action)

        # store last screen from the emulator
        self.last_orig_screen = self.current_screen()

        if chainer.config.train:
            # Pick next screen from sim environment
            (next_screen, reward, is_terminal, info) = self.sim_env.step(
                my_orig_screen, action, 0, False)

        else:
            # Next step using orig frames
            if (self.sim_env.current_steps % self.chain_period == 0 or
                    self.sim_env.current_steps < UNROLL_STEPS):
                (next_screen, reward, is_terminal, info) = self.sim_env.step(
                    my_orig_screen, action, 0, False)

                print "WHAATT", self.sim_env.current_steps

            else:
                print "NEXT"
                # Next step using prev sim frames
                (next_screen, reward, is_terminal, info) = self.sim_env.step(
                    my_screen, action, 0, False, is_sim_frame=True)

            # Build the state with the returned frame
            # from the sim environment (Overwrite last screen)
            # agents will never see a real screen during tests, muahahaha
            # self.last_raw_screen = next_screen  # FIXME this is wrong
            # if not self._terminal:
            self.last_screens[-1] = next_screen
            self.last_sim_screen = next_screen

            if self.render:
                self._render_sim()

        return self.state, self._reward, self._terminal, {}

    def _render_sim(self, close=False):
        if close:
            if self.viewer_sim is not None:
                self.viewer_sim.close()
                self.viewer_sim = None
            return

        # Pick last screen
        img = self.last_screens[-1]

        from gym.envs.classic_control import rendering

        if self.viewer_sim is None:
            self.viewer_sim = rendering.SimpleImageViewer()
        self.viewer_sim.imshow(imresize(gray2rgb(img), (512, 512)))
