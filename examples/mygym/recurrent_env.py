import logging
import sys
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainerrl.initializers import LeCunNormal
from recurrent import RecurrentChainMixin
import iclr_acer_link
import agent_data_set
from recurrent import Recurrent
from recurrent import state_kept
from chainerrl.misc.batch_states import batch_states
from chainerrl.agent import AttributeSavingMixin
from cond_lstm import LSTMAction as LSTM


class RNNEnv(chainer.ChainList, RecurrentChainMixin):
    """ RNN model for building a recurrent environment

    """

    def __init__(self, n_actions, n_input_channels,
                 n_output_channels, input_size):

        """ Initialization stuff
        
        Parameters:
        n_actions: number of actions
        n_input_channels: number of input channels (always 1)
        n_output_channels: number of output channels (lstm dimimensions)
        input_size: a tuple with the image size that enters the network

        """
        
        self.head = iclr_acer_link.ICLRSimHeadv3(
            n_input_channels=1,
            n_output_channels=n_output_channels,
            input_size=input_size)

        self.deconv = iclr_acer_link.ICLRSimDeconvv3(
            n_input_channels=n_output_channels,
            n_output_channels=n_input_channels,
            input_size=input_size)

        self.lstm = LSTM(self.head.n_output_channels,
                         n_actions,
                         self.head.n_output_channels)
        
        super(RNNEnv, self).__init__(self.head, self.lstm, self.deconv)

    def next_pred(self, state, action):
        """ Next state prediction
        Parameters:
        state: current state
        action: a one hot vector with the current action

        """

        # Obtain the state codification
        # state is an image
        h = self.head(state)

        # Combination of action and state
        # This is the baseline of the original work
        # predict the next internal state (ht+1)
        h = self.lstm(h, action)

        # Next prediction image after deconv
        out = self.deconv(h)
        return out


def phi(obs, mean_image=None):
    """ Function to transform the input images

    Parameters:
    -----------
    obs: list of images to be transformed
    mean_image: current mean pixel value

    """

    raw_values = np.asarray(obs, dtype=np.float32)
    if mean_image is not None:
        raw_values -= mean_image
    raw_values /= 255.0
    return raw_values


def unphi(obs, mean_image=None):
    """ Function to reverse the phi transformation

    Parameters.
    obs: list of images to be transformed
    mean_image: current mean pixel value

    """

    raw_values = np.asarray(obs, dtype=np.float32)
    raw_values *= 255.0
    if mean_image is not None:
        raw_values += mean_image

    return raw_values


class Recurrent_Env(AttributeSavingMixin):
    """ Functionality to tran a recurrent environment

    Training is performed by measuring the MSE error obs_dep frames and
    pred_dep frames

    Parameters:
    optimizer: network optimizer (e.g. rmsprop, adam, etc)
    width: width of image (after resize)
    height: height of image (after resize)
    num_actions: number of possible actions in the environment
    unroll_dep: frames used to set up the internal state
    obs_dep: observation dependent frames (during training)
    pred_dep: prediction dependent frames (during training)
    warm_up_steps: wait these steps before training the system
    buffer_size: internal buffer to store frames and actions
                 obtained during gameplay
    num_output_channels: dimensions of the internal state
    training_steps: steps between training
    save_steps: steps between checkpoints
                (a checkpoint is only saved if the error is less than the
                 best model so far)
    outdir: output folder to store the checkpoints

    """

    def __init__(self, optimizer, width, height, num_actions,
                 unroll_dep=15,
                 obs_dep=10,
                 pred_dep=30, batch_size=16,
                 warm_up_steps=6000,
                 buffer_size=1e6,
                 num_output_channels=256, training_steps=30,
                 save_steps=3000,
                 outdir="mew",
                 max_mean_steps=3000):

        self.num_actions = num_actions
        self.width = width
        self.height = height
        self.buffer = None
        self.current_steps = 0
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.unroll_dep = unroll_dep
        self.obs_dep = obs_dep
        self.pred_dep = pred_dep
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.outdir = outdir
        self.save_steps = save_steps
        self.max_mean_steps = max_mean_steps

        # stat vars
        self.best_cum_error = sys.float_info.max
        self.cum_error = 0
        self.cum_samples = 0

        rng = np.random.RandomState()

        if chainer.config.train:
            # Buffer to store samples (during training)
            self.buffer = agent_data_set.DataSet(
                width, height,
                rng, max_steps=int(buffer_size),
                phi_length=1)

        self.rnn_env = RNNEnv(n_actions=num_actions,
                              n_input_channels=1,
                              n_output_channels=num_output_channels,
                              input_size=(width, height))

        self.optimizer.setup(self.rnn_env)

        self.rep_image = None
        self.num_reps = 0

    def train(self):
        # print "GOING TO TRAIN"

        # pick samples
        state_batch = []
        action_batch = []

        for i in range(self.batch_size):
            (states, actions, _,
             _) = self.buffer.pick_consecutive_batch(
                 self.unroll_dep + self.obs_dep + self.pred_dep + 1)

            state_batch.append(states)
            action_batch.append(actions)

        states = phi(state_batch, mean_image=self.rep_image/self.num_reps)

        # print ("STATE SHAPE ", np.shape(states))
        # print ("ACTION SHAPE ", np.shape(action_batch))

        # states shifted by 1
        target_states = states[:, 1:]

        # build the actions matrix (one hot encoding)
        a_t = np.zeros((len(action_batch),
                        action_batch[0].shape[0], self.num_actions),
                       dtype="float32")
        for i in range(len(action_batch)):
            for j in range(len(action_batch[i])):
                a_t[i, j, action_batch[i][j]] = 1

        obs_loss = 0
        pred_loss = 0

        # print "MEAN TARGET STATES ", np.mean(target_states), " ",
        # np.max(target_states), " ", np.min(target_states)

        with state_kept(self.rnn_env):
            self.reset_state()
            # only to init the internal state
            with chainer.no_backprop_mode():
                for i in range(self.unroll_dep):
                    next_states = self.rnn_env.next_pred(
                        np.asarray(states[:, i], dtype="float32"),
                        np.asarray(a_t[:, i], dtype="float32"))

            for i in range(self.obs_dep):
                next_states = self.rnn_env.next_pred(
                    np.asarray(states[:, i+self.unroll_dep], dtype="float32"),
                    np.asarray(a_t[:, i+self.unroll_dep], dtype="float32"))

                preds = F.flatten(next_states)
                target = F.flatten(target_states[:, i+self.unroll_dep])

                noise = F.gaussian(np.zeros(target.shape[0], dtype="float32"),
                                   np.zeros(target.shape[0], dtype="float32"))

                # the original paper adds the gaussian noise
                obs_loss += F.sum(F.squared_difference(target,
                                                       preds))  # + noise))
                                
            for i in range(self.pred_dep):
                next_states = self.rnn_env.next_pred(
                    next_states,
                    np.asarray(a_t[:, self.obs_dep + self.unroll_dep + i],
                               dtype="float32"))

                preds = F.flatten(next_states)
                target = F.flatten(target_states[:, self.obs_dep +
                                                 self.unroll_dep + i])

                noise = F.gaussian(np.zeros(target.shape[0], dtype="float32"),
                                   np.zeros(target.shape[0], dtype="float32"))

                pred_loss += F.sum(F.squared_difference(target,
                                                        preds))  # + noise))

            # Compute gradients using thread-specific model
            total_loss = obs_loss + pred_loss

            self.cum_error += total_loss.data
            self.cum_samples += target_states.shape[0] * target_states.shape[1]

            self.rnn_env.cleargrads()
            total_loss.backward()
            if isinstance(self.rnn_env, Recurrent):
                self.rnn_env.unchain_backward()
            self.optimizer.update()
            # print "TOTAL_LOSS ", total_loss.data
        pass

    def step(self, state, action, reward, terminal, is_sim_frame=False):

        if not is_sim_frame and self.num_reps < self.max_mean_steps:
            if self.rep_image is None:
                self.rep_image = np.array(state, dtype="float32")
            else:
                self.rep_image += np.array(state, dtype="float32")

            self.num_reps += 1

        if chainer.config.train:
            # add sample to replay buffer
            self.buffer.add_sample(state, action, reward, terminal)

            # train the system
            if (self.current_steps > self.warm_up_steps):
                if self.current_steps % self.training_steps == 0:
                    self.train()

                if self.current_steps % 100 == 0:
                    logging.info("Eval {}".format(self.cum_error/self.cum_samples))

                # check if we reached a saving step
                if self.current_steps % self.save_steps == 0:
                    my_error = self.cum_error/self.cum_samples
                    logging.info("Current error {} Best_error {}".format(
                        my_error,
                        self.best_cum_error))
                    if my_error < self.best_cum_error:
                        if self.outdir is not None:
                            self.save(self.outdir)
                            pass
                        self.best_cum_error = my_error
                    self.cum_error = 0
                    self.cum_samples = 0

        # Obtain next state prediction for the agent
        with chainer.no_backprop_mode():
            statevar = phi([[state]],
                           mean_image=self.rep_image/self.num_reps)
            # one-hot encoding of action
            a_t = np.zeros((1, self.num_actions), dtype="float32")
            a_t[0][action] = 1

            next_pred = self.rnn_env.next_pred(statevar, a_t)
            next_state = next_pred.data[0]
            next_state = next_state.reshape([self.width,
                                             self.height])

        self.current_steps += 1

        # obtain the image
        next_state = unphi(next_state,
                           mean_image=self.rep_image/self.num_reps)

        # print "PRENEXT ", np.max(next_state)," ",  np.min(next_state), " ",
        # np.mean(next_state)

        next_state = next_state.clip(0, 255.0)
        next_state = np.array(next_state, dtype="uint8")

        return next_state, reward, False, {}

    def load(self, dirname):
        super(Recurrent_Env, self).load(dirname)

    def reset_state(self):
        """ Reset interal LSTM state """
        self.rnn_env.reset_state()

    @property
    def saved_attributes(self):
        return ('rnn_env', 'optimizer')
