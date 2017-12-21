import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainerrl.initializers import LeCunNormal


class ICLRACERHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1, input_size=(84, 84)):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        fl_input = (np.array(input_size, dtype="int32") - 8)/4 + 1
        sl_input = (fl_input - 4)/2 + 1
        tl_input = (sl_input - 3)/1 + 1

        linear_input = tl_input[0] * tl_input[1] * 64

#        w = chainer.initializers.HeNormal()

        # Layers
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 8) / 4 + 1, 20 x 20 x 32
        # 2nd layer: (20 - 4)/2 + 1, 9 x 9 x 64
        # 3rd layer: (9 - 3)/1 + 1, 7 x 7 x 64 = 3136

        layers = [
            L.Convolution2D(
                n_input_channels,
                            out_channels=32,
                            ksize=8,
                            stride=4,
                #                            initialW=w,
                            initial_bias=bias),
            L.Convolution2D(32,
                            out_channels=64,
                            ksize=4,
                            stride=2,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Convolution2D(64,
                            out_channels=64,
                            ksize=3,
                            stride=1,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Linear(linear_input, n_output_channels,   # FIXME, check INPUT
                     #                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRACERHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRACERDeconv(chainer.ChainList):
    """ Deconvolution of ICLRAcerHead """

    def __init__(self, n_input_channels, n_output_channels,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.activation = activation

        layers = [
            L.Linear(n_input_channels, 3136,
                     initial_bias=bias),
            L.Deconvolution2D(64,
                              out_channels=64,
                              ksize=3,
                              stride=1,
                              initial_bias=bias),
            L.Deconvolution2D(64,
                              out_channels=32,
                              ksize=4,
                              stride=2,
                              initial_bias=bias),
            L.Deconvolution2D(32,
                              out_channels=n_output_channels,
                              ksize=8,
                              stride=4,
                              initial_bias=bias)
            ]
        super(ICLRACERDeconv, self).__init__(*layers)

    def __call__(self, state):
        h = state

        for i in range(len(self)):
            if i == 1:
                # FIXME. hardcoded
                h = F.reshape(h, (h.data.shape[0], 64, 7, 7))

            h = self.activation(self[i](h))

        return h


class ICLRACERHeadMini(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

#        w = chainer.initializers.HeNormal()

        # Layers
        # Input should be 60 x 60 (n channels)
        # 1st layer: (60 - 8) / 4 + 1, 14 x 14 x 32
        # 2nd layer: (14 - 4)/2 + 1, 6 x 6 x 64
        # 3rd layer: (6 - 3)/1 + 1, 4 x 4 x 64 = 256

        layers = [
            L.Convolution2D(n_input_channels,
                            out_channels=32,
                            ksize=8,
                            stride=4,
                            #                            initialW=w,
                            initial_bias=bias),
            L.Convolution2D(32,
                            out_channels=64,
                            ksize=4,
                            stride=2,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Convolution2D(64,
                            out_channels=64,
                            ksize=3,
                            stride=1,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Linear(1024, n_output_channels,   # FIXME, check INPUT
                     #                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRACERHeadMini, self).__init__(*layers)

    def __call__(self, state):
        h = state

        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRSimHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

#        w = chainer.initializers.HeNormal()

        # Layers
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 8) / 4 + 1, 20 x 20 x 64
        # 2nd layer: (20 - 4)/2 + 1, 9 x 9 x 128
        # 3rd layer: (9 - 3)/1 + 1, 7 x 7 x 128 = 6272

        layers = [
            L.Convolution2D(
                n_input_channels,
                out_channels=64,
                ksize=8,
                stride=4,
                #                            initialW=w,
                initial_bias=bias),
            L.Convolution2D(64,
                            out_channels=128,
                            ksize=4,
                            stride=2,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Convolution2D(128,
                            out_channels=128,
                            ksize=3,
                            stride=1,
                            #                            initialW=w,
                            initial_bias=bias),

            L.Linear(6272, n_output_channels,   # FIXME, check INPUT
                     #                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRSimHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRSimDeconv(chainer.ChainList):
    """ Deconvolution of ICLRAcerHead """

    def __init__(self, n_input_channels, n_output_channels,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.activation = activation

        layers = [
            L.Linear(n_input_channels, 6272,
                     initial_bias=bias),
            L.Deconvolution2D(128,
                              out_channels=128,
                              ksize=3,
                              stride=1,
                              initial_bias=bias),
            L.Deconvolution2D(128,
                              out_channels=64,
                              ksize=4,
                              stride=2,
                              initial_bias=bias),
            L.Deconvolution2D(64,
                              out_channels=n_output_channels,
                              ksize=8,
                              stride=4,
                              initial_bias=bias)
            ]
        super(ICLRSimDeconv, self).__init__(*layers)

    def __call__(self, state):
        h = state

        for i in range(len(self)):
            if i == 1:
                # FIXME. hardcoded
                h = F.reshape(h, (h.data.shape[0], 128, 8, 8))
                h = self.activation(self[i](h))
            elif i == len(self) - 1:
                # last layer => no activation
                h = self[i](h)
            else:
                h = self.activation(self[i](h))

        return h


class ICLRSimHeadv2(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1, input_size=(84, 84)):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        w = chainer.initializers.HeNormal()

        # Layers (84x84)
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 8) / 2 + 1, 39 x 39 x 64
        # 2nd layer: (39 - 7)/2 + 1, 17 x 17 x 128
        # 3rd layer: (17 - 3)/2 + 1, 8 x 8 x 128 = 8192

        # Layers 92x92
        # 1st layer: (92 - 8)/2 + 1, 43 x 43 x 64
        # 2nd layer: (43-7)/2 + 1, 19 x 19 x 128
        # 3rd layer: (19 - 3)/2 + 1, 9 x 9 x 128 = 10368

        # Layers (100x100)
        # 1st layer: (100 - 8)/2 + 1, 47 x 47 x 64
        # 2nd layer: (47 - 7)/2 + 1, 21 x 21 x 128
        # 3rd layer: (21 - 3)/2 + 1, 10 x 10 x 128 = 12800

        fl_input = (np.array(input_size, dtype="int32") - 8)/2 + 1
        sl_input = (fl_input - 7)/2 + 1
        tl_input = (sl_input - 3)/2 + 1

        linear_input = tl_input[0] * tl_input[1] * 64

        print("FL INPUT ", fl_input)
        print("SL INPUT ", sl_input)
        print("TL INPUT ", tl_input)
        print("LINEAR INPUT ", linear_input)

        layers = [
            L.Convolution2D(
                n_input_channels,
                out_channels=32,  # 64 initially
                ksize=8,
                stride=2,
                initialW=w,
                initial_bias=bias),
            L.Convolution2D(32,  # 64 initially
                            out_channels=32,
                            ksize=7,
                            stride=2,
                            initialW=w,
                            initial_bias=bias),

            L.Convolution2D(32,
                            out_channels=64,
                            ksize=3,
                            stride=2,
                            initialW=w,
                            initial_bias=bias),

            L.Linear(linear_input, n_output_channels,   # FIXME, check INPUT
                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]

        super(ICLRSimHeadv2, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRSimDeconvv2(chainer.ChainList):
    """ Deconvolution of ICLRAcerHead """

    def __init__(self, n_input_channels, n_output_channels,
                 activation=F.relu, bias=0.1, input_size=(84, 84)):
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.activation = activation

        w = chainer.initializers.HeNormal()

        fl_input = (np.array(input_size, dtype="int32") - 8)/2 + 1
        sl_input = (fl_input - 7)/2 + 1
        tl_input = (sl_input - 3)/2 + 1

        self.tl_input = tl_input

        linear_input = tl_input[0] * tl_input[1] * 64

        layers = [
            L.Linear(n_input_channels, linear_input,
                     initial_bias=bias),
            L.Deconvolution2D(64,
                              out_channels=32,
                              ksize=3,
                              stride=2,
                              initialW=w,
                              initial_bias=bias),
            L.Deconvolution2D(32,
                              out_channels=32,  # 64
                              ksize=7,
                              stride=2,
                              initialW=w,
                              initial_bias=bias),
            L.Deconvolution2D(32,  # 64
                              out_channels=n_output_channels,
                              ksize=8,
                              stride=2,
                              initialW=w,
                              initial_bias=bias)
            ]
        super(ICLRSimDeconvv2, self).__init__(*layers)

    def __call__(self, state):
        h = state

        for i in range(len(self)):
            if i == 1:
                h = F.reshape(h, (h.data.shape[0], 64,
                                  self.tl_input[0],
                                  self.tl_input[1]))  # FIXME. hardcoded
                h = self.activation(self[i](h))
            elif i == len(self) - 1:
                #                print "Fluxiiiii"
                # last layer => no activation
                h = self[i](h)
            else:
                h = self.activation(self[i](h))

        return h
#        for layer in self:
#            h = self.activation(layer(h))
#        return h


class ICLRSimHeadv3(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1, input_size=(84, 84)):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        w = chainer.initializers.HeNormal()

        # Layers (84x84)
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 8) / 2 + 1, 39 x 39 x 64
        # 2nd layer: (39 - 7)/2 + 1, 17 x 17 x 128
        # 3rd layer: (17 - 3)/2 + 1, 8 x 8 x 128 = 8192

        # Layers 92x92
        # 1st layer: (92 - 8)/2 + 1, 43 x 43 x 64
        # 2nd layer: (43-7)/2 + 1, 19 x 19 x 128
        # 3rd layer: (19 - 3)/2 + 1, 9 x 9 x 128 = 10368

        # Layers (100x100)
        # 1st layer: (100 - 8)/2 + 1, 47 x 47 x 64
        # 2nd layer: (47 - 7)/2 + 1, 21 x 21 x 128
        # 3rd layer: (21 - 3)/2 + 1, 10 x 10 x 128 = 12800

        # Layers (84x84)
        # Input should be 84 x 84 (n channels)
        # 1st layer: (84 - 4) / 2 + 1, 41 x 41 x 32
        # 2nd layer: (41 - 3)/2 + 1, 20 x 20 x 32
        # 3rd layer: (20 - 2)/1 + 1, 19 x 19 x 32
        # 4th layer: (19 -3)/2 + 1,  9 x 9 x 64
        
        fl_input = (np.array(input_size, dtype="int32") - 4)/2 + 1
        sl_input = (fl_input - 3)/2 + 1
        sl_2_input = (sl_input - 2)/1 + 1
        tl_input = (sl_2_input - 3)/2 + 1

        linear_input = tl_input[0] * tl_input[1] * 64

        print("FL INPUT ", fl_input)
        print("SL INPUT ", sl_input)
        print("SL2 INPUT ", sl_2_input)
        print("TL INPUT ", tl_input)
        print("LINEAR INPUT ", linear_input)

        layers = [
            L.Convolution2D(
                n_input_channels,
                out_channels=32,  # 64 initially
                ksize=4,
                stride=2,
                initialW=w,
                initial_bias=bias),
            L.Convolution2D(32,  # 64 initially
                            out_channels=32,
                            ksize=3,
                            stride=2,
                            initialW=w,
                            initial_bias=bias),
            
            L.Convolution2D(32,  # 64 initially
                            out_channels=32,
                            ksize=2,
                            stride=1,
                            initialW=w,
                            initial_bias=bias),

            L.Convolution2D(32,
                            out_channels=64,
                            ksize=3,
                            stride=2,
                            initialW=w,
                            initial_bias=bias),

            L.Linear(linear_input, n_output_channels,   # FIXME, check INPUT
                     initialW=LeCunNormal(1e-3),
                     initial_bias=bias),
        ]
        super(ICLRSimHeadv3, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class ICLRSimDeconvv3(chainer.ChainList):
    """ Deconvolution of ICLRAcerHead """

    def __init__(self, n_input_channels, n_output_channels,
                 activation=F.relu, bias=0.1, input_size=(84, 84)):
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.activation = activation

        w = chainer.initializers.HeNormal()

        fl_input = (np.array(input_size, dtype="int32") - 4)/2 + 1
        sl_input = (fl_input - 3)/2 + 1
        sl_2_input = (sl_input - 2)/1 + 1
        tl_input = (sl_2_input - 3)/2 + 1

        self.tl_input = tl_input

        linear_input = tl_input[0] * tl_input[1] * 64

        layers = [
            L.Linear(n_input_channels, linear_input,
                     initial_bias=bias),
            L.Deconvolution2D(64,
                              out_channels=32,
                              ksize=3,
                              stride=2,
                              initialW=w,
                              initial_bias=bias),

            L.Deconvolution2D(32,
                              out_channels=32,  # 64
                              ksize=2,
                              stride=1,
                              initialW=w,
                              initial_bias=bias),

            L.Deconvolution2D(32,
                              out_channels=32,  # 64
                              ksize=3,
                              stride=2,
                              initialW=w,
                              initial_bias=bias),
            L.Deconvolution2D(32,  # 64
                              out_channels=n_output_channels,
                              ksize=4,
                              stride=2,
                              initialW=w,
                              initial_bias=bias)
            ]
        super(ICLRSimDeconvv3, self).__init__(*layers)

    def __call__(self, state):
        h = state

        for i in range(len(self)):
            if i == 1:
                h = F.reshape(h, (h.data.shape[0], 64,
                                  self.tl_input[0],
                                  self.tl_input[1]))  # FIXME. hardcoded
                h = self.activation(self[i](h))
            elif i == len(self) - 1:
                #                print "Fluxiiiii"
                # last layer => no activation
                h = self[i](h)
            else:
                h = self.activation(self[i](h))

        return h
#        for layer in self:
#            h = self.activation(layer(h))
#        return h
