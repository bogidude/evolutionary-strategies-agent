import logging
import pickle

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from . import tf_util as U

import gym

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, render=False, timestep_limit=10000, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []
        t = 0

        tuple_space = isinstance(env.observation_space, gym.spaces.Tuple)
        if tuple_space:
            n = len(env.observation_space.spaces)
            discrete_spaces = \
                [self.space_to_int(env.action_space.spaces[i]) for i in range(n)]
        else:
            n = 1
            discrete_spaces = [self.space_to_int(env.action_space)]

        if save_obs:
            obs = []

        converted_actions = [None for _ in range(n)]
        ob = env.reset()
        if not tuple_space:
            ob = [ob]
        last_features = [self.get_initial_features() for _ in range(n)]
        for print_q in range(int(timestep_limit)):
            for i in range(n):
                fetched = self.act(ob, random_stream=random_stream, features=last_features[i])
                ac, value_, last_features[i] = fetched[0], fetched[1], fetched[2:]
                if discrete_spaces[i] == 0:
                    converted_actions[i] = ac.argmax()
                elif discrete_spaces[i] == 1 & tuple_space:
                    converted_actions[i] = np.unravel_index(
                        ac.argmax(), env.action_space.spaces[i].nvec)
                elif discrete_spaces[i] == 1:
                    converted_actions[i] = list(np.unravel_index(
                        ac.argmax(), env.action_space.nvec))
                else:
                    converted_actions[i] = float(action)

            if not tuple_space:
                converted_actions = converted_actions[0]
            if save_obs:
                obs.append(ob)
            ob_, rew, done, info = env.step(converted_actions)
            ob_ = np.copy(ob_)

            if tuple_space:
                if ('scrimmage_err' in info or
                   any([s.size == 0 for s in temp_last_state])):
                    done = True
                else:
                    ob = ob_
            else:
                if not done:
                    ob = [ob_]
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break

        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)
        return rews, t

    def act(self, ob, random_stream=None, features=None):
        raise NotImplementedError

    def get_initial_features(self):
        pass

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def space_to_int(self, space):
        if isinstance(space, gym.spaces.Discrete):
            return 0
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return 1
        elif isinstance(space, gym.spaces.Tuple):
            return 2
        else:
            return 3

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


def bins(x, dim, num_bins, name):
    scores = U.dense(x, dim * num_bins, name, U.normc_initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1

class MultinomialWithEntropy(tf.distributions.Multinomial):
    def __init__(self, *args, **kwargs):
        super(MultinomialWithEntropy, self).__init__(*args, **kwargs)

    def entropy(self):
        log_prob = tf.log(tf.clip_by_value(self._probs, 1.0e-9, 1.0))
        return -tf.reduce_sum(log_prob * self._probs)

def flatten(x):
    """Move 2d tensor to 1d (the -1 is similar to None).

    Note that this method does not do anything if the tensor is 1d.
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


class LSTMPolicy(Policy):

    # Helper functions that might be needed
    def normalized_columns_initializer(self, std=1.0):
        """Create initializer function (see use in function linear)."""
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    def _features_to_feed_dict(self, feed_dict, features):
        for i, (state_in_c, state_in_h) in enumerate(self.state_in):
            feed_dict[state_in_c] = features[i][0]
            feed_dict[state_in_h] = features[i][1]
        return feed_dict

    def lstm_layer(self, scope, inp, width, sequence_length,keep_prob=1.0):
        with tf.variable_scope(scope):
            lstm = rnn.BasicLSTMCell(width, state_is_tuple=True)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)
            layer_inputs_init = \
                [np.zeros((1, lstm.state_size.c), np.float32),
                 np.zeros((1, lstm.state_size.h), np.float32)]

            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            layer_inputs_placeholders = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_final_state = \
                tf.nn.dynamic_rnn(
                    lstm, inp, initial_state=state_in,
                    sequence_length=sequence_length,
                    time_major=False)

            lstm_output_flat = tf.reshape(lstm_outputs, [-1, width])
            # lstm_output_batch_normed = \
            #     tf.contrib.layers.batch_norm(lstm_output_flat, center=True,
            #                                  scale=True, is_training=True,
            #                                  scope='bn')

            # lstm_output_nonlinearity = tf.nn.relu(lstm_output_flat)

        return (layer_inputs_init,
                layer_inputs_placeholders,
                lstm_output_flat,
                lstm_final_state)


    def conv2d(self, x, num_filters, name, filter_size=(3, 3),
               stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
        """Perform convolution. Small wrapper around tf.nn.conv2d."""
        with tf.variable_scope(name):

            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0],         # patch size x
                            filter_size[1],         # patch size y
                            int(x.get_shape()[3]),  # input channels
                            num_filters]            # output channels

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = np.prod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W",
                                filter_shape,
                                dtype,
                                tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b",
                                [1, 1, 1, num_filters],
                                initializer=tf.constant_initializer(0.0),
                                collections=collections)

            return tf.nn.conv2d(x, w, stride_shape, pad) + b


    def linear(self, x, size, name, initializer=None, bias_init=0):
        """Create weight/bias in "name" scope and perform affine transform."""
        w = tf.get_variable(name + "/w",
                            [x.get_shape()[1], size],
                            initializer=initializer)
        b = tf.get_variable(name + "/b",
                            [size],
                            initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b


    def multinomial_dist(logits):
        mx = tf.reduce_max(logits, [1], keep_dims=True)
        centered_logits = logits - mx
        return tf.multinomial(logits=centered_logits, num_samples=1)


    def categorical_sample(logits, d):
        """Sample the actor output w/ logit distribution/return as one-hot."""
        value = tf.squeeze(tf.multinomial(
            logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
        return tf.one_hot(value, d)


    def get_initial_features(self):
        """Return tf.Variable for initial state."""
        return self.state_init


    def _initialize(self, ob_space, ac_space, architecture, use_conv_layer=False):
        """Create convolutional layer followed 1 layer LSTM."""
        # super(LSTMPolicyBase, self).__init__(ob_space, ac_space, architecture)
        ############################################################
        # This first section defines the size of x to be (1, ?, ob_space)
        # where ob_space is the size of the observation space. when obs_space
        # is 2d, the size is the number of elements in obs_space.
        # ? is the batch size to be filled in later
        # the [None] is the batch size to be filled in later
        ############################################################
        with tf.variable_scope(type(self).__name__) as scope:
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            # Placeholder for keep probabliltiy for dropout, set to 1 during testing
            self.keep_prob = tf.placeholder_with_default(1.0,[])
            if use_conv_layer:
                # the first 4 hidden layers are convolutional layers
                # where the filter size is 3x3 and strides are of length 2
                # it is followed by the (nonlinear) exponential linear unit
                # each hidden layer halves the size of the observation due to the
                # 2-step stride
                for i in range(4):
                    x = tf.nn.elu(self.conv2d(x, 32, "l{}".format(i + 1),
                                         [3, 3], [2, 2]))

            ############################################################
            # setup the RNN which consists of a bunch of LSTM cells
            # see also: https://www.tensorflow.org/tutorials/recurrent
            ############################################################
            sequence_length = tf.shape(self.x)[:1]

            output_values = \
                tf.contrib.layers.batch_norm(
                    x, center=True, scale=True, is_training=True, scope='bn')
            output_values = x
            self.state_init = []
            self.state_in = []
            self.state_out = []
            for i, layer_i in enumerate(architecture):
                import pdb
                layer_type = layer_i['layer_type']
                width = layer_i['width']
                if layer_type == "lstm":
                    # introduce a "fake" batch dimension of 1 after flatten so that
                    # we can do LSTM over time dim after the flatten operation the
                    # tensor will have shape (?, obs_space) after the expand dims
                    # it will have shape (1, ?, obs_space)
                    flattened_output_values = \
                        tf.expand_dims(flatten(output_values), [0])

                    layer = self.lstm_layer("l{}".format(i + 1),
                                       flattened_output_values,
                                       width, sequence_length, self.keep_prob)
                    self.state_init.append(layer[0])
                    self.state_in.append(layer[1])
                    output_values = layer[2]
                    output_c, output_h = layer[3]
                    self.state_out.append([output_c[:1, :], output_h[:1, :]])
                elif layer_type == "fc":
                    self.fc = self.linear(output_values, width, "fc{}".format(i+1),
                                     self.normalized_columns_initializer(0.01))
                    self.fc_nonlinear = tf.nn.relu(self.fc)
                    self.fc_nonlinear_drop = tf.nn.dropout(self.fc_nonlinear,self.keep_prob)

                    output_values = self.fc_nonlinear_drop

            ############################################################
            # create the output layer
            # logits:
            #   affine transformation of x mapped to size ac_space.
            #   these are later interpreted as unnormalized log-probabilities
            #   in the categorical_sample function (see tf.multinomial)
            #
            # vf: value function (output size 1)
            ############################################################
            if isinstance(ac_space, gym.spaces.Discrete):
                self.logits = self.linear(output_values, ac_space.n, "action",
                                     self.normalized_columns_initializer(0.01))
                self.probs = tf.nn.softmax(self.logits)
                self.dist = \
                    MultinomialWithEntropy(total_count=1., probs=self.probs)
            elif isinstance(ac_space, gym.spaces.MultiDiscrete):
                self.logits = []
                self.probs_list = []
                self.max_action_space = max(ac_space.nvec)
                num_actions_per = [a for a in ac_space.nvec]

                self.logits = self.linear(output_values, np.prod(num_actions_per), "action{}".format(i+1),
                                     self.normalized_columns_initializer(0.01))
                self.probs = tf.nn.softmax(self.logits)
                # Add uniform distribution to maintain minimum probabilities
                uniform_dist = tf.ones(shape=tf.shape(self.probs))/np.prod(num_actions_per)
                omega = 0.9999
                self.probs = omega*self.probs + (1-omega)*uniform_dist
                self.dist = \
                    MultinomialWithEntropy(total_count=1., probs=self.probs)
            else:
                self.stats = {}
                use_normal = True
                if use_normal:
                    self.stats['mean'] = \
                        self.linear(output_values, 1, "mean",
                               self.normalized_columns_initializer(0.01),
                               bias_init=0)
                    self.stats['stdev'] = \
                        tf.exp(self.linear(output_values, 1, "stdev",
                                      self.normalized_columns_initializer(0.01),
                                      bias_init=0))
                    self.dist = tf.contrib.distributions.Normal(
                        self.stats['mean'], self.stats['stdev'])
                else:
                    self.stats['alpha'] = \
                        tf.exp(self.linear(output_values, 1, "alpha",
                                      self.normalized_columns_initializer(0.01),
                                      bias_init=0))
                    self.stats['beta'] = \
                        tf.exp(self.linear(output_values, 1, "beta",
                                      self.normalized_columns_initializer(0.01),
                                      bias_init=0))
                    self.dist = tf.distributions.Beta(
                        self.stats['alpha'], self.stats['beta'])

            self.sample = tf.squeeze(self.dist.sample([1]))

            self.vf = \
                tf.reshape(self.linear(output_values, 1, "value",
                                  self.normalized_columns_initializer(1.0)), [-1])
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              tf.get_variable_scope().name)
        return scope


    def act(self, observation, features, random_stream=None):
        """Return action, value, and output features.

        action - one-hot sampling of the output of the actor
        value - scalar of the value function
        features
        """
        sess = tf.get_default_session()
        feed_dict = {self.x:  observation}
        feed_dict = self._features_to_feed_dict(feed_dict, features)
        feed_dict[self.x] = observation
        return sess.run([self.sample, self.vf] + self.state_out, feed_dict)


    def initialize_from(self, filename, ob_stat=None):
        pass

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False
    
    


class MujocoPolicy(Policy):
    def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.ac_bins = ac_bins
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1
        assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
            'Action bounds required'

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': U.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = U.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
        assert isinstance(self.ac_bins, str)
        ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')

        if ac_bin_mode == 'uniform':
            # Uniformly spaced bins, from ac_space.low to ac_space.high
            num_ac_bins = int(ac_bin_arg)
            aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
            ac_range_1a = (ahigh - alow)[None, :]
            a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]

        elif ac_bin_mode == 'custom':
            # Custom bins specified as a list of values from -1 to 1
            # The bins are rescaled to ac_space.low to ac_space.high
            acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
            logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
            assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
            acvals_ak = (
                (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
                + alow[:, None]
            )

            aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
            a = tf.gather_nd(
                acvals_ak,
                tf.concat(2, [
                    tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
                    tf.expand_dims(aidx_na, -1)
                ])  # (n,a,2)
            )  # (n,a)
        elif ac_bin_mode == 'continuous':
            a = U.dense(x, adim, 'out', U.normc_initializer(0.01))
        else:
            raise NotImplementedError(ac_bin_mode)

        return a

    def act(self, ob, random_stream=None):
        a = self._act(ob)
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a

    @property
    def needs_ob_stat(self):
        return True

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)
