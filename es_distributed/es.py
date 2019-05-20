import logging
import time
import sys
from collections import namedtuple

import numpy as np

from .dist import MasterClient, WorkerClient
import gym
import scrimmage
import scrimmage.utils
import scrimmage.bindings
import tensorflow as tf
import os
import pdb
import shutil

try:
    import lvdb
except ImportError:
    pass


logger = logging.getLogger(__name__)

Config = namedtuple('Config', [
    'l2coeff', 'noise_stdev', 'episodes_per_batch', 'timesteps_per_batch',
    'calc_obstat_prob', 'eval_prob', 'snapshot_freq',
    'return_proc_mode', 'episode_cutoff_mode', 'num_models_to_save', 'save_every_n_model'
])
Task = namedtuple('Task', ['params', 'ob_mean', 'ob_std', 'timestep_limit'])
Result = namedtuple('Result', [
    'worker_id',
    'noise_inds_n', 'returns_n2', 'signreturns_n2', 'lengths_n2',
    'eval_return', 'eval_length',
    'ob_sum', 'ob_sumsq', 'ob_count', 'info'
])


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 100000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        logger.info('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def make_session(single_threaded):
    import tensorflow as tf
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def create_scrimmage_env(env_id, visualise, port_num, scrimmage_mission,
                         timestep, global_sensor, combine_actors, static_obs_space = "True"):
    """Add scrimmage to gym registry."""
    try:
        return gym.make(env_id)
    except gym.error.Error:
        mission_file = \
            scrimmage.utils.find_mission(scrimmage_mission)

        address = "localhost:" + str(port_num)
        boole = lambda x: x == "True" or x == "true"
        gym.envs.register(
            id=env_id,
            entry_point='scrimmage.bindings:ScrimmageOpenAIEnv',
            max_episode_steps=1e9,
            reward_threshold=1e9,
            kwargs={"enable_gui": boole(visualise),
                    "mission_file": mission_file,
                    "global_sensor": boole(global_sensor),
                    "static_obs_space": boole(static_obs_space),
                    "timestep": float(timestep),
                    "combine_actors": boole(combine_actors)}
        )

        spec = gym.envs.registration.registry.spec(env_id)
        return gym.make(env_id)


def setup(exp, single_threaded):
    # gym.undo_logger_setup()
    from . import policies, tf_util

    config = Config(**exp['config'])
    if exp['exp_prefix'] == "scrimmage":
        env = create_scrimmage_env(**exp['env'])
    else:
        env = gym.make(exp['env_id'])
    sess = make_session(single_threaded=single_threaded)

    # in the case that we have Tuple spaces we make the assumption
    # that all of the spaces are the same so that we can use a shared
    # model
    if isinstance(env.action_space, gym.spaces.Tuple):
        observation_space = env.observation_space.spaces[0]
        action_space = env.action_space.spaces[0]
    else:
        observation_space = env.observation_space
        action_space = env.action_space

    policy = getattr(policies, exp['policy']['type'])(observation_space, action_space, **exp['policy']['args'])

    tf_util.initialize()

    return config, env, sess, policy


def run_master(master_redis_cfg, log_dir, exp):
    """

    Parameters
    ----------

    master_redis_cfg : dict
        defines how to connect to master using redis (i.e., the socket path)

    log_dir : str
        where to log

    exp : dict
        experiment configuration (hyperparams, gym env, policy, etc)
        see action.json for an example

    There is a distinction between an experiment and a task. A Task is the
    set of parameters to use for an experiment. Specifically, given a gym
    environment, a task is the specific policy to use on that environment.
    """

    ###########################################################
    # initialization: logging, environment, noise, connection to master, policy
    ###########################################################
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env, sess, policy = setup(exp, single_threaded=False)
    mission_file = \
            scrimmage.utils.find_mission(exp['env']['scrimmage_mission'])
    save_mission_loc = log_dir + "/config/" + exp['env']['scrimmage_mission']
    save_mission_loc_final = save_mission_loc
    counter = 1
    while os.path.exists(save_mission_loc_final):
        save_mission_loc_final = save_mission_loc[:-4] + "_{}.xml".format(counter)
        counter = counter + 1
    shutil.copy2(mission_file, save_mission_loc_final)
    master = MasterClient(master_redis_cfg)
    optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](policy, **exp['optimizer']['args'])
    noise = SharedNoiseTable()
    rs = np.random.RandomState()
    ob_stat = RunningStat(
        env.observation_space.shape,
        eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
    )
    if 'init_from' in exp['policy']:
        logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
        policy.initialize_from(exp['policy']['init_from'], ob_stat)

    if config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = int(arg0), float(arg1), float(arg2)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio))
    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio = None, None, None
        tslimit = 1000
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)

    ###########################################################
    # main learning loop
    ###########################################################
    saver = tf.train.Saver(max_to_keep=config.num_models_to_save)
    last_checkpoint = tf.train.latest_checkpoint(os.path.abspath(log_dir))
    if last_checkpoint:
        saver.restore(sess, last_checkpoint)

        # exploit that saver.save appends the global step
        master.task_counter = 1 + int(last_checkpoint.split('-')[-1])
        tlogger._Logger.CURRENT.tbwriter.step = master.task_counter

    while True:
        step_tstart = time.time()
        theta = policy.get_trainable_flat()
        assert theta.dtype == np.float32

        # declare the task
        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            timestep_limit=tslimit
        ))
        tlogger.info('********** Iteration {} **********'.format(curr_task_id))

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
        num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
        eval_infos, population_infos = [], []
        # lvdb.set_trace()
        while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
            # Wait for a result
            sys.stdout.write(
                'num_episodes popped is ' + str(num_episodes_popped) + '\r')
            sys.stdout.flush()
            task_id, result = master.pop_result()
            assert isinstance(task_id, int) and isinstance(result, Result)
            assert (result.eval_return is None) == (result.eval_length is None)
            worker_ids.append(result.worker_id)

            if result.eval_length is not None:
                # This was an eval job
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for current tasks
                if task_id == curr_task_id:
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
                    eval_infos.append(result.info)
            else:
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                assert result.returns_n2.dtype == np.float32
                # Update counts
                result_num_eps = result.lengths_n2.size
                result_num_timesteps = result.lengths_n2.sum()
                episodes_so_far += result_num_eps
                timesteps_so_far += result_num_timesteps
                # Store results only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    num_timesteps_popped += result_num_timesteps
                    population_infos += result.info
                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        # Compute skip fraction
        frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        if num_results_skipped > 0:
            logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                num_results_skipped, 100. * frac_results_skipped))

        # Assemble results
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif config.return_proc_mode == 'sign':
            proc_returns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])
        elif config.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(np.concatenate([r.signreturns_n2 for r in curr_task_results]))
        else:
            raise NotImplementedError(config.return_proc_mode)
        # Compute and take step
        g, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (noise.get(idx, policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio = optimizer.update(-g + config.l2coeff * theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if policy.needs_ob_stat:
            policy.set_ob_stat(ob_stat.mean, ob_stat.std)

        # Update number of steps to take
        if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            old_tslimit = tslimit
            tslimit = int(tslimit_incr_ratio * tslimit)
            logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

        step_tend = time.time()
        tlogger.record_tabular("Iteration", curr_task_id)
        tlogger.record_tabular("EpRewMean", returns_n2.mean())
        tlogger.record_tabular("EpRewStd", returns_n2.std())
        tlogger.record_tabular("EpLenMean", lengths_n2.mean())

        tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
        tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
        tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
        tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))
        tlogger.record_tabular("EvalEpCount", len(eval_rets))

        tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))
        tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
        tlogger.record_tabular("UpdateRatio", float(update_ratio))

        tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
        tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
        tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

        num_unique_workers = len(set(worker_ids))
        tlogger.record_tabular("UniqueWorkers", num_unique_workers)
        tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
        tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
        tlogger.record_tabular("ObCount", ob_count_this_batch)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)

        # just record the mean for now
        def record_info(data_type, dicts):
            keys = set((k for d in dicts for k in d.keys()))
            for key in keys:
                save_key = data_type + "/" + key
                data = np.array([float(d[key]) for d in dicts if key in d])
                tlogger.record_tabular(save_key, data.mean())
        record_info("Eval", eval_infos)
        record_info("Population", population_infos)

        tlogger.dump_tabular()
        if (curr_task_id % config.save_every_n_model) == 0:
            saver.save(sess, os.path.abspath(log_dir) + "/model.ckpt", global_step=curr_task_id)

        # if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
        #     import os.path as osp
        #     filename = osp.join(tlogger.get_dir(), 'snapshot_iter{:05d}_rew{}.h5'.format(
        #         curr_task_id,
        #         np.nan if not eval_rets else int(np.mean(eval_rets))
        #     ))
        #     assert not osp.exists(filename)
        #     policy.save(filename)
        #     tlogger.log('Saved snapshot {}'.format(filename))
        #     print("Saved snapshot to {}".format(filename))


def rollout_and_update_ob_stat(policy, env, timestep_limit, rs, task_ob_stat, calc_obstat_prob):
    if policy.needs_ob_stat and calc_obstat_prob != 0 and rs.rand() < calc_obstat_prob:
        rollout_rews, rollout_len, obs, info = policy.rollout(
            env, timestep_limit=timestep_limit, save_obs=True, random_stream=rs)
        task_ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
    else:
        rollout_rews, rollout_len, info = policy.rollout(env, timestep_limit=timestep_limit, random_stream=rs)
    return rollout_rews, rollout_len, info


def adjust_info(d):
    # for scrimmage openai environments
    try:
        out = d['info']
    except KeyError:
        out = {}
    for key in d:
        if key != 'info':
            out[key] = d[key]
    return out


def run_worker(relay_redis_cfg, noise, min_task_runtime=.2):

    ###########################################################
    # get parameters
    ###########################################################
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(relay_redis_cfg)
    exp = worker.get_experiment()
    config, env, sess, policy = setup(exp, single_threaded=True)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)
    win = 0
    loss = 0
    draw = 0
    ###########################################################
    # worker loop
    ###########################################################
    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)
        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length, info = policy.rollout(env)  # eval rollouts don't obey task_data.timestep_limit

            if not isinstance(info, list):
                info = [info]
                # convert to column vec for compatiblity with multi-vehicle
                # case
                # eval_rews = eval_rews[:, np.newaxis]

            # eval_rews = [eval_rews[:, i] for i in range(len(info))]
            eval_length = [eval_length for i in range(len(info))]

            val = 0
            for i in range(len(info)):
                eval_return = sum(eval_rews[i]) # sum agent i's reward over the entire mission
                val = val + eval_return
                logger.info('Eval result: task={} return={: 7.3f} length={}'.format(task_id, eval_return, eval_length))
                worker.push_result(task_id, Result(
                    worker_id=worker_id,
                    noise_inds_n=None,
                    returns_n2=None,
                    signreturns_n2=None,
                    lengths_n2=None,
                    eval_return=eval_return,
                    eval_length=eval_length[i],
                    ob_sum=None,
                    ob_sumsq=None,
                    ob_count=None,
                    info=adjust_info(info[i])
                ))
            if val > 0:
                    win = win + 1.0
            elif val < 0:
                loss = loss + 1.0
            else:
                draw = draw + 1.0
            total = int(win + loss + draw)
            w_rate = win / (total)
            logger.info("Win rate after {:3d} missions: {:4.3f}".format(total, w_rate))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths = [], [], [], []
            infos = []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:

                # sample the population
                noise_idx = noise.sample_index(rs, policy.num_params)
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)

                # evaluate the fitness
                policy.set_trainable_flat(task_data.params + v)
                rews_pos, len_pos, info_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg, info_neg = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)


                tuple_space = isinstance(env.observation_space, gym.spaces.Tuple)
                if tuple_space:
                    n = len(env.observation_space.spaces)
                    for i in range(n):
                        # Fix how rewards are handled here to allow for entities dying
                        noise_inds.append(noise_idx)
                        returns.append([sum(rews_pos[i]), sum(rews_neg[i])])
                        signreturns.append([np.sign(rews_pos[i]).sum(), np.sign(rews_neg[i]).sum()])
                        lengths.append([len_pos, len_neg])

                        adjusted_info_pos = adjust_info(info_pos[i])
                        adjusted_info_neg = adjust_info(info_neg[i])

                        infos.append(adjusted_info_pos)
                        infos.append(adjusted_info_neg)
                else:
                    noise_inds.append(noise_idx)
                    returns.append([sum(rews_pos), sum(rews_neg)])
                    signreturns.append([np.sign(rews_pos).sum(), np.sign(rews_neg).sum()])
                    lengths.append([len_pos, len_neg])
                    infos.append(info_pos)
                    infos.append(info_neg)

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count,
                info=infos
            ))
