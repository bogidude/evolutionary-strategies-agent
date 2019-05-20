"""Main code for starting master and workers."""
import errno
import json
import logging
import os
import sys
import signal
import multiprocessing as mp

import click

from .dist import RelayClient
from .es import run_master, run_worker, SharedNoiseTable

try:
    import lvdb
except ImportError:
    pass

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)


@cli.command()
@click.option('--exp_str')
@click.option('--exp_file')
@click.option('--master_socket_path', required=True)
@click.option('--master_port', default=None)
@click.option('--master_host')
@click.option('--log_dir')
def master(exp_str, exp_file, master_socket_path, log_dir, master_host, master_port):
    """Load json params and call run_master."""
    # Start the master
    assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
    if exp_str:
        exp = json.loads(exp_str)
    elif exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)

    if master_port:
        redis_cfg = {'host': master_host, 'port': master_port}
    else:
        redis_cfg = {'unix_socket_path': master_socket_path}
    run_master(redis_cfg, log_dir, exp)


@cli.command()
@click.option('--master_host', required=True)
@click.option('--master_port', default=None, type=int)
@click.option('--relay_socket_path', required=True)
@click.option('--num_workers', type=int, default=0)
def workers(master_host, master_port, relay_socket_path, num_workers):
    """Start num_workers that wait for tasks delivered by master.

    The following are started:
    1 RelayClient - this is the interconnect between master and the workers
    n workers

    See https://stackoverflow.com/a/24538608 for details on fork.
    Note that memory prior to the fork is copied and memory is not shared
    between the forks. os.fork() == 0 for the new process.
    """
    # Start the relay
    # master_redis_cfg = {'host': master_host, 'port': master_port}
    if master_port:
        master_redis_cfg = {'host': master_host, 'port': master_port}
        relay_redis_cfg = {'host': master_host, 'port': master_port}
    else:
        master_redis_cfg = {'unix_socket_path': relay_socket_path}
        relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    child_pid_list = []
    relay_pid = os.fork()
    if relay_pid == 0:
        # This is the interconnect between master and the workers
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return
    child_pid_list.append(relay_pid)
    # Start the workers
    noise = SharedNoiseTable()  # Workers share the same noise
    num_workers = num_workers if num_workers else mp.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    def shutdown(signal_, frame):
        logging.info("Received {} on pid: {}".format(signal_, os.getpid()))
        for pid in child_pid_list:
            logging.info("Killing {}".format(pid))
            os.kill(pid, signal.SIGINT)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for _ in range(num_workers):
        os_child_id = os.fork()
        if os_child_id == 0:
            run_worker(relay_redis_cfg, noise=noise)
            return
        else:
            child_pid_list.append(os_child_id)
    os.wait()


if __name__ == '__main__':
    cli()
