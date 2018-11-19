"""Main code for starting master and workers."""
import errno
import json
import logging
import os
import sys

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
@click.option('--log_dir')
def master(exp_str, exp_file, master_socket_path, log_dir):
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
    run_master({'unix_socket_path': master_socket_path}, log_dir, exp)


@cli.command()
@click.option('--master_host', required=True)
@click.option('--master_port', default=6379, type=int)
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
    master_redis_cfg = {'unix_socket_path': relay_socket_path}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        # This is the interconnect between master and the workers
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return
    # Start the workers
    noise = SharedNoiseTable()  # Workers share the same noise
    num_workers = num_workers if num_workers else os.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            run_worker(relay_redis_cfg, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()
