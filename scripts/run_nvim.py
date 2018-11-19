import subprocess as sp
import os
import argparse
import time
import multiprocessing as mp
import random
import psutil

try:
    import lvdb
except ImportError:
    pass


def print_sockaddrs():
    """Copied from neovim-remote.

    nvr does not work with python2 but this function does.
    """
    sockaddrs = []

    for proc in psutil.process_iter():
        if proc.name() == 'nvim':
            for conn in proc.connections('inet4'):
                sockaddrs.insert(0, ':'.join(map(str, conn.laddr)))
            for conn in proc.connections('unix'):
                if conn.laddr:
                    sockaddrs.insert(0, conn.laddr)

    return sockaddrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "redis_conf", help="Redis configuration file (e.g., redis_master.conf")
    parser.add_argument("exp_file", help="Experiment file (e.g., action.json)")
    parser.add_argument(
        "-s", "--session-name", default="",
        help='tmux session name, default="exp_m_d_H_M_s"')
    parser.add_argument(
        "-e", "--local-env-setup", default="",
        help='file to source prior to starting terminals')
    parser.add_argument(
        '-j', '--num_workers', type=int, default=mp.cpu_count())
    parser.add_argument(
        "-l", '--log_dir', default="",
        help="where to store tensorboard output")

    NVIM_LISTEN_ADDRESS = \
        '/tmp/nvim_evolutionary_strat' + str(random.randint(0, 1e12))
    args = parser.parse_args()

    if not args.session_name:
        args.session_name = time.strftime("exp_%m_%d_%H_%M_%S")

    args.log_dir = os.path.expanduser(args.log_dir) if args.log_dir \
        else '/tmp/es_master_{}'.format(os.getpid())

    def send_tmux_enter():
        sp.check_call(["tmux", "send-keys", "Enter"])

    def send_tmux_keys(cmd):
        sp.check_call(["tmux", "send-keys", '-t', args.session_name, cmd])
        send_tmux_enter()

    def send_nvim_keys(cmd):
        sp.check_call(
            ['nvr', '--servername', NVIM_LISTEN_ADDRESS, '--remote-send', cmd])

    def nvim_insert_mode():
        return "<esc>A"

    def wait_for_nvr():
        while NVIM_LISTEN_ADDRESS not in print_sockaddrs():
            time.sleep(0.1)

    def start_master():
        start_cmd = \
            ("python -m es_distributed.main master "
             "--master_socket_path /tmp/redis.sock "
             "--log_dir " + args.log_dir + " " +
             "--exp_file " + args.exp_file + "<cr>")
        return nvim_rename_buffer("master") + \
            nvim_source_env_file() + \
            nvim_insert_mode() + \
            start_cmd

    def nvim_terminal_mode():
        return r"<C-\><C-n>"

    def new_nvim_term():
        return nvim_terminal_mode() + \
            r":tabnew | term<cr>"

    def start_redis():
        return nvim_rename_buffer("redis") + \
            nvim_insert_mode() + \
            nvim_source_env_file() + \
            nvim_insert_mode() + \
            "redis-server " + args.redis_conf + "<cr>"

    def nvim_rename_buffer(name):
        return nvim_terminal_mode() + \
            r':file ' + name + "<cr>"

    def nvim_source_env_file():
        if args.local_env_setup:
            return nvim_insert_mode() + \
                "source " + args.local_env_setup + "<cr>"
        else:
            return ""

    def start_workers():
        start_cmd = \
            ("python -m es_distributed.main workers "
             "--master_host localhost --relay_socket_path "
             "/tmp/redis.sock --num_workers " + str(args.num_workers) + "<cr>")
        return nvim_rename_buffer("worker") + \
            nvim_source_env_file() + \
            nvim_insert_mode() + \
            start_cmd

    def start_tensorboard():
        return nvim_rename_buffer("tensorboard") + \
            nvim_source_env_file() + \
            nvim_insert_mode() + \
            "tensorboard --logdir " + args.log_dir + "<cr>"

    sp.check_call(["gnome-terminal", "-x",
                   "tmux", "new", "-s", args.session_name])
    if args.local_env_setup:
        sp.check_call(
            ["tmux", "send-keys", 'source ' + args.local_env_setup])

    send_tmux_enter()
    send_tmux_keys(r'NVIM_LISTEN_ADDRESS=' + NVIM_LISTEN_ADDRESS + '; nvim')
    wait_for_nvr()

    start_term = "<esc>:term<cr>"
    cmd = start_term + \
        start_redis() + \
        new_nvim_term() + \
        start_tensorboard() + \
        new_nvim_term() + \
        start_master() + \
        new_nvim_term() + \
        start_workers()

    send_nvim_keys(cmd)

if __name__ == '__main__':
    main()
