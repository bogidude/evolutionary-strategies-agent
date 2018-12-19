#!/usr/bin/env python
import subprocess as sp
import os
import re
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
    parser.add_argument(
        "-n", '--new_terminal', action="store_true",
        help=("whether to start tmux detached in the current terminal "
              "(default) or in a new gnome-terminal"))
    parser.add_argument(
        "-p", '--port', default="12345",
        help='tensorboard port (default 12345)')
    parser.add_argument(
        "-c", '--cluster_info', nargs='*', help='node ips')
    parser.add_argument(
        "--nvim", action="store_true", help='run in nvim terminal')
    parser.add_argument(
        "--ram-thresh", type=float, default=0,
        help="ram usage before restarting job (local only, units are Gb)")

    NVIM_LISTEN_ADDRESS = \
        '/tmp/nvim_evolutionary_strat' + str(random.randint(0, 1e12))
    args = parser.parse_args()
    if args.local_env_setup:
        args.local_env_setup = os.path.abspath(args.local_env_setup)
    args.redis_conf = os.path.abspath(args.redis_conf)

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

    def nvim_terminal_mode():
        return r"<C-\><C-n>"

    def new_nvim_term():
        return nvim_terminal_mode() + \
            r":tabnew | term<cr>"

    def new_tmux_term(name):
        sp.check_call(
            ["tmux", "new-window", "-t", args.session_name, "-n", name])

    def ssh_login(address):
        return os.environ['USER'] + '@' + address

    def start_redis(addresses, port):
        cmd = "redis-server " + args.redis_conf
        # if port:
        #     cmd += " --protected-mode no "

        if addresses:
            cmd = " ssh {} '{}'".format(ssh_login(addresses[0]), cmd)

        if args.nvim:
            return nvim_rename_buffer("redis") + nvim_insert_mode() + \
                source_env_file() + nvim_insert_mode() + cmd + "<cr>"
        else:
            source_env_file()
            send_tmux_keys(cmd)

    def nvim_rename_buffer(name):
        return nvim_terminal_mode() + \
            r':file ' + name + "<cr>"

    def source_env_file():
        if args.local_env_setup:
            if args.nvim:
                return nvim_insert_mode() + \
                    "source " + args.local_env_setup + "<cr>"
            else:
                send_tmux_keys(r"source " + args.local_env_setup)
        else:
            return ""

    def start_master(addresses, socket_file, port):
        host = addresses[0] if addresses else 'localhost'
        start_cmd = \
            ("python -m es_distributed.main master "
             " --master_socket_path " + socket_file +
             " --master_host " + host +
             " --log_dir " + args.log_dir + " " +
             " --exp_file " + args.exp_file)
        if port: 
            start_cmd += " --master_port " + str(port) + " "

        if addresses:
            start_cmd = \
                "ssh {} 'source {} && {}'".format(
                    ssh_login(addresses[0]), args.local_env_setup, start_cmd)

        if args.nvim:
            return nvim_rename_buffer("master") + source_env_file() + \
                nvim_insert_mode() + start_cmd + "<cr>"
        else:
            new_tmux_term("master")
            source_env_file()
            send_tmux_keys(start_cmd)

    def start_workers(addresses, socket_file, port, new_term=True):
        host = addresses[0] if addresses else 'localhost'
        start_cmd = \
            (" python -m es_distributed.main workers "
             "--master_host " + host + " " +
             "--relay_socket_path " + socket_file)
        if port:
            start_cmd += " --master_port " + port

        if args.nvim:
            if not addresses:
                new_nvim_term_str = new_nvim_term() if new_term else ""
                return [new_nvim_term_str + nvim_rename_buffer("worker") +
                        source_env_file() + nvim_insert_mode() + start_cmd +
                        " --num_workers " + str(args.num_workers) + "<cr>"]
            else:
                out = []
                for i, addr in enumerate(addresses):
                    # 2 taken with redis/master on first worker
                    num_workers = 24 if i > 0 else 22
                    out.append(new_nvim_term() + nvim_rename_buffer(addr) +
                        nvim_insert_mode() + source_env_file() +
                        " ssh {} 'source {} && {}'<cr>".format(
                            ssh_login(addr),
                            args.local_env_setup,
                            start_cmd + " --num_workers " +
                            str(num_workers)))
            return out
        else:
            if not addresses:
                new_tmux_term("worker")
                source_env_file()
                send_tmux_keys(start_cmd)
            else:
                for i, addr in enumerate(addresses):
                    # 2 taken with redis/master on first worker
                    num_workers = 24 if i > 0 else 22
                    new_tmux_term(addr)
                    source_env_file()
                    send_tmux_keys("ssh {} 'source {} && {}'".format(
                        ssh_login(addr), args.local_env_setup,
                        start_cmd + " --num_workers " +
                        str(num_workers)))
            return None

    def start_tensorboard():
        cmd = "tensorboard --port {} --logdir {}" .format(
            args.port, args.log_dir)
        if args.nvim:
            return nvim_rename_buffer("tensorboard") + source_env_file() + \
                nvim_insert_mode() + cmd + "<cr>"
        else:
            new_tmux_term("tensorboard")
            send_tmux_keys(cmd)

    def get_ending_digits(s):
        r = re.compile(r'\d')
        i = next((i for i in range(len(s) - 1, -1, -1) if not r.match(s[i])))
        prefix = s[:i + 1]
        digits = s[i + 1:]
        return prefix, int(digits)

    def parse_addresses():
        addresses = []
        if args.cluster_info:
            for addr in args.cluster_info:
                if '-' in addr:
                    beg, end = addr.split("-")
                    beg_prefix, beg_digit = get_ending_digits(beg)
                    end_prefix, end_digit = get_ending_digits(end)
                    err_msg = "node prefixes don't match {} vs {}".format(
                        beg_prefix, end_prefix)
                    assert beg_prefix == end_prefix, err_msg
                    addresses += [beg_prefix + str(i)
                                  for i in range(beg_digit, end_digit + 1)]
                else:
                    addresses.append(addr)
        return addresses

    def start_tmux():
        if args.new_terminal:
            sp.check_call([
                "gnome-terminal", "--", "tmux",
                "new-session", "-s", args.session_name,
                "-n", "redis"])
        else:
            sp.check_call([
                "tmux", "new-session", "-s", args.session_name,
                "-n", "redist", '-d'])

        if args.local_env_setup:
            sp.check_call(
                ["tmux", "send-keys", 'source ' + args.local_env_setup])

        send_tmux_enter()

    def get_config_val(prefix):
        r = re.compile(prefix + r'\s+(.*)')
        with open(args.redis_conf, 'r') as f:
            return next((r.match(l) for l in f.read().splitlines()
                         if r.match(l))).group(1)

    def get_port():
        try:
            port = get_config_val('port')
        except StopIteration:
            port = '6379'

        if port == '0':
            port = None

        return port

    def nvim_last_tab():
        return nvim_terminal_mode() + r":$tabnext<cr>"

    def start(addresses, socket_file, port):

        start_tmux()

        if args.nvim:
            send_tmux_keys(
                r'NVIM_LISTEN_ADDRESS=' + NVIM_LISTEN_ADDRESS + '; nvim')
            wait_for_nvr()

            start_term = "<esc>:term<cr>"

            send_nvim_keys(start_term + start_redis(addresses, port))
            send_nvim_keys(new_nvim_term() + start_tensorboard())
            send_nvim_keys(
                new_nvim_term() + start_master(addresses, socket_file, port))

            for cmd in start_workers(addresses, socket_file, port):
                send_nvim_keys(cmd)

        else:
            source_env_file()
            start_redis(addresses, port)
            start_tensorboard()
            start_master(addresses, socket_file, port)
            start_workers(addresses, socket_file, port)

    addresses = parse_addresses()
    socket_file = get_config_val('unixsocket')
    port = get_port()

    start(addresses, socket_file, port)
    if args.ram_thresh:
        while True:
            time.sleep(5)
            if psutil.virtual_memory().available / 1e9 < args.ram_thresh:
                print('shutting down and restarting job')
                sp.call(['pkill', '-9', '-f', 'es_distributed.main workers'])
                if args.nvim:
                    send_nvim_keys(nvim_last_tab())
                    for cmd in start_workers(
                            addresses, socket_file, port, False):
                        send_nvim_keys(cmd)
                else:
                    start_workers(addresses, socket_file, port)


if __name__ == '__main__':
    main()
