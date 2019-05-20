
# Evolutionary Strategies Fork

This a fork of the openai [evolutionary-strategies](https://github.com/openai/evolution-strategies-starter) git repo. It currently works on one agent!
Please follow the installations and usage instructions below to see if it works for you as well. If you want to change the default setups of things, please look at the customization section.

## Todos:

The following work still needs to be done:

* Figure out how to setup redis to work on the cluster - whether or not it opens ports, etc (done)
* Figure out what neural network structure works best with evolutionary strategies
* Use tensorflow APIs so we don't define our own linear layer, flatten, normalized columns, etc. (still needs to completed)
* Update the running script to accept arguments like number of workers, where to connect for redis, save location (done)
* Figure out how the network gets saved and loaded (done)
* Double check multiple agents work (done)
* Turn this into a python package that can be installed? (done)

## Installation Instructions
First we will need to install redis (version 3.2 or greater).

### redis on Ubuntu 18.04

    $ sudo apt install redis # version is 4.0.9

### redis on Ubuntu 16.04

Sadly the apt-get package is not a new enough version so we will need to install from source. Here are the instructions I used from `scripts/dependencies.sh`

```bash
  $ wget --quiet http://download.redis.io/releases/redis-3.2.7.tar.gz -O redis-3.2.7.tar.gz
  $ tar -xvzf redis-3.2.7.tar.gz
  $ cd redis-3.2.7
  $ make
  $ sudo make install
  $ sudo mkdir -p /etc/redis
  $ sudo cp redis.conf /etc/redis # Setup default configuration file
  $ cd ..
  $ rm -rf redis-3.2.7 redis-3.2.7.tar.gz # Remove the installation files
```

### continuing ...

Now we need to setup the pip packages. I recommend using virtualenv to create a new python environment to install these packages in. Here is a quick how-to on setting up a new virtualenv environment "evol-strats" in the `~/venvs/` folder using python2.

```bash
  $ sudo apt-get install virtualenv -y
  $ virtualenv ~/venvs/evol-strats -p $(which python2)
  $ source ~/venvs/evol-strats/bin/activate # Enter virtual environment
```
Once the environment is set up, then the pip packages can be installed. The quickest way to install everything would be to just install this package.

```bash
  (evol-strats) $ cd /path/to/evolutionary-strategies-agent
  (evol-strats) $ pip install -e .
```
You can also install each package separately as seen below. Note these package versions are the ones that have been used. Others might work but there are no guarantees.
```bash
  (evol-strats) $ pip install click \
      h5py  \
      tensorflow==1.3.0 \
      gym==0.10.8 \
      grpcio==1.2.1 \
      protobuf==3.3.0 \
      redis \
      psutil \
      shutil
  (evol-strats) $ cd /path/to/scrimmage/python
```

The following command will need sudo if you installed the scrimmage python bindings for your system python to overwrite it in the virtualenv environment folder. It won't overwrite your system's scrimmage python binding. You might also need to run `scrimmage/setup/install-binaries.sh` with python 2 dependencies. More info on how to do that is located [here](https://github.com/gtri/scrimmage#install-binary-dependencies)

```bash
  (evol-strats) $ pip install -e .
```

To get out of the virtualenv environment, simply type `deactivate`.

## Usage

If you want to run something locally, ``redis_master.conf`` should have these lines:

    bind 127.0.0.1
    port 0

otherwise, bind should be the ip address of the host of your redis server and master node
and port should be whatever you want your traffic to go through. Also, setup your
``local_env_setup.sh`` file to source your environment:

    # an example
    source ~/.bashrc
    source ~/venvs/evol-strats/bin/activate
    cd ~/scrimmage/evolutionary-strategies-agent

If you use neovim with terminal mode, there is an additional dependency:

    # additionally install neovim remote
    $ pip install neovim-remote

For local training, do the following:

    $ python scripts/run_es.py \
        redis_config/redis_master.conf \
        configurations/scrimmage.json \
        -s exp \
        --local-env-setup scripts/local_env_setup.sh \
        -l ~/.rl_exp/my_exp \
        -j 1

For remote training, say training on nodes xxx.xxx.xxx.3 through xxx.xxx.xxx.8, do

    $ python scripts/run_es.py \
        redis_config/redis_master.conf \
        configurations/scrimmage.json \
        -s exp \
        --local-env-setup scripts/local_env_setup.sh \
        -l ~/.rl_exp/my_exp \
        --cluster_info xxx.xxx.xxx.3-xxx.xxx.xxx.8

To stop training running on nodes xxx.xxx.xxx.3 through xxx.xxx.xxx.8, add the ``-k`` flag to the above command

By default, ``run_es.py`` will save the git commit hash and diff of evolutionary-strategies in the log directory as ``config/hashes.txt`` and ``/config/evolutionary-strats.diff`` respectively. If you have other repos you would also like to track the hashes and git diffs of for a given experiment (scrimmage for example), you can use the `--git-dir` argument. For example, in the following:

    $ python scripts/run_es.py \
        redis_config/redis_master.conf \
        configurations/scrimmage.json \
        -s exp \
        --local-env-setup scripts/local_env_setup.sh \
        -l ~/.rl_exp/my_exp \
        --git-dir /path/to/scrimmage \
        --git-dir /path/to/example_repo

The current hashes of evolutionary strategies, scrimmage, and example_repo will be saved to `~/.rl_exp/my_exp/config/hashes.txt` and diffs of the repos will be saved under `~/.rl_exp/my_exp/config/scrimmage.diff` and `~/.rl_exp/my_exp/config/example_repo.diff`.

## Tips

``es_distributed.py`` forks itself and does not close down
when you execute ``tmux kill-session -t exp`` so you can
close things down with the following:

  ``pkill -9 -f redis``

On a cluster, it might look like this:

  ``for i in {1..3}; do ssh 192.168.90.$i "pkill -f -9 redis"; done``

Locally, it can be helpful to add ``--new_terminal`` and ``--nvim``
options.

Finally, on ubuntu 18.04 (not 16.04), it seems that extended
training results in unbounded RAM growth. To monitor the training
and restart it when RAM usage gets above some bound, add "--ram-thresh 0.5"
to restart jobs when there is less than 500Mb of RAM left.

## Customization

These are the things I know in the JSON file:

* **config -> eval_prob** - the probability the workers will do an evaluation experiment instead of a training experiment. It just lets you know how the current parameters are doing for a single experiment.
* **config -> num\_models\_to\_save** - The number of last checkpoints of the model to keep
* **env** - This is the information that we need to give to openai to create a scrimmage gym environment. This stuff can be changed as needed
 * **env -> global\_sensor** - Use just one agent's sensor information for all agents
 * **env -> static\_obs\_space** - Setting this to false will tell scrimmage to fill in state information for missing entities
 * **env -> combine_actors** - Treat separate agents in scrimmage as a single large agent (combine actions, etc.)
* **exp_prefix** - Just leave this as scrimmmage because I've hard coded this to work for now
* optimizer - We can change the optimizer a little bit here. I think this goes into tf_utils.py and creates an optimizer from there. It can probably be left to Adam
* **policy -> type** - This creates a neural network of the type LSTMPolicy. Leave it as is for now as we don't have to use LSTMs in this policy and we don't have any other policy types
* **policy -> args** - Here we can specify the network shape and type.
* **policy -> args -> keep_prob** - The probability to keep layers for dropout
* **policy -> args -> architecture** - List of layers defining the neural network. The layer types can be LSTM or fully-connected. The activation function for each layer can also be specified.

# OLD README BELOW

# Distributed evolution

This is a distributed implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation here uses a master-worker architecture: at each iteration, the master broadcasts parameters to the workers, and the workers send returns back to the master. The humanoid scaling experiment in the paper was generated with an implementation similar to this one.

The code here runs on EC2, so you need an AWS account. It's resilient to worker termination, so it's safe to run the workers on spot instances.

## Instructions

### Build AMI
The humanoid experiment depends on Mujoco. Provide your own Mujoco license and binary in `scripts/dependency.sh`.

Install [Packer](https://www.packer.io/), and then build images by running (you can optionally configure `scripts/packer.json` to choose build instance or AWS regions)
```
cd scripts && packer build packer.json
```

Packer should return you a list of AMI ids, which you should place in `AMI_MAP` in `scripts/launch.py`.

### Launching
Use `scripts/launch.py` along with an experiment JSON file. An example JSON file is provided in the `configurations` directory. You must fill in all command-line arguments to `scripts/launch.py`.
