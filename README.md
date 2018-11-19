
# Evolutionary Strategies Fork

This a fork of the openai [evolutionary-strategies](https://github.com/openai/evolution-strategies-starter) git repo. It currently works on one agent!
Please follow the installations and usage instructions below to see if it works for you as well. If you want to change the default setups of things, please look at the customization section.

## Todos:

The following work still needs to be done:

* Figure out how to setup redis to work on the cluster - whether or not it opens ports, etc.
* Figure out what neural network structure works best with evolutionary strategies
* Use tensorflow APIs so we don't define our own linear layer, flatten, normalized columns, etc.
* Update the running script to accept arguments like number of workers, where to connect for redis, save location
* Figure out how the network gets saved and loaded (look in policies.py)
* Reduce the code copied from A3C and try to import it from A3C instead? (import models from model.py if possible, etc.)
* Double check multiple agents work.
* Turn this into a python package that can be installed?

## Installation Instructions
First we will need to install redis (version 3.2 or greater). Sadly the apt-get package is not a new enough version so we will need to install from source. Here are the instructions I used from `scripts/dependencies.sh`

	$ wget --quiet http://download.redis.io/releases/redis-3.2.7.tar.gz -O redis-3.2.7.tar.gz
	$ tar -xvzf redis-3.2.7.tar.gz
	$ cd redis-3.2.7
	$ make
	$ sudo make install
	$ sudo mkdir -p /etc/redis
	$ sudo cp redis.conf /etc/redis # Setup default configuration file
	$ cd ..
	$ rm -rf redis-3.2.7 redis-3.2.7.tar.gz # Remove the installation files

Now we need to setup the pip packages. I recommend using virtualenv to create a new python environment to install these packages in. Here is a quick how-to on setting up a new virtualenv environment "evol-strats" in the `~/venvs/` folder using python2.

	$ sudo apt-get install virtualenv -y
	$ virtualenv ~/venvs/evol-strats -p $(which python2)
	$ source ~/venvs/evol-strats/bin/activate # Enter virtual environment
	(evol-strats) $ pip install click \
			h5py  \
			tensorflow==1.3.0 \
			gym==0.10.8 \
			grpcio==1.2.1 \
			protobuf==3.3.0 \
			redis
	(evol-strats) $ cd /path/to/scrimmage/python

The following command will need sudo if you installed the scrimmage python bindings for your system python to overwrite it in the virtualenv environment folder. It won't overwrite your system's scrimmage python binding. You might also need to run `scrimmage/setup/install-binaries.sh` with python 2 dependencies. More info on how to do that is located [here](https://github.com/gtri/scrimmage#install-binary-dependencies)

	(evol-strats) $ pip install -e .

To get out of the virtualenv environment, simply type `deactivate`.

## Usage

### neovim users

The following will start a new terminal with neovim running inside a tmux session.

    $ python scripts/run_nvim.py \
        redis_config/redis_master.conf \
        configurations/action.json \
        -s exp \
        -l scripts/local_env_setup.sh \
        -j 1

### non-neovim users

This needs more work to make it more automated but currently these are the steps to get it running.

In one terminal, run the following:

	$ redis-server /path/to/evolution-strategies-agent/redis_config/redis_master.conf

This starts up the redis server connecting on a socket file (I think that is all.).

Next, we will run evolutionary strategies using:
	
	$ cd /path/to/evolutionary-strategies-agent
	$ ./scripts/local_run_exp.sh configurations/scrimmage.json

The JSON file specifies the setup of the problem. This will open up a tmux with two windows. One will be the master client, the other will be the workers. If it is working correctly, the worker clients should have a relay message about the average batch size every 5 seconds or so. There will also occasionally be an evaluation result message. I think every 10 minutes or so, the master client will output some data about the average training rewards and evaluation rewards.

## Customization

These are the things I know in the JSON file:

* **config -> eval_prob** - the probability the workers will do an evaluation experiment instead of a training experiment. It just lets you know how the current parameters are doing for a single experiment.
* **env** - This is the information that we need to give to openai to create a scrimmage gym environment. This stuff can be changed as needed
* **exp_prefix** - Just leave this as scrimmmage because I've hard coded this to work for now
* optimizer - We can change the optimizer a little bit here. I think this goes into tf_utils.py and creates an optimizer from there. It can probably be left to Adam
* **policy -> type** - This creates a neural network the way done in A3C. Leave it as is for now
* **policy -> args** - Here we can specify the network shape and type. 



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
