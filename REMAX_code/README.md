

This is the code for implementing the REMAX algorithm.
The code is modified from https://github.com/openai/maddpg

For Multi-Agent Particle Environments (MPE) installation, please refer to https://github.com/openai/multiagent-particle-envs

Replace the file 'environment.py' in 'multiagent-particle-envs' - 'multiagent'

Add the file 'maze.py' in 'scenarios' in 'multiagent-particle-envs' - 'multiagent' - 'scenarios'

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python3 train.py --scenario maze``

- You can replace `maze` with any environment in the MPE you'd like to run.

### Command-line options

#### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"maze"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `50`)

- `--num-episodes` total number of training episodes (default: `50000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--VAE-latent-dim`: Latent space dim. in the VAE (default: `1`)
