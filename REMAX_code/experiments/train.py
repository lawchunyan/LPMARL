import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer, VAETrainer
import tensorflow.contrib.layers as layers
tfd = tf.contrib.distributions

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="maze", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes") #60000
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents") #maddpg #mmmaddpg
    parser.add_argument("--bad-policy", type=str, default="maddpg", help="policy of adversaries") #maddpg
    parser.add_argument("--VAE", type=str, default="remax", help="REMAX")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--vae-batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    parser.add_argument("--VAE-latent-dim", type=int, default=1, help="VAE's latent space dimension")
    parser.add_argument("--generated-state-num", type=int, default=400, help="number of generated states")
    parser.add_argument("--VAE-lr", type=float, default=1e-4, help="VAE's learning rate")
    parser.add_argument("--KDE-bandwidth", type=float, default=0.05, help="bandwidth for KDE")
    parser.add_argument("--VAE-epoch", type=int, default=3, help="VAE's epoch")
    parser.add_argument("--gen-state-prob", type=float, default=0.8, help="gen_state_prob")
    parser.add_argument("--gradient-reg", type=float, default=0, help="reg for gradient of surrogate model")
    parser.add_argument("--surrogate-lambda", type=float, default=1e-3, help="lambda for property of surrogate model")
    parser.add_argument("--surrogate-noise", type=float, default=1, help="noise for Z of surrogate model")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='maze', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")#./tmp/policy/model-100
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def mu_sig_de_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=int(num_units/2), activation_fn=tf.nn.relu)
        loc = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        scale = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.softplus)
        return loc, scale

def mu_sig_en_model(input, num_outputs, scope, reuse=False, num_units=64, num_agents=4, num_landmarks=0, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        inputs = input #bat x (2N+2N)
        num_agents_shape = int(inputs.shape.as_list()[1]/4) # N
        inputs_t = tf.transpose(inputs,(1,0)) #(2N+2N) x bat
        pos_inputs_t = inputs_t[:2*num_agents_shape] #2N x bat # pos
        vel_inputs_t = inputs_t[2*num_agents_shape:] #2N x bat # vel

        pos_inputs_t_split = tf.reshape(pos_inputs_t, (2, num_agents_shape, -1)) #2 x N x bat # pos
        vel_inputs_t_split = tf.reshape(vel_inputs_t, (2, num_agents_shape, -1)) #2 x N x bat # vel

        pos_vel_inputs_t_split = tf.concat([pos_inputs_t_split,vel_inputs_t_split],axis=0) #4 x N x bat #pos+vel

        pos_vel_inputs_split = tf.transpose(pos_vel_inputs_t_split, (2,1,0)) # bat x N x 4 #pos+vel

        num_multi_head = 2
        multi_head = []
        for k in range(num_multi_head):
            Wh = tf.layers.dense(pos_vel_inputs_split, 16) # bat x N x 16
            Wh_t = tf.transpose(Wh, (1,0,2)) # N x bat x 16

            new_h_list = []
            for i in range(num_agents_shape):
                self_Wh = tf.tile(Wh_t[i:i+1], [num_agents_shape, 1, 1]) # N x bat x 16
                concat_Wh = tf.concat([self_Wh, Wh_t], axis = 2) # N x bat x 32
                concat_Wh_t = tf.transpose(concat_Wh, (1,0,2)) # bat x N x 32

                logit = tf.layers.dense(concat_Wh_t, 1, tf.nn.leaky_relu) # bat x N x 1
                attention_coeff = tf.nn.softmax(logit, axis = 1) # bat x N x 1

                new_h_list.append(tf.reduce_sum(Wh * attention_coeff, axis = 1, keepdims=True)) # bat x 1 x 16

            new_h = tf.concat(new_h_list, axis = 1) #bat x N x 16
            # print('new_h:',new_h.shape)
            multi_head.append(new_h)
        multi_head = tf.concat(multi_head, axis = 2)
        # print('multi_head:',multi_head.shape) # bat x N x 16*num_multi_head

        new_h_vector = tf.reshape(multi_head, (-1, num_agents_shape*16*num_multi_head))#bat x 16N*num_multi_head
        new_h_vector = tf.nn.relu(new_h_vector)


        loc = tf.layers.dense(new_h_vector, num_outputs)#bat x Nxnum_outputs
        scale = tf.layers.dense(new_h_vector, num_outputs, tf.nn.softplus)#bat x Nxnum_outputs
        return loc, scale, attention_coeff

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.set_world)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    vae_trainer = []
    model = mlp_model
    VAE_en_model = mu_sig_en_model
    VAE_de_model = mu_sig_de_model
    trainer = MADDPGAgentTrainer
    vaetrainer = VAETrainer
    for i in range(num_adversaries):
        print("{} bad agents".format(i))
        policy_name = arglist.bad_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print("{} good agents".format(i))
        policy_name = arglist.good_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    if arglist.VAE =='remax':
        policy_name = "VAE"
        vae_trainer = vaetrainer(
            "VAE", VAE_en_model, VAE_de_model, model, obs_shape_n, env.action_space, 0, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg')
    return trainers, vae_trainer


def train(arglist):
    if arglist.test:
        np.random.seed(71)
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers, vae_trainer = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and bad policy {} with {} adversaries'.format(arglist.good_policy, arglist.bad_policy, num_adversaries))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
            if arglist.load_name == "":
                # load seperately
                bad_var_list = []
                for i in range(num_adversaries):
                    bad_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(bad_var_list)
                U.load_state(arglist.load_bad, saver)

                good_var_list = []
                for i in range(num_adversaries, env.n):
                    good_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(good_var_list)
                U.load_state(arglist.load_good, saver)
            else:
                print('Loading previous state from {}'.format(arglist.load_name))
                U.load_state(arglist.load_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_rewards_for_test = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_rewards_for_test = []
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        episode_step_for_test = 0
        train_step = 0
        t_start = time.time()
        if arglist.render:
            time.sleep(0.1)
            env.render()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            if arglist.render:
                time.sleep(0.1)
                env.render()
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            # done = all(done_n)
            # done = all(done_n)
            # if np.sum(rew_n)>=env.n:#0:
            #     done = True
            # else:
            done = False
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done, terminal)
            if arglist.VAE =='remax':
                vae_trainer.B0UB1_experience(obs_n, action_n, rew_n, new_obs_n, done, terminal)


            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:

                ### TESTING ###
                obs_n = env.reset()
                while True:
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    episode_step_for_test += 1

                    # if np.sum(rew_n)>=env.n:#0:
                    #     done_for_test = True
                    # else:
                    done_for_test = False
                    terminal_for_test = (episode_step_for_test >= arglist.max_episode_len)
                    obs_n = new_obs_n

                    episode_rewards_for_test[-1] += rew_n[0]
                    if done_for_test or terminal_for_test:
                        episode_rewards_for_test.append(0)
                        episode_step_for_test = 0
                        break


                ###
                if arglist.VAE =='remax':
                    # vae_trainer.rewarding_buffer.clear()

                    if len(vae_trainer.generated_buffer) == 0:
                        obs_n = env.reset()
                    else:
                        if np.random.rand()<arglist.gen_state_prob:
                            generated_sample_index = vae_trainer.generated_buffer.make_index(1)
                            obs, _, _, _, _ = vae_trainer.generated_buffer.sample_index_and_del(generated_sample_index)
                            # print("obs",obs)
                            obs_n = env.set(obs[0])

                        else:
                            obs_n = env.reset()
                else:
                    obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1
            # print(train_step)

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            # if arglist.display:
            #     time.sleep(0.1)
            #     env.render()
            #     continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.test:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                if arglist.VAE =='remax':
                    vae_trainer.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)
                if arglist.VAE =='remax':
                    vae_trainer.update(trainers, len(episode_rewards)-1, episode_step)
            # save model, display training output
            if (terminal or done) and (len(episode_rewards) % arglist.save_rate == 0):
                if not arglist.test:
                    U.save_state(arglist.save_dir, global_step = len(episode_rewards), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate-1:-1]), round(time.time()-t_start, 3)))
                    print("mean episode reward TEST: {}".format(np.mean(episode_rewards_for_test[-arglist.save_rate-1:-1])))
                else:
                    print("{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(arglist.bad_policy, arglist.good_policy,
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate-1:-1]),
                        [np.mean(rew[-arglist.save_rate-1:-1]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                    print("mean episode reward TEST: {}".format(np.mean(episode_rewards_for_test[-arglist.save_rate-1:-1])))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate-1:-1]))
                final_ep_rewards_for_test.append(np.mean(episode_rewards_for_test[-arglist.save_rate-1:-1]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate-1:-1]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                suffix = '_test.pkl' if arglist.test else '.pkl'
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards' + suffix
                rew_file_name_for_test = arglist.plots_dir + arglist.exp_name + '_rewards_for_test' + suffix
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards' + suffix

                if not os.path.exists(os.path.dirname(rew_file_name)):
                    try:
                        os.makedirs(os.path.dirname(rew_file_name))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise

                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                with open(rew_file_name_for_test, 'wb') as fp:
                    pickle.dump(final_ep_rewards_for_test, fp)
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
