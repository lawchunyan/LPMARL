import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer
from sklearn.neighbors import KernelDensity
import copy
import time
tfd = tf.contrib.distributions



def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        if adversarial:
            num_agents = len(act_input_n)
            if p_index < num_adversaries:
                adv_rate = [adv_eps_s *(i < num_adversaries) + adv_eps * (i >= num_adversaries) for i in range(num_agents)]
            else:
                adv_rate = [adv_eps_s *(i >= num_adversaries) + adv_eps * (i < num_adversaries) for i in range(num_agents)]
            print("      adv rate for p_index : ", p_index, adv_rate)
            raw_perturb = tf.gradients(pg_loss, act_input_n)
            perturb = [tf.stop_gradient(tf.nn.l2_normalize(elem, axis = 1)) for elem in raw_perturb]
            perturb = [perturb[i] * adv_rate[i] for i in range(num_agents)]
            new_act_n = [perturb[i] + act_input_n[i] if i != p_index
                    else act_input_n[i] for i in range(len(act_input_n))]

            adv_q_input = tf.concat(obs_ph_n + new_act_n, 1)
            adv_q = q_func(adv_q_input, 1, scope = "q_func", reuse=True, num_units=num_units)[:,0]
            pg_loss = -tf.reduce_mean(adv_q)#(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]

        if adversarial:
            num_agents = len(act_ph_n)
            if q_index < num_adversaries:
                adv_rate = [adv_eps_s *(i < num_adversaries) + adv_eps * (i >= num_adversaries) for i in range(num_agents)]
            else:
                adv_rate = [adv_eps_s *(i >= num_adversaries) + adv_eps * (i < num_adversaries) for i in range(num_agents)]
            print("      adv rate for q_index : ", q_index, adv_rate)

            pg_loss = -tf.reduce_mean(target_q)
            raw_perturb = tf.gradients(pg_loss, act_ph_n)
            perturb = [adv_eps * tf.stop_gradient(tf.nn.l2_normalize(elem, axis = 1)) for elem in raw_perturb]
            new_act_n = [perturb[i] + act_ph_n[i] if i != q_index
                    else act_ph_n[i] for i in range(len(act_ph_n))]
            adv_q_input = tf.concat(obs_ph_n + new_act_n, 1)
            target_q = q_func(adv_q_input, 1, scope ='target_q_func', reuse=True, num_units=num_units)[:,0]

        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

def vae_train(make_obs_ph_n, act_space_n, vae_index, vae_func_en, vae_func_de, vae_surrogate_func, optimizer, adversarial, adv_eps, \
                adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, \
                num_units=64, VAE_latent_dim=3, gradient_reg=0):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        surrogate_target_ph = tf.placeholder(tf.float32, [None], name="surrogate_target")

        encoder_input = obs_ph_n[vae_index]
        # print(vae_func(encoder_input, VAE_latent_dim, scope="vae_encoder_func", reuse = tf.AUTO_REUSE, num_units=num_units)[0])
        encoder_loc = vae_func_en(encoder_input, VAE_latent_dim, scope="vae_encoder_func", reuse = tf.AUTO_REUSE, num_units=num_units)[0]
        encoder_scale = vae_func_en(encoder_input, VAE_latent_dim, scope="vae_encoder_func", reuse = tf.AUTO_REUSE, num_units=num_units)[1]
        vae_encoder_vars = U.scope_vars(U.absolute_scope_name("vae_encoder_func"))

        posterior = tfd.MultivariateNormalDiag(encoder_loc, encoder_scale) #encoder out
        encoding = posterior.sample()

        prior_loc = tf.zeros(VAE_latent_dim)
        prior_scale = tf.ones(VAE_latent_dim)
        prior = tfd.MultivariateNormalDiag(prior_loc, prior_scale, name='prior')

        zt = tf.placeholder(tf.float32,shape=[None, VAE_latent_dim],name="zt")

        decoder_input_zt = zt

        decoder_loc_zt = vae_func_de(decoder_input_zt, int(obs_ph_n[vae_index][0].shape[0]), scope="vae_decoder_func",\
                            reuse = tf.AUTO_REUSE, num_units=num_units)[0]
        decoder_scale_zt = vae_func_de(decoder_input_zt, int(obs_ph_n[vae_index][0].shape[0]), scope="vae_decoder_func",\
                            reuse = tf.AUTO_REUSE, num_units=num_units)[1]

        decoder_out_zt = tfd.Independent(tfd.Normal(decoder_loc_zt, decoder_scale_zt), 1)
        # print("decoder_out_zt",decoder_out_zt.mean())

        decoder_input = encoding

        surrogate = vae_surrogate_func(encoding, 1, scope="surrogate_func", num_units=num_units)[:,0]
        surrogate_loss = tf.reduce_mean(tf.square(surrogate - surrogate_target_ph))

        surr_zt = tf.placeholder(tf.float32, shape=[None, VAE_latent_dim],name="surr_zt")
        surrogate_zt = vae_surrogate_func(surr_zt, 1, scope="surrogate_func", reuse = tf.AUTO_REUSE, num_units=num_units)[:,0]
        surrogate_func_vars = U.scope_vars(U.absolute_scope_name("surrogate_func"))

        surrogate_gradient = tf.gradients(surrogate_zt-gradient_reg*tf.norm(surr_zt),surr_zt)#1e-2 converge same points

        decoder_loc = vae_func_de(decoder_input, int(obs_ph_n[vae_index][0].shape[0]), scope="vae_decoder_func",\
                            reuse = tf.AUTO_REUSE, num_units=num_units)[0]
        decoder_scale = vae_func_de(decoder_input, int(obs_ph_n[vae_index][0].shape[0]), scope="vae_decoder_func",\
                            reuse = tf.AUTO_REUSE, num_units=num_units)[1]

        vae_decoder_vars = U.scope_vars(U.absolute_scope_name("vae_decoder_func"))

        decoder_out = tfd.Independent(tfd.Normal(decoder_loc, decoder_scale), 1)

        likelihood = decoder_out.log_prob(encoder_input)
        divergence = tfd.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        loss = -elbo + surrogate_loss

        vae_func_vars = vae_encoder_vars + vae_decoder_vars + surrogate_func_vars

        optimize_expr = U.minimize_and_clip(optimizer, loss, vae_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=[obs_ph_n[vae_index]] + [surrogate_target_ph], outputs=loss, updates=[optimize_expr])
        encode = U.function(inputs=[obs_ph_n[vae_index]], outputs=encoding)
        decode = U.function(inputs=[zt], outputs=decoder_out_zt.mean())
        z_gradient = U.function(inputs=[surr_zt], outputs=surrogate_gradient)
        elbo_print = U.function(inputs=[obs_ph_n[vae_index]], outputs=-elbo)
        surrogate_loss_print = U.function(inputs=[obs_ph_n[vae_index]] + [surrogate_target_ph], outputs=surrogate_loss)

        return train, encode, decode, prior, z_gradient, {'elbo_loss': elbo_print, 'surrogate_loss': surrogate_loss_print}

class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func, policy_name, adversarial):
        self.name = name
        self.scope = self.name + "_" + policy_name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial = adversarial,
            adv_eps = args.adv_eps,
            adv_eps_s = args.adv_eps_s,
            num_adversaries = args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial = adversarial,
            adv_eps = args.adv_eps,
            adv_eps_s = args.adv_eps_s,
            num_adversaries = args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.generated_state_num * args.max_episode_len#args.batch_size * args.max_episode_len#args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.policy_name = policy_name
        self.adversarial = adversarial
        self.act_space_n = act_space_n
        self.local_q_func = local_q_func

    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
                'policy_name': self.policy_name, 'adversarial': self.adversarial,
                'local_q_func':self.local_q_func,
                'adv_eps': self.args.adv_eps}

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]


class VAETrainer(AgentTrainer):
    def __init__(self, name, model_en, model_de, mlp_model, obs_shape_n, act_space_n, agent_index, args, local_q_func, policy_name, adversarial):
        self.name = name
        self.scope = self.name + "_" + policy_name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        if agent_index < args.num_adversaries:
            self.col_idx = [i for i in range(args.num_adversaries)]
            self.opp_idx = [i for i in range(args.num_adversaries,self.n)]
        else:
            self.col_idx = [i for i in range(args.num_adversaries,self.n)]
            self.opp_idx = [i for i in range(args.num_adversaries)]
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.vae_train, self.encode, self.decode, self.prior, self.z_gradient, self.vae_debug = vae_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            vae_index=agent_index,
            vae_func_en=model_en,
            vae_func_de=model_de,
            vae_surrogate_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.VAE_lr),
            adversarial = adversarial,
            adv_eps = args.adv_eps,
            adv_eps_s = args.adv_eps_s,
            num_adversaries = args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            VAE_latent_dim = args.VAE_latent_dim,
            gradient_reg = args.gradient_reg
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.B0UB1 = ReplayBuffer(1e6)
        self.generated_buffer = ReplayBuffer(args.generated_state_num)
        self.max_replay_buffer_len = args.generated_state_num * args.max_episode_len#args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.policy_name = policy_name
        self.adversarial = adversarial
        self.act_space_n = act_space_n
        self.local_q_func = local_q_func

    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
                'policy_name': self.policy_name, 'adversarial': self.adversarial,
                'local_q_func':self.local_q_func,
                'adv_eps': self.args.adv_eps}

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def B0UB1_experience(self, obs, act, rew, new_obs, done, terminal):
        self.B0UB1.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None
        self.B0UB1_sample_index = None

    def update(self, agents, t, t_step):
        
        if not t % self.args.generated_state_num == 0:  # only update every 100 steps
            return
        if not t_step == 0:
            return
        if t < self.args.generated_state_num:  # only update every 100 steps ###check
            return
        if len(self.B0UB1)<self.args.vae_batch_size:
            return

        self.generated_buffer.clear()
        U.get_session().run(tf.variables_initializer(var_list=tf.get_collection(\
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)))


        for i in range(self.args.VAE_epoch):
            for j in range(int(len(self.B0UB1)/self.args.vae_batch_size)):
                obs_n = []

                t_prev = time.time()

                self.B0UB1_sample_index = self.B0UB1.make_index(self.args.vae_batch_size)
                obs, act, rew, obs_next, done = self.B0UB1.sample_index(self.B0UB1_sample_index)
                obs_n = [np.transpose(obs, (1,0,2))[ii] for ii in range(self.n)]
                act_n = [np.transpose(act, (1,0,2))[ii] for ii in range(self.n)]
                rew_n = [np.transpose(rew, (1,0))[ii] for ii in range(self.n)]
                obs_next_n = [np.transpose(obs_next, (1,0,2))[ii] for ii in range(self.n)]
                done_n = done

                t_current = time.time()


                t_prev = time.time()

                target_act_next_n = [agents[ii].p_debug['target_act'](obs_next_n[ii]) for ii in range(self.n)]
                UCB = 0.0
                t_current = time.time()


                t_prev = time.time()

                target_q_next = [agents[jj].q_debug['target_q_values'](*(obs_next_n + target_act_next_n)) for jj in range(self.n)]
                target_q = [rew_n[jj] + self.args.gamma * (1.0 - done_n) * target_q_next[jj] for jj in range(self.n)]

                t_current = time.time()

                t_prev = time.time()

                q = [agents[jj].q_debug['q_values'](*(obs_n + act_n)) for jj in range(self.n)]

                t_current = time.time()

                t_prev = time.time()

                UCB = [self.args.surrogate_lambda*np.absolute(target_q[jj]-q[jj])+q[jj] for jj in self.col_idx]

                t_current = time.time()


                t_prev = time.time()

                UCB = np.mean(UCB,axis=0)

                surr_tar = UCB
                t_current = time.time()

                t_prev = time.time()

                loss = self.vae_train(*([obs_n[self.agent_index]]+[surr_tar]))
                t_current = time.time()

                t_prev = time.time()
                elbo_loss = self.vae_debug['elbo_loss'](*([obs_n[self.agent_index]]))
                t_current = time.time()

                t_prev = time.time()
                surrogate_loss = self.vae_debug['surrogate_loss'](*([obs_n[self.agent_index]]+[surr_tar]))
                t_current = time.time()

        num_sample = self.args.generated_state_num
        zt_init = tf.random.normal([num_sample, self.args.VAE_latent_dim]).eval()
        z_list = []

        for i in range(400):
            z_gradient = self.z_gradient(zt_init)
            zt_init = zt_init + 0.1*(z_gradient[0]+self.args.surrogate_noise*np.random.normal(loc=0.0, scale=1/(i+1), size=z_gradient[0].shape))

        for ns in range(num_sample):
            z_list.append(copy.deepcopy(zt_init[ns]))

        st = self.decode(z_list)

        for st_i in range(st.shape[0]):

            self.generated_buffer.add(st[st_i], None, None, None, None)

        self.B0UB1.clear()
        print('Done')

        return
