import numpy as np
import math
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.environment import MultiAgentEnv


class Scenario(BaseScenario):
    def make_world(self, num_agents=10, num_landmarks=10):
        assert num_agents == num_landmarks

        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents
        num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states

        theta = 2 * math.pi / len(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-0.9, 0.9, 2)
            # agent.state.p_pos = np.array([math.cos(theta * i) * 0.3, math.sin(theta * i) * 0.3])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([math.cos(theta * i) * 2, math.sin(theta * i) * 2])
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.num_hit = 0
        world.t = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 10
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on distance to each landmark, penalized for collisions
        rew = 0
        min_dist = 5000

        # dist rwd
        for l in world.landmarks:
            # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            # rew -= min(dists)
            min_dist = min(min_dist, np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))))

        rew -= min_dist * 0.1

        if min_dist < 0.5:
            rew += 10
            world.num_hit += 1

        # rwd for collision
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a != agent:
                    rew -= 10

        # rwd for boundary
        def bound(x):
            if x < 1.8:
                return 0
            if x < 2.0:
                return (x - 1.8) * 10
            return min((x - 2) ** 2, 10)

        for x in agent.state.p_pos:
            rew -= bound(abs(x))

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        world.t += 1

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def done(self, agent, world):
        if world.t >= 50:
            return True
        else:
            return False


def make_env(n_ag, n_en):
    # load scenario from script
    scenario = Scenario()

    # create world
    world = scenario.make_world(n_ag, n_en)

    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
    env.shared_reward = False

    return env


def get_landmark_state(env: MultiAgentEnv):
    out_state = []
    for landmark in env.world.landmarks:
        out_state.append(landmark.state.p_pos)

    return np.array(out_state)

# def vis_state(env):
#     fig = plt.figure(figsize=(5,5))
#     ax = fig.add_subplot(111)
#
#     # msking rectangle
#     ax.set_xlim([-3, 3])
#     ax.set_ylim([-3, 3])
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     # plotting agent and landmarks
#     for entity in env.world.entities:
#         color = entity.color
#         size = entity.size
#         center = entity.state.p_pos
#         draw_circle = plt.Circle(center, size, color=color)
#         ax.add_artist(draw_circle)
#     plt.show()
#     clear_output(wait=0.1)
