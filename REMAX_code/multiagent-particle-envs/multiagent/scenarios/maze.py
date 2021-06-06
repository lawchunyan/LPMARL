import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0#2
        num_agents = 1
        # num_target = 1
        num_landmarks = 35#10
        # num_total = num_agents+num_target
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05#0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if i==0:
                landmark.collide = False # pass
                landmark.size = 0.10
            else:
                landmark.collide = True  # not pass
                landmark.size = 0.15

            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i==0:
                landmark.color = np.array([0.85, 0.35, 0.35])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states

        for agent in world.agents:
            # if k==0:
            #     agent.state.p_pos = np.array([0.2,0.2])#np.random.uniform(0, 0, world.dim_p)
            # else:
            #     agent.state.p_pos = np.array([-0.2,-0.2])
            # k+=1
            agent.state.p_pos = np.array([0.5,-0.15])#np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        k=0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.landmarks[0].state.p_pos = np.array([-0.5,0.15])
        k+=1

        """ wall"""
        ## in map

        for ii in range(3):
            world.landmarks[k].state.p_pos = np.array([-0.25,0.5-ii*0.25])
            k+=1

        for ii in range(3):
            world.landmarks[k].state.p_pos = np.array([0.25,-0.5+ii*0.25])
            k+=1

        #out map
        for ii in range(7):
            world.landmarks[k].state.p_pos = np.array([-0.75,-0.75+ii*0.25])
            k+=1

            world.landmarks[k].state.p_pos = np.array([+0.75,+0.75-ii*0.25])
            k+=1

            world.landmarks[k].state.p_pos = np.array([-0.75+ii*0.25,-0.75])
            k+=1

            world.landmarks[k].state.p_pos = np.array([+0.75-ii*0.25,+0.75])
            k+=1





        """"""



    def set_world(self, world,obs_n):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i==0:
                landmark.color = np.array([0.85, 0.35, 0.35])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        # k=0
        for i,agent in enumerate(world.agents,0):
            # if k==0:
            #     agent.state.p_pos = np.array([0.2,0.2])#np.random.uniform(-1, +1, world.dim_p)
            # else:
            #     agent.state.p_pos = np.array([-0.2,-0.2])
            # k+=1
            agent.state.p_pos = np.squeeze(obs_n[i*2:(i+1)*2])
            agent.state.p_vel = np.squeeze(obs_n[(1*2+i*2):(1*2+(i+1)*2)])#np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        k=0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)


        world.landmarks[0].state.p_pos = np.array([-0.5,0.15])
        k+=1

        """ wall"""
        ## in map

        for ii in range(3):
            world.landmarks[k].state.p_pos = np.array([-0.25,0.5-ii*0.25])
            k+=1

        for ii in range(3):
            world.landmarks[k].state.p_pos = np.array([0.25,-0.5+ii*0.25])
            k+=1

        #out map
        for ii in range(7):
            world.landmarks[k].state.p_pos = np.array([-0.75,-0.75+ii*0.25])
            k+=1

            world.landmarks[k].state.p_pos = np.array([+0.75,+0.75-ii*0.25])
            k+=1

            world.landmarks[k].state.p_pos = np.array([-0.75+ii*0.25,-0.75])
            k+=1

            world.landmarks[k].state.p_pos = np.array([+0.75-ii*0.25,+0.75])
            k+=1

        # world.landmarks[0].state.p_pos = np.array([-0.5,-0.4])
        # k+=1
        #
        # """ wall"""
        # ## in map
        #
        # for ii in range(3):
        #     world.landmarks[k].state.p_pos = np.array([-0.5,1.0-ii*0.5])
        #     k+=1
        #
        # for ii in range(3):
        #     world.landmarks[k].state.p_pos = np.array([0.5,-1.0+ii*0.5])
        #     k+=1
        #
        # #out map
        # for ii in range(7):
        #     world.landmarks[k].state.p_pos = np.array([-1.5,-1.5+ii*0.5])
        #     k+=1
        #
        #     world.landmarks[k].state.p_pos = np.array([+1.5,+1.5-ii*0.5])
        #     k+=1
        #
        #     world.landmarks[k].state.p_pos = np.array([-1.5+ii*0.5,-1.5])
        #     k+=1
        #
        #     world.landmarks[k].state.p_pos = np.array([+1.5-ii*0.5,+1.5])
        #     k+=1

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
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        #
        rew = 0#0
        a = agent

        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[0].state.p_pos)))]
        if min(dists)<0.1:
            rew += 1
        # rew -= min(dists)
        # if agent.collide:
        #     for wa in world.agents:
        #         # if wa is agent: #for training needed (collide)
        #         #     continue
        #         # else:
        #         if self.is_collision(wa, agent):
        #             rew -= 1#10
        def bound(x):
            if x < 0.75:
                return 0
            return 1
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def observation(self, agent, world):

        # comp_pos = []
        # for comp in range(2):
        #     comp_pos.append(np.array([0,0]))
        #
        # comp_vel = []
        # for comp in range(2):
        #     comp_vel.append(np.array([0,0]))

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        entity_pos.append(world.landmarks[0].state.p_pos)

        pos = []
        vel = []
        for other in world.agents:
            # if other is agent: continue
            # comm.append(other.state.c)
            pos.append(other.state.p_pos)
            # if not other.adversary:
            vel.append(other.state.p_vel)
        # print(comm)
        return np.concatenate(pos + vel)# + entity_pos)
