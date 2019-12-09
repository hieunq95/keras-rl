import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
from config import transition_probability, unexpected_ev_prob, state_space_size

class AV_Environment(gym.Env):
    def __init__(self):
        self.observation_space = spaces.MultiDiscrete(
            [state_space_size['data_size'], state_space_size['channel_size'], state_space_size['road_size'],
             state_space_size['weather_size'], state_space_size['speed_size'], state_space_size['object_size']])
        self.action_space = spaces.Discrete(2)
        self.episode_observation = {
            'step_counter': 0,
            'unexpected_ev_counter': 0,
        }
        self.seed(123)
        self.state = self.reset()
        print(self.state)

    def markov_transition(self, current_state, state_id):
        if current_state > 1 or current_state < 0:
            raise Exception('Invalid current_state')
        if state_id > 5 or state_id < 1:
            raise Exception('state_id should not exceed 5 or below 1')

        markov_probability = 0.0
        if current_state == 1:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_bad_to_bad']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_bad_to_bad']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_bad_to_bad']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_fast_to_fast']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_moving_to_moving']
            if self.nprandom.uniform() < markov_probability:
                current_state = 1
            else:
                current_state = 0
        else:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_good_to_good']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_good_to_good']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_good_to_good']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_slow_to_slow']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_static_to_static']
            if self.nprandom.uniform() < markov_probability:
                current_state = 0
            else:
                current_state = 1

        return current_state

    def state_transition(self, state, action):
        data_queue_state = state[0]
        channel_state = state[1]
        road_state = state[2]
        weather_state = state[3]
        speed_state = state[4]
        object_state = state[5]
        #  state transition
        if action == 0:
            transmitted_packets = 2 if channel_state == 1 else 4
        else:
            transmitted_packets = 0
        arrived_packets = self.nprandom.poisson(transition_probability['arrival_mean'])
        data_queue_state = data_queue_state - transmitted_packets + arrived_packets
        next_data_queue_state = min(max(0, data_queue_state), state_space_size['data_size'] - 1)
        # TODO transition for Markov chain
        next_channel_state = self.markov_transition(channel_state, 1)
        next_road_state = self.markov_transition(road_state, 2)
        next_weather_state = self.markov_transition(weather_state, 3)
        if object_state == 1:
            if action == 1:
                next_speed_state = 0
                next_object_state = self.markov_transition(object_state, 5)
            else:
                next_speed_state = self.markov_transition(speed_state, 4)
                next_object_state = self.markov_transition(object_state, 5)
        else:
            next_speed_state = self.markov_transition(speed_state, 4)
            next_object_state = self.markov_transition(object_state, 5)

        next_state = np.array([next_data_queue_state, next_channel_state, next_road_state,
                               next_weather_state, next_speed_state, next_object_state]).flatten()
        return next_state

    def risk_assessment(self, state):
        road_state = state[2]
        weather_state = state[3]
        speed_state = state[4]
        object_state = state[5]

        if road_state == 1:
            if self.nprandom.uniform() < unexpected_ev_prob['occur_with_bad_road']:
               unexpected_event = 1
            else:
               unexpected_event = 0
        else:
            if self.nprandom.uniform() < unexpected_ev_prob['occur_with_good_road']:
                unexpected_event = 1
            else:
                unexpected_event = 0
        if unexpected_event == 1:
            return 1

        if weather_state == 1:
           if self.nprandom.uniform() < unexpected_ev_prob['occur_with_bad_weather']:
               unexpected_event = 1
           else:
               unexpected_event = 0
        else:
            if self.nprandom.uniform() < unexpected_ev_prob['occur_with_good_weather']:
                unexpected_event = 1
            else:
                unexpected_event = 0
        if unexpected_event == 1:
            return 1

        if speed_state == 1:
           if self.nprandom.uniform() < unexpected_ev_prob['occur_with_fast_speed']:
               unexpected_event = 1
           else:
               unexpected_event = 0
        else:
            if self.nprandom.uniform() < unexpected_ev_prob['occur_with_slow_speed']:
                unexpected_event = 1
            else:
                unexpected_event = 0
        if unexpected_event == 1:
            return 1

        if object_state == 1:
           if self.nprandom.uniform() < unexpected_ev_prob['occur_with_moving_object']:
               unexpected_event = 1
           else:
               unexpected_event = 0
        else:
            if self.nprandom.uniform() < unexpected_ev_prob['occur_with_static_object']:
                unexpected_event = 1
            else:
                unexpected_event = 0
        if unexpected_event == 1:
            return 1

        return 0


    def get_reward(self, state, action):
        unexpected_ev_occurs = self.risk_assessment(state)
        channel_state = state[1]
        nb_bad_bits = state[2] + state[3] + state[4] + state[5]
        reward = 0
        if action == 0:
            if unexpected_ev_occurs == 0:
                if channel_state == 0:
                    reward += 2
                else:
                    reward += 1
            else:
                reward -= 50
        else:
            if unexpected_ev_occurs == 0:
                reward += 0
            else:
                reward += 5 * (nb_bad_bits + 1)
        if unexpected_ev_occurs == 1:
            self.episode_observation['unexpected_ev_counter'] += 1
        # print(nb_bad_bits)
        return reward

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = np.copy(self.state)
        next_state = self.state_transition(state, action)
        reward = self.get_reward(state, action)
        self.episode_observation['step_counter'] += 1
        if self.episode_observation['step_counter'] == 400:
            done = True
            # print('Unexpected events / Episode steps = {} / {}'
            #       .format(self.episode_observation['unexpected_ev_counter'], self.episode_observation['step_counter']))
        else:
            done = False
        self.state = next_state

        return next_state, reward, done, {}

    def reset(self):
        self.episode_observation['step_counter'] = 0
        self.episode_observation['unexpected_ev_counter'] = 0
        self.state = np.array([
            random.randint(0, state_space_size['data_size'] - 1),
            random.randint(0, state_space_size['channel_size'] - 1),
            random.randint(0, state_space_size['road_size'] - 1),
            random.randint(0, state_space_size['weather_size'] - 1),
            random.randint(0, state_space_size['speed_size'] - 1),
            random.randint(0, state_space_size['object_size'] - 1)
        ])
        return self.state
    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]
    def is_terminated(self):
        ...

