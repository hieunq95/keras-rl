#  System parameters for Radar communication in Autonomous Vehicle
from __future__ import division

UNEXPECTED_EV_PROB = 0.05
transition_probability = {
    'channel_sw_bad_to_bad': 0.1,
    'channel_sw_good_to_good': 0.9,
    'road_sw_bad_to_bad': 0.1,
    'road_sw_good_to_good': 0.9,
    'weather_sw_bad_to_bad': 0.1,
    'weather_sw_good_to_good': 0.9,
    'speed_sw_fast_to_fast': 0.1,
    'speed_sw_slow_to_slow': 0.9,
    'object_sw_moving_to_moving': 0.1,
    'object_sw_static_to_static': 0.9,
    'arrival_mean': 1,
}

unexpected_ev_prob = {
    'occur_with_bad_road': UNEXPECTED_EV_PROB,
    'occur_with_good_road': UNEXPECTED_EV_PROB / 10,
    'occur_with_bad_weather': UNEXPECTED_EV_PROB,
    'occur_with_good_weather': UNEXPECTED_EV_PROB / 10,
    'occur_with_fast_speed': UNEXPECTED_EV_PROB * 3,
    'occur_with_slow_speed': 0,
    'occur_with_moving_object': UNEXPECTED_EV_PROB,
    'occur_with_static_object': UNEXPECTED_EV_PROB / 10,
}

state_space_size = {
    'data_size': 11,
    'channel_size': 2,
    'road_size': 2,
    'weather_size': 2,
    'speed_size': 2,
    'object_size': 2,
}

action_space_size = {
    'action_size': 2,
}

#  Parameters for testing DQN agent
test_parameters = {
    'test_id': 16,
    'nb_steps': 1000000,
    'target_model_update': 1e-3,
}


