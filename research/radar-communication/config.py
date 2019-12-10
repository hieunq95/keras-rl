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
    'occur_with_fast_speed': UNEXPECTED_EV_PROB * 20,  # variable 1
    'occur_with_slow_speed': UNEXPECTED_EV_PROB / 10,
    'occur_with_moving_object': UNEXPECTED_EV_PROB,  # variable 2
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
    'test_id': 35,
    'nb_steps': 1000000,
    'nb_epsilon_linear': 600000,
    'target_model_update': 1e-3,
}

"""
'test_id': 15,
'occur_with_fast_speed': UNEXPECTED_EV_PROB,
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 24,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 2,
--------------------------
'test_id': 16,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 4,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 4,
--------------------------
'test_id': 25,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 8,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 8,
--------------------------
'test_id': 17,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 16,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 16,
--------------------------
--------------------------
'test_id': 26,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2, # 0.1
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 27,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 4, # 0.2
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 28,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 6, # 0.3
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 29,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 8, # 0.4
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 30,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 10, # 0.5
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 31,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 12, # 0.6
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 32,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 14, # 0.7
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 33,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 16, # 0.8
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 34,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 18, # 0.9
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 35,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 20, # 1.0
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------

"""


