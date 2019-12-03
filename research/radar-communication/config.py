#  System parameters for Radar communication in Autonomous Vehicle

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
    'occur_with_bad_road': 0.1,
    'occur_with_good_road': 0.01,
    'occur_with_bad_weather': 0.1,
    'occur_with_good_weather': 0.01,
    'occur_with_fast_speed': 0.1,
    'occur_with_slow_speed': 0.01,
    'occur_with_moving_object': 0.1,
    'occur_with_static_object': 0.01,
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
    'test_id': 11,
    'nb_steps': 1000000,
    'target_model_update': 1e-3,
}


