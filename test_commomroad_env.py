"""
Unit tests of the module gym_commonroad.commonroad_env
"""
import os
import gym
import pytest
import numpy as np

from commonroad_rl.tests.common.marker import *
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
from commonroad_rl.tests.common.path import resource_root

__author__ = "Mingyang Wang"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Mingyang Wang"
__email__ = "mingyang.wang@tum.de"
__status__ = "Integration"

env_id = "commonroad-v0"
resource_path = resource_root("test_commonroad_env")
meta_scenario_path = os.path.join(resource_path, "pickles", "meta_scenario")
problem_path = os.path.join(resource_path, "pickles", "problem")


@pytest.mark.parametrize(
    ("strict_off_road_check", "action", "expected_is_off_road_list"),
    [
        (True, np.array([0.0, 0.0]), [0] * 20),
        (True, np.array([0.0, 0.2]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]),
        (False, np.array([0.0, 0.2]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        (True, np.array([0.0, 0.1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]),
        (False, np.array([0.0, 0.1]), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    ],
)
@unit_test
@functional
def test_check_off_road(strict_off_road_check, action, expected_is_off_road_list):

    env = CommonroadEnv(
        meta_scenario_path=meta_scenario_path,
        train_reset_config_path=problem_path,
        test_reset_config_path=problem_path,
        strict_off_road_check=strict_off_road_check,
    )
    env.reset()

    result_list = []

    i = 1
    # while not done:
    for i in range(20):
        obs, reward, done, info = env.step(action)
        result_list.append(info["is_off_road"])
        env.render()
        # ego vehicle has already off road, break the loop
        if i > 2 and result_list[-1] == 0 and result_list[-2] == 1:
            break
    assert result_list == expected_is_off_road_list
