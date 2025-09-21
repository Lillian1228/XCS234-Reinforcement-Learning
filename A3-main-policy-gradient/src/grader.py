#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from itertools import product
from graderUtil import graded, CourseTestRunner, GradedTestCase
import gym
import numpy as np
import torch
import torch.nn as nn
import yaml
import time

from utils.general import join
from utils.network_utils import np2torch

# Import submission
from submission.baseline_network import BaselineNetwork
from submission.mlp import build_mlp
from submission.policy_gradient import PolicyGradient

# Import reference solution
if os.path.exists("./solution"):
    from solution.baseline_network import BaselineNetwork as RefBaselineNetwork
    from solution.mlp import build_mlp as ref_build_mlp
    from solution.policy_gradient import PolicyGradient as RefPolicyGradient
else:
    RefBaselineNetwork = BaselineNetwork
    ref_build_mlp = build_mlp
    RefPolicyGradient = PolicyGradient

# Import configuration settings
yaml.add_constructor("!join", join)

cartpole_config_file = open("config/cartpole_baseline.yml")
cartpole_config = yaml.load(cartpole_config_file, Loader=yaml.FullLoader)

pendulum_config_file = open("config/pendulum_baseline.yml")
pendulum_config = yaml.load(pendulum_config_file, Loader=yaml.FullLoader)

cheetah_config_file = open("config/cheetah_baseline.yml")
cheetah_config = yaml.load(cheetah_config_file, Loader=yaml.FullLoader)

if os.path.exists("./submission/model_artifacts"):
    model_path = "./submission/model_artifacts"
else:
    model_path = "./submission"

device = torch.device("cpu")

### BEGIN_HIDE ###
### END_HIDE ###

#########
# TESTS #
#########

### BEGIN_HIDE ###
### END_HIDE ###


# Baseline
class Test_1b(GradedTestCase):

    def setUp(self):

        self.config = cartpole_config
        self.config["model_training"]["device"] = "cpu"

    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """1b-0-basic: test baseline for the existence of optimizer"""
        env = gym.make(self.config["env"]["env_name"])
        baseline = BaselineNetwork(env, self.config)
        self.assertTrue(hasattr(baseline, "optimizer"))
        self.assertTrue(isinstance(baseline.optimizer, torch.optim.Optimizer))

    ### BEGIN_HIDE ###
### END_HIDE ###


# Policy
class Test_1c(GradedTestCase):

    def setUp(self):

        self.config = cartpole_config
        self.config["model_training"]["device"] = "cpu"

    @graded(timeout=3, is_hidden=False)
    def test_0(self):
        """1c-0-basic: test policy for the existence of optimizer"""
        env = gym.make(self.config["env"]["env_name"])
        pg = PolicyGradient(env, self.config, seed=self.config["env"]["seed"][0])
        self.assertTrue(hasattr(pg, "optimizer"))
        self.assertTrue(isinstance(pg.optimizer, torch.optim.Optimizer))

    ### BEGIN_HIDE ###
### END_HIDE ###

# Policy Gradient: get_returns
class Test_1e(GradedTestCase):
    @graded(timeout=1, is_hidden=False)
    def test_0(self):
        """1e-0-basic: test get_returns with basic trajectory"""
        config = cartpole_config
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=1)
        paths = [{"reward": np.zeros(11)}]
        returns = pg.get_returns(paths)
        expected = np.zeros(11)
        self.assertEqual(returns.shape, (11,))
        diff = np.mean((returns - expected) ** 2)
        self.assertAlmostEqual(diff, 0, delta=0.001)

    @graded(timeout=1, is_hidden=False)
    def test_1(self):
        """1e-1-basic: test get_returns for discounted trajectory"""
        config = cartpole_config
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=1)
        paths = [{"reward": np.ones(5)}]
        returns = pg.get_returns(paths)
        gamma = config["hyper_params"]["gamma"]
        expected = np.array(
            [
                1 + gamma + gamma**2 + gamma**3 + gamma**4,
                1 + gamma + gamma**2 + gamma**3,
                1 + gamma + gamma**2,
                1 + gamma,
                1,
            ]
        )
        diff = np.mean((returns - expected) ** 2)
        self.assertAlmostEqual(diff, 0, delta=0.001)

    ### BEGIN_HIDE ###
### END_HIDE ###


# Policy Gradient: sampled actions
class Test_1f(GradedTestCase):
    @graded(timeout=5, is_hidden=False)
    def test_0(self):
        """1f-0-basic: test sampled actions (cartpole)"""
        config = cartpole_config
        config["model_training"]["device"] = "cpu"
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        pg.policy = pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        actions = pg.policy.act(rand_obs)
        action_space = env.action_space
        discrete = isinstance(action_space, gym.spaces.Discrete)
        for action in actions:
            if discrete:
                self.assertTrue(action_space.contains(action))
            else:
                # We don't use contains because technically the Gaussian policy
                # doesn't respect the action bounds
                self.assertEqual(action_space.shape, action.shape)

    @graded(timeout=7, is_hidden=False)
    def test_1(self):
        """1f-1-basic: test sampled actions (pendulum)"""
        config = pendulum_config
        config["model_training"]["device"] = "cpu"
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        pg.policy = pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        actions = pg.policy.act(rand_obs)
        action_space = env.action_space
        discrete = isinstance(action_space, gym.spaces.Discrete)
        for action in actions:
            if discrete:
                self.assertTrue(action_space.contains(action))
            else:
                # We don't use contains because technically the Gaussian policy
                # doesn't respect the action bounds
                self.assertEqual(action_space.shape, action.shape)

    @graded(timeout=7, is_hidden=False)
    def test_2(self):
        """1f-2-basic: test sampled actions (cheetah)"""
        config = cheetah_config
        config["model_training"]["device"] = "cpu"
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        pg.policy = pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        actions = pg.policy.act(rand_obs)
        action_space = env.action_space
        discrete = isinstance(action_space, gym.spaces.Discrete)
        for action in actions:
            if discrete:
                self.assertTrue(action_space.contains(action))
            else:
                # We don't use contains because technically the Gaussian policy
                # doesn't respect the action bounds
                self.assertEqual(action_space.shape, action.shape)


# Policy Gradient
class Test_1g(GradedTestCase):
    @graded(timeout=4, is_hidden=False)
    def test_0(self):
        """1g-0-basic: test log probabilities (cartpole)"""
        config = cartpole_config
        config["model_training"]["device"] = "cpu"
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        ref_pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        policy = pg.policy.to(device)
        ref_policy = ref_pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        ref_policy.load_state_dict(policy.state_dict())
        actions = np2torch(policy.act(rand_obs))
        observations = np2torch(rand_obs)
        with torch.no_grad():
            log_probs = (
                policy.action_distribution(observations).log_prob(actions).cpu().numpy()
            )
            ref_log_probs = (
                ref_policy.action_distribution(observations)
                .log_prob(actions)
                .cpu()
                .numpy()
            )
        diff = np.mean((log_probs - ref_log_probs) ** 2)
        self.assertAlmostEqual(diff, 0, delta=0.01)

    @graded(timeout=4, is_hidden=False)
    def test_1(self):
        """1g-1-basic: test log probabilities (pendulum)"""
        config = pendulum_config
        device = torch.device("cpu")
        if config["model_training"]["device"] == "gpu":
            if torch.cuda.is_available(): 
                device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        ref_pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        policy = pg.policy.to(device)
        ref_policy = ref_pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        ref_policy.load_state_dict(policy.state_dict())
        actions = np2torch(policy.act(rand_obs))
        observations = np2torch(rand_obs)
        with torch.no_grad():
            log_probs = (
                policy.action_distribution(observations).log_prob(actions).cpu().numpy()
            )
            ref_log_probs = (
                ref_policy.action_distribution(observations)
                .log_prob(actions)
                .cpu()
                .numpy()
            )
        diff = np.mean((log_probs - ref_log_probs) ** 2)
        self.assertAlmostEqual(diff, 0, delta=0.01)

    @graded(timeout=4, is_hidden=False)
    def test_2(self):
        """1g-2-basic: test log probabilities (cheetah)"""
        config = cheetah_config
        config["model_training"]["device"] = "cpu"
        env = gym.make(config["env"]["env_name"])
        pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        ref_pg = PolicyGradient(env, config, seed=config["env"]["seed"][0])
        policy = pg.policy.to(device)
        ref_policy = ref_pg.policy.to(device)
        rand_obs = np.random.randn(10, pg.observation_dim)
        ref_policy.load_state_dict(policy.state_dict())
        actions = np2torch(policy.act(rand_obs))
        observations = np2torch(rand_obs)
        with torch.no_grad():
            log_probs = (
                policy.action_distribution(observations).log_prob(actions).cpu().numpy()
            )
            ref_log_probs = (
                ref_policy.action_distribution(observations)
                .log_prob(actions)
                .cpu()
                .numpy()
            )
        diff = np.mean((log_probs - ref_log_probs) ** 2)
        self.assertAlmostEqual(diff, 0, delta=0.01)


### BEGIN_HIDE ###
### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
