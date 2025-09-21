import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from utils.general import join, plot_combined
from submission.policy_gradient import PolicyGradient

import random
import yaml

yaml.add_constructor("!join", join)

parser = argparse.ArgumentParser()
parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--plot_config_filename", required=False, type=str)
parser.add_argument("--run_basic_tests", required=False, type=bool)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_filename is not None:
        config_file = open("config/{}.yml".format(args.config_filename))
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
        for seed in config["env"]["seed"]:
            torch.random.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            env = gym.make(config["env"]["env_name"])

            # train model
            model = PolicyGradient(env, config, seed)
            model.run()
    else:
        print("Skipping model training as no config provided.")

    if args.plot_config_filename is not None:
        print(args.plot_config_filename)
        config_file = open("config/{}.yml".format(args.plot_config_filename))
        config = yaml.load(config_file, Loader=yaml.FullLoader)

        for env in config.keys():
            gym_env_name = config[env]["env_name"]

            all_results = {"Baseline": [], "No baseline": []}
            for seed in config[env]["seed"]:
                baseline_directory = "./results/{}-{}-baseline/".format(
                    gym_env_name, seed
                )
                no_baseline_directory = "./results/{}-{}-no-baseline/".format(
                    gym_env_name, seed
                )
                if not os.path.isdir(no_baseline_directory):
                    sys.exit(
                        "{} was not found. Please ensure you have generated results for this environment, seed and baseline combination".format(
                            no_baseline_directory
                        )
                    )
                if not os.path.isdir(baseline_directory):
                    sys.exit(
                        "{} was not found. Please ensure you have generated results for this environment, seed and baseline combination".format(
                            baseline_directory
                        )
                    )
                all_results["Baseline"].append(
                    np.load(baseline_directory + "scores.npy")
                )
                all_results["No baseline"].append(
                    np.load(no_baseline_directory + "scores.npy")
                )

            plt.figure()
            plt.title(gym_env_name)
            plt.xlabel("Iteration")
            for name, results in all_results.items():
                print(name)
                plot_combined(name, results)
            plt.legend()
            plt.savefig("./results/{}".format(gym_env_name), bbox_inches="tight")
            print("plot saved.")
    else:
        print("Skipping generating plot of multiple seeds as no config provided")

    if args.run_basic_tests is not None:
        suite = unittest.defaultTestLoader.discover("basic_tests")
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        print("Basic tests not run")
