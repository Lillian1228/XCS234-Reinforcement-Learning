import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger, join
from utils.test_env import EnvTest
from .q3_schedule import LinearExploration, LinearSchedule
from .q5_linear_torch import Linear

import yaml
from collections import OrderedDict
yaml.add_constructor("!join", join)

config_file = open("config/q6_dqn.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 6: Implementing DeepMind's DQN
############################################################

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0):
    h = ( (h_w[0] + 2 * pad - kernel_size  )/ stride) + 1
    w = ( (h_w[1] + 2 * pad - kernel_size)  / stride) + 1
    return h, w

class NatureQN(Linear):
    """
    Implementation of DeepMind's Nature paper, please consult the methods section
    of the paper linked below for details on model configuration.
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """

    ############################################################
    # Problem 6a: initialize_models

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The in_channels 
        to Conv2d networks will n_channels * self.config["hyper_params"]["state_history"]

        Args:
            q_network (torch model): variable to store our q network implementation

            target_network (torch model): variable to store our target network implementation

        TODO:
             (1) Set self.q_network to the architecture defined in the Nature paper associated to this question.
                Padding isn't addressed in the paper but here we will apply padding of size 2 to each dimension of
                the input to the first conv layer (this should be an argument in nn.Conv2d).
            (2) Set self.target_network to be the same configuration self.q_network but initialized from scratch
            (3) Be sure to use nn.Sequential in your implementation.

        Hints:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) The following functions might be useful
                - nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
                - nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
                - nn.ReLU (https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
                - nn.Flatten (https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
                - nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        ### START CODE HERE ###
        state_history = self.config["hyper_params"]["state_history"]
#         print("img_height: ", img_height)
#         print("img_width: ", img_width)
#         print("n_channels: ", n_channels)
#         print("n_actions: ", num_actions)        
#         print(n_channels*state_history )
        
        Q_network = OrderedDict()
        Q_network['0'] = nn.Conv2d(n_channels *state_history, 32, 8, stride=4, padding=2)
        Q_network['1'] = nn.ReLU()
        conv1_h, conv1_w = conv_output_shape((img_height, img_width), kernel_size=8, stride=4, pad=2)
        
        Q_network['2'] = nn.Conv2d(32, 64, 4, stride=2)
        Q_network['3'] = nn.ReLU()
        conv2_h, conv2_w = conv_output_shape((conv1_h, conv1_w), kernel_size=4, stride=2, pad=0)
        Q_network['4'] = nn.Conv2d(64, 64, 3, stride=1)
        Q_network['5'] = nn.ReLU()  
        conv3_h, conv3_w = conv_output_shape((conv2_h, conv2_w), kernel_size=3, stride=1, pad=0)
#         print(conv3_h, conv3_w)
        input_size = int(conv3_h*conv3_w*64)
#         print(input_size)
        Q_network['6'] = nn.Flatten()
        Q_network['7'] = nn.Linear(input_size, 512)
        Q_network['8'] = nn.ReLU()  
        Q_network['9'] = nn.Linear(512, num_actions)
        
        self.q_network = nn.Sequential(Q_network)

        target_network = OrderedDict()
        target_network['0'] = nn.Conv2d(n_channels *state_history, 32, 8, stride=4, padding=2)
        target_network['1'] = nn.ReLU()
        target_network['2'] = nn.Conv2d(32, 64, 4, stride=2)
        target_network['3'] = nn.ReLU()
        target_network['4'] = nn.Conv2d(64, 64, 3, stride=1)
        target_network['5'] = nn.ReLU()  
        target_network['6'] = nn.Flatten()
        target_network['7'] = nn.Linear(input_size, 512)
        target_network['8'] = nn.ReLU()  
        target_network['9'] = nn.Linear(512, num_actions)
        
        self.target_network = nn.Sequential(target_network)

        ### END CODE HERE ###

    ############################################################
    # Problem 6b: get_q_values

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state (torch tensor): shape = (batch_size, img height, img width,
                                            nchannels x config["hyper_params"]["state_history"])

            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network
            and return the output values.


        Hints:
            (1) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
            (2) Look up torch.permute (https://pytorch.org/docs/stable/generated/torch.permute.html)
        """
        out = None

        ### START CODE HERE ###
        state = torch.permute(state,(0,3,1,2))
        #print(state.shape)
        if network=="q_network":
            out = self.q_network(state)
        elif network=="target_network":
            out = self.target_network(state)
        else:
            pass
        ### END CODE HERE ###

        return out
