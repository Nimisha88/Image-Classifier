import pandas as pd
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import network, utils
from workspace_utils import active_session



def get_parser():
    ''' Fetch an Argument Parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', type=str, help='Directory containing Datasets')
    parser.add_argument('--save_dir', action='store', nargs='?', type=str, default='.',
                        help='Save Checkpoint to directory')
    parser.add_argument('--arch', action='store', nargs='?', type=str, default='resnet50',
                        help='Pretrained Model to be used')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--epochs', action='store', nargs='?', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', action='store', nargs='?', type=float, default=0.003,
                        help='which learning rate to start with')
    parser.add_argument('--hidden_units', action='store', nargs='*', type=int, default = 0,
                        help='Number of In Features for Hidden Layers')
    parser.add_argument('--output_size', action='store', nargs='?', type=int, default=102, help='Number for Out Features')

    return parser


def create_model(data_dir, arch, hidden_units, output_size, drop):
    # Loading Data
    print("Loading Data from Data directory: {}".format(data_dir))
    image_datasets, dataloaders = utils.load_data(data_dir)

    # Fetching Pretrained Model and it's Classifier
    model, model_classifier_name = utils.get_trained_model(arch)

    # Add class to index to Model
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Fetching the Num of Input for the custom Classifier to be created
    input_size = utils.get_in_units(getattr(model, model_classifier_name))
    print("Input Size: {}".format(input_size))

    # Tackle Hidden Units
    if (hidden_units == 0):
        hidden_units = []
        hidden_units.append(int(in_units/2))
    print("Hidden Units: {}".format(hidden_units))
    print("Output Size: {}".format(output_size))

    # Create the Custom Classifier
    classifier = network.Network(input_size, hidden_units, output_size, drop)

    # Load the Pretrained Model with the custom Classifier
    setattr(model, model_classifier_name, classifier)

    return model, model_classifier_name, image_datasets, dataloaders


if __name__ == '__main__':

    args = get_parser().parse_args()

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    if not (args.save_dir[-1] == '/'):
        save_dir = args.save_dir + '/'
    else:
        save_dir = args.save_dir

    model, model_classifier_name, image_datasets, dataloaders = create_model(args.data_dir, args.arch,
                                                                             args.hidden_units, args.output_size,
                                                                             drop = 0.35)
    print("Model:\n", model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(getattr(model, model_classifier_name).parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the Model and save the best Model
    with active_session():
        print("Training and Validating the Model.. ")
        model = network.train(model, dataloaders, criterion, optimizer, scheduler, args.epochs, device)

        print("Saving the Model in dir({}).. ".format(save_dir))
        model.to("cpu")
        utils.save_checkpoint(model, model_classifier_name, optimizer, save_dir, args.arch)

    # Calculate Accuracy on Test Data Set
    print("Calculating Test Accuracy of the trained Model.. ")
    network.test_model_accuracy(model, dataloaders, device)
