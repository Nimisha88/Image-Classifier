import pandas as pd
import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import network, utils



def get_parser():
    ''' Fetch an Argument Parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store', help='Path to image to be classified')
    parser.add_argument('checkpoint', action='store', help='Path to stored model')
    parser.add_argument('--top_k', action='store', nargs='?', type=int, default=5,
                        help='How many most probable classes')
    parser.add_argument('--category_names', action='store', nargs='?', type=str, default = 'cat_to_name.json',
                        help='Classes to names mapping file')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU')
    return parser



def predict(model, image, vtopk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()

    # Implement the code to predict the class from an image file
    with torch.no_grad():
        logps = model.forward(image.unsqueeze_(0))
        ps = torch.exp(logps)
        probs, classes = ps.topk(vtopk, dim=1)

    probs, classes = probs.squeeze().tolist(), classes.squeeze().tolist()
    print("Probabilities: {} \nClasses: {}".format(probs, classes))
    flowers = utils.fetch_flower(classes, model, cat_to_name)
    print("Flowers: {}".format(flowers))

    return probs, flowers



if __name__ == '__main__':

    args = get_parser().parse_args()

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print("Device in use: {}".format(device))

    model, model_classifier_name = utils.load_checkpoint(args.checkpoint, device)
    print("Model: \n{}".format(model))

    image = utils.process_image(args.input)
    cat_to_name = utils.label_mapping(args.category_names)
    title = utils.get_image_title(args.input, cat_to_name)
    
    probs, flowers = predict(model, image, args.top_k, device, cat_to_name)
    utils.view_classify(title, probs, flowers)
