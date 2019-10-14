import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import network, predict

def load_data(data_dir):

    # Datapath to the Training, Validation and Testing Sets
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms for the Training, Validation and Testing Sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                       'valid' : transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                       'test' : transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])}
    # Load the datasets with ImageFolder
    image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=50, shuffle=True),
                   'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=25, shuffle=True),
                   'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=10)}

    return image_datasets, dataloaders



def get_trained_model(arch):
     try:
            model = getattr(models, arch)(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False

            model_classifier_name = [name for name, child in model.named_children()][-1]

            return model, model_classifier_name
     except AttributeError:
         print("{} is not a valid model".format(arch))
         raise SystemExit
     else:
        print("Error loading model")
        raise SystemExit



def get_in_units(classifier):
    classifier_children = [child for child in classifier.children()
                          if type(child) is torch.nn.modules.linear.Linear]

    if classifier_children:
        return classifier_children[0].in_features
    else:
        return classifier.in_features



def save_checkpoint(model, model_classifier_name, optimizer, save_dir, arch):

    checkpoint = {'arch': arch,
                  'input_size': getattr(model, model_classifier_name).hidden_layers[0].in_features,
                  'output_size': getattr(model, model_classifier_name).output.out_features,
                  'hidden_layers': [layer.out_features for layer in getattr(model, model_classifier_name).hidden_layers],
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optim_state_dict': optimizer.state_dict,
                  'dropout': getattr(model, model_classifier_name).dropout.p}

    print("Saving in Directory: {}".format(save_dir))
    torch.save(checkpoint, save_dir+'checkpoint.pth')
    print("Saved")



def load_checkpoint(file_path, device):

    if device == "cuda":
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    model, model_classifier_name = get_trained_model(checkpoint['arch'])

    classifier = network.Network(checkpoint['input_size'],
                                 checkpoint['hidden_layers'],
                                 checkpoint['output_size'],
                                 checkpoint['dropout'])

    setattr(model, model_classifier_name, classifier)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model, model_classifier_name



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    width, height = img.size

    # Resize
    if width < height:
        img = img.resize((256, int(256*(height/width))))
    else:
        img = img.resize((int(256*(width/height)), 256))

    width, height = img.size

    # Center Crop
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))

    # Color Encoding
    img = np.array(img)/255

    # Normalize
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std

    # Transpose
    img = img.transpose((2, 0, 1))

    # Turn into Tensor
    image = torch.from_numpy(img)
    image = image.float()

    return image



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        plt.title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



def label_mapping(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name



def fetch_flower(classes, model, cat_to_name):

    idx_to_cat = {val: key for key, val in model.class_to_idx.items()}

    catgs = [idx_to_cat[ele] for ele in classes]
    flowers = [cat_to_name[ele] for ele in catgs]

    to_flower= {idx: flower for idx, flower in zip(classes, flowers)}

    return flowers



def get_image_title(image_path, cat_to_name):
    flower_catg = image_path.split('/')[2]
    return cat_to_name[flower_catg]



def view_matlab_classify(image, title, probs, flowers):

    print("Displaying Image and its Predictions.. ")

    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    print("Show Image.. ")
    imshow(image, ax, title)

    print("Show Prediction.. ")
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()

def view_classify(title, probs, flowers):
    print("Label of the Image: {}".format(title))
    print("--------------------------------------------------")

    for prob, flower in zip(probs, flowers):
        print("Probability of the image being {} is {}".format(flower, prob))

    print("--------------------------------------------------")
