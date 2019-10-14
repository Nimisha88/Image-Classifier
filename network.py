import copy
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''

        super().__init__()

        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

def validation(model, validdataloader, criterion, device):
    ''' Validate the model

        Arguments
        ---------
        model: model to be validated
        dataloader: dataloader to loop through validation dataset
        criterion: loss criterion
    '''

    # print('\tValidating the Model..')

    # print("Device in use: {}".format(device))
    model.to(device)

    model.eval()
    running_val_loss = 0
    val_accuracy = 0
    batch_processed = 0

    with torch.no_grad():
        for images, labels in validdataloader:
            images, labels = images.to(device), labels.to(device)

            vlogps = model.forward(images)
            val_loss = criterion(vlogps, labels)
            running_val_loss += val_loss.item()

            ps = torch.exp(vlogps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            batch_processed += 1
            # print("\tValidation Batch Processed: {}/{}".format(batch_processed, len(validdataloader)))

    valid_loss = running_val_loss/len(validdataloader)
    acc = val_accuracy/len(validdataloader)

    return valid_loss, acc

def train(model, dataloaders, criterion, optimizer, scheduler, epochs, device):
    ''' Train the model

        Arguments
        ---------
        model: model to be validated
        dataloader: dataloader to loop through train or validation dataset
        criterion: loss criterion
        optimizer: optimize the weights and bias
        scheduler: scheduler to change learning rate step wise
        epochs: how many times the dataset must be looped
    '''

    print('\tTraining the Model..')

    print("Device in use: {}".format(device))
    model.to(device)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print('Epoch: {}/{}'.format(epoch+1, epochs))

        running_train_loss = 0
        batch_processed = 0

        # Train the Model
        scheduler.step()
        model.train()

        # print('\tTraining the Model')
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            train_loss = criterion(logps, labels)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

            batch_processed += 1
            # print("\tTraining Batch Processed: {}/{}".format(batch_processed, len(dataloaders['train'])))

        valid_loss, acc = validation(model, dataloaders['valid'], criterion, device)
        train_loss = running_train_loss/len(dataloaders['train'])

        print('Validation Accuracy: {:.3f}'.format(acc))
        print('Training Loss: {:.3f} \tValidation Loss: {:.3f}'.format(train_loss, valid_loss))

        if acc > best_acc:
            print('Found new Best Accuracy at Epoch: {}'.format(epoch+1))
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print('Best Validation Accuracy: {:.3f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def test_model_accuracy(model, dataloaders, device):
    model.to(device)

    model.eval()

    batch_processed = 0
    total_accuracy = 0

    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            batch_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            total_accuracy += batch_accuracy

            batch_processed += 1
            # print("\tTestingg Batch Processed: {}/{}".format(batch_processed, len(dataloaders['test'])))

    print("Test Accuracy: {:.3f}".format(total_accuracy/len(dataloaders['test'])))
