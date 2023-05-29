import torch

import torch.nn as nn

from torchvision.datasets import Country211
from torchvision import transforms

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights

from ignite.engine import *
from ignite.metrics import *

from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np


NAME = "resnet_simple_l2_0_epochs_100_lr_1e-3"

EPOCHS = 100
lr = 1e-3
l2 = 0

# dataset/dataloader stuff

batch_size = 64

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.Resize((448,448)), # resize the images to 224x224 pixels
            transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
            transforms.GaussianBlur((5, 5), sigma=(0.1, 0.3)),
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

train_set = Country211("./", "train", transform=transform)
val_set = Country211("./", "valid", transform=transform)
test_set = Country211("./", "test", transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# pretrained backbone

# loading pretrained model
device = torch.device("cuda")

model = resnet50(weights=ResNet50_Weights.DEFAULT)

layer_names = []

for name, layer in model.named_modules():
        layer_names += [name]

print(layer_names)

feature_extractor = create_feature_extractor(model, 
        return_nodes=["avgpool"]).to(device)

# custom classification head

class ClassificationHead(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassificationHead, self).__init__()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(input_size, output_size)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)

        x = self.linear(x)

        return self.softmax(x)

classification_head = ClassificationHead(2048, 211)
classification_head.to(device)

# optimization stuff

crossentropy_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classification_head.parameters(), lr=lr)

# main training loop

N_batch = 40

train_losses = []
# train_accuracies = []

val_losses = []
# val_accuracies = []

feature_extractor.eval()

for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")

    classification_head.train()

    for i, (sample, target) in tqdm(enumerate(train_loader), total=len(train_loader)-1):
        optimizer.zero_grad()

        one_hot_target = nn.functional.one_hot(target, num_classes = 211).type(torch.FloatTensor).to(device)

        with torch.no_grad():
            output = feature_extractor(sample.to(device))
        
        output = classification_head(output["avgpool"])

        loss = crossentropy_loss(output, one_hot_target)

        loss.backward()
        
        optimizer.step()

        del sample
        del target
    
    classification_head.eval()

    temp_losses = []

    for i, (sample, target) in tqdm(enumerate(train_loader), total=N_batch):
        one_hot_target = nn.functional.one_hot(target, num_classes = 211).type(torch.FloatTensor).to(device)

        with torch.no_grad():
            output = feature_extractor(sample.to(device))
            output = classification_head(output["avgpool"])

        loss = crossentropy_loss(output, one_hot_target)

        temp_losses += [loss.item()]

        del sample
        del target
        
        if i == N_batch-1:
            train_losses.append(np.mean(np.array(temp_losses)))
            print("Train loss: " + str(np.mean(np.array(temp_losses))))
            break
    
    temp_losses = []

    for i, (sample, target) in tqdm(enumerate(val_loader), total=len(val_loader)-1):
        one_hot_target = nn.functional.one_hot(target, num_classes = 211).type(torch.FloatTensor).to(device)

        with torch.no_grad():
            output = feature_extractor(sample.to(device))
            output = classification_head(output["avgpool"])

        loss = crossentropy_loss(output, one_hot_target)

        temp_losses += [loss.item()]
        
        if i == len(val_loader)-1:
            val_losses.append(np.mean(np.array(temp_losses)))
            print("Val loss: " + str(np.mean(np.array(temp_losses))))

            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.savefig("loss_graphs/" + NAME + ".png")

            # save head

            torch.save(classification_head, "models/" + NAME)

        del sample
        del target