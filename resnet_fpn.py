import torch

import torch.nn as nn

from torchvision.datasets import Country211
from torchvision import transforms

from torchvision.ops import FeaturePyramidNetwork

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights

from ignite.engine import *
from ignite.metrics import *

from tqdm import tqdm

import matplotlib.pyplot as plt

import math

import numpy as np

import copy


NAME = "run_3_resnet_fpn_full_l2_0_epochs_100_lr_1e-4_conv_0_group_4"

EPOCHS = 100
lr = 1e-4
l2 = 0

# dataset/dataloader stuff

batch_size = 32

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

model1 = resnet50(weights=ResNet50_Weights.DEFAULT)

feature_extractor1 = create_feature_extractor(model1, 
        return_nodes=["layer1.0.conv3"]).to(device)


model2 = copy.deepcopy(model1)

feature_extractor2 = create_feature_extractor(model2, 
        return_nodes=["layer2.0.conv3"]).to(device)


model3 = copy.deepcopy(model1)

feature_extractor3 = create_feature_extractor(model3, 
        return_nodes=["layer3.0.conv3"]).to(device)


model4 = copy.deepcopy(model1)

feature_extractor4 = create_feature_extractor(model4, 
        return_nodes=["layer4.0.conv3"]).to(device)

# custom classification head

# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ClassificationHead(nn.Module):
    def __init__(self, output_size):
        super(ClassificationHead, self).__init__()
        # pooling to make sure dimensionality is the same for features across multiple layers
        self.pool = nn.AdaptiveAvgPool2d(14)
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.bnorm1 = nn.BatchNorm2d(256)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm4 = nn.BatchNorm2d(256)

        # final layers
        self.final_conv1 = nn.Conv2d(1024, 1024, (3, 3), bias=False, groups=4)
        self.bnorm5 = nn.BatchNorm2d(1024)

        self.final_conv2 = nn.Conv2d(1024, 1024, (3, 3), bias=False, groups=4)
        self.bnorm6 = nn.BatchNorm2d(1024)

        self.final_conv3 = nn.Conv2d(1024, 1024, (3, 3), bias=False, groups=4)
        self.bnorm7 = nn.BatchNorm2d(1024)

        # flatten layers creates large number of features
        # self.flatten = nn.Flatten()

        self.aggregate = nn.AdaptiveAvgPool2d((1, 1)) # global feature aggregation for regression output connection

        self.dense1 = nn.Linear(1024, output_size)

        self.gelu = GELU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out1 = self.pool(self.bnorm1(x["layer1.0.conv3"]))
        out2 = self.pool(self.bnorm2(x["layer2.0.conv3"]))
        out3 = self.bnorm3(x["layer3.0.conv3"])
        out4 = self.upsamplex2(self.bnorm4(x["layer4.0.conv3"]))

        concat_output = torch.cat((out1, out2, out3, out4), 1)

        final = self.gelu(self.bnorm5(self.final_conv1(concat_output)))
        final = self.gelu(self.bnorm6(self.final_conv2(final)))
        final = self.gelu(self.bnorm7(self.final_conv3(final)))

        final = self.aggregate(final).squeeze(2).squeeze(2)

        # final linear output
        final = self.softmax(self.dense1(final))

        return final

classification_head = ClassificationHead(211)
classification_head.to(device)

fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256).to(device)

# optimization stuff

crossentropy_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(feature_extractor1.parameters()) + list(feature_extractor2.parameters()) + list(feature_extractor3.parameters()) + list(feature_extractor4.parameters()) + list(fpn.parameters()) + list(classification_head.parameters()), lr=lr, weight_decay=l2)

# main training loop

N_batch = 40

train_losses = []
# train_accuracies = []

val_losses = []
# val_accuracies = []


for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")

    feature_extractor1.train()
    feature_extractor2.train()
    feature_extractor3.train()
    feature_extractor4.train()
    fpn.train()
    classification_head.train()

    for i, (sample, target) in tqdm(enumerate(train_loader), total=len(train_loader)-1):
        optimizer.zero_grad()

        one_hot_target = nn.functional.one_hot(target, num_classes = 211).type(torch.FloatTensor).to(device)

        output = feature_extractor1(sample.to(device))
        output.update(feature_extractor2(sample.to(device)))
        output.update(feature_extractor3(sample.to(device)))
        output.update(feature_extractor4(sample.to(device)))
        output = fpn(output)
        output = classification_head(output)

        loss = crossentropy_loss(output, one_hot_target)

        loss.backward()
        
        optimizer.step()

        del sample
        del target
    
    feature_extractor1.eval()
    feature_extractor2.eval()
    feature_extractor3.eval()
    feature_extractor4.eval()
    fpn.eval()
    classification_head.eval()

    temp_losses = []

    for i, (sample, target) in tqdm(enumerate(train_loader), total=N_batch):
        one_hot_target = nn.functional.one_hot(target, num_classes = 211).type(torch.FloatTensor).to(device)

        with torch.no_grad():
            output = feature_extractor1(sample.to(device))
            output.update(feature_extractor2(sample.to(device)))
            output.update(feature_extractor3(sample.to(device)))
            output.update(feature_extractor4(sample.to(device)))
            output = fpn(output)
            output = classification_head(output)

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
            output = feature_extractor1(sample.to(device))
            output.update(feature_extractor2(sample.to(device)))
            output.update(feature_extractor3(sample.to(device)))
            output.update(feature_extractor4(sample.to(device)))
            output = fpn(output)
            output = classification_head(output)

        loss = crossentropy_loss(output, one_hot_target)

        temp_losses += [loss.item()]
        
        if i == len(val_loader)-1:
            val_losses.append(np.mean(np.array(temp_losses)))
            print("Val loss: " + str(np.mean(np.array(temp_losses))))

            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.savefig("loss_graphs/" + NAME + ".png")

            # save fpn, and head

            torch.save(feature_extractor1, "models/feature_extractor1_" + NAME)
            torch.save(feature_extractor2, "models/feature_extractor2_" + NAME)
            torch.save(feature_extractor3, "models/feature_extractor3_" + NAME)
            torch.save(feature_extractor4, "models/feature_extractor4_" + NAME)
            torch.save(fpn, "models/fpn_" + NAME)
            torch.save(classification_head, "models/" + NAME)

        del sample
        del target