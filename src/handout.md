# Projecting Your Data
![](https://i.imgur.com/69GOnM9.gif)

In this assignment, we will use Pytorch to implement a Convolutional Neural Network (CNN) classifier for the MNIST datasets and discusses how to use Pytorch to build a neural network model from scratch for a custom dataset. After finishing the classifier, we will build an embedding projector based on the model to observe the model's behavior on the dataset. This assignment will combine the concepts including classification, convolution neural network, and working with embeddings.

Different hyperparameters or model architectures can lead to different outputs on the same dataset, and different dimensionality reduction algorithms can change the way we project the embeddings into a 2D or 3D space, which in turn affects the way we interpret the model. After building the embedding projector, we will explore how the projection of the embeddings changes under different settings and analyze the behavior of the model. 


This assignment is organized as follows:
- Part A. Data exploration
- Part B. Building the classifier
- Part C. Building the embedding projector
- Part D. Analyzing the embeddings

Much of this assignment is inspired by Google's embedding projector [[1]](#References), and we will explore how to make and improve our own embedding projector based on our needs.


## Part A. Data exploration

### MNIST Dataset

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

![](https://i.imgur.com/gE3EESR.png)


### Part A.1 - Load Dataset [3 pt]

A package called torchvision, that has data loaders for common datasets such as MNIST, Imagenet, CIFAR10, etc. For the following example, we will use the MNIST dataset. It has ten classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. The images in MNIST are of size 1x28x28, i.e. 1-channel images of 28x28 pixels in size.

Run the following example to load the dataset:
``` python
import torch
import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, # TODO
                                          shuffle=True, num_workers=2) 

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, # TODO
                                         shuffle=False, num_workers=2)

classes = (0, 1, 2, 3, 4, 5, 6, 7 , 8, 9)
```

#### Explain why the training dataset needs to be shuffled and the testing dataset does not.

### Part A.2 - Preview the Dataset [2 pt]

Run the following codes to preview the dataset:
``` python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
```

#### 

### Part A.3 - Build Your Dataset [5 pt]
`torch.utils.data.Dataset` is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:
- `__len__` so that `len(dataset)`` returns the size of the dataset.
- `__getitem__` to support the indexing such that dataset[i] can be used to get the ith sample.


Adjust the following codes to build your MNIST dataset:
``` python
import torch


class MNIST(torch.utils.data.Dataset):

    def __init__():
    
        # TODO
        
    def __len__():
    
        # TODo
    
    def __getitem__(self, index):
    
        # TODO

```

### Part A.4 - Data Augmentation [5 pt]

Data augmentations are techniques used to increase the amount of data by adding slightly modified copies of existing data or other synthetic data that we produce in the process of data augmentation.


The following example shows two different augmentation methods:
``` python
import torchvision.transforms as transforms

transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(28),
])
```

#### Explain your choice from Pytorch's transformations [[2]](#References) and why.

## Part B. Building the classifier

### Part B.1 - Define a Convolutional Neural Network [5 pt]

Build a convolutional neural network that takes the (1x28x28) images as input and outputs a 10-dimensional vector to indicate the relative possibilities of the ten classes. Your model should be a subclass of nn.Module.

Following is an example:

``` python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


net = Net()
```

#### Explain your choice of neural network architecture:
- Try to change the number of convolutional layers in the forward step, and observe the resulting classification qualities. What is the number of convolutional layers that seem to offer the best performance?
- What types of layers, i.e., conv1, conv2, or a different design of yours, did you use?

### Part B.2 - Define a Loss function and optimizer [3 pt]

#### Loss Function

A loss function or cost function is a function that maps an event or values of one or more variables onto a real number, intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize the loss/cost - [Wikipedia](https://en.wikipedia.org/wiki/Loss_function).

Pytorch offers different loss functions for various purposes [[3]](#References).

#### Optimizer

Optimizers are software agents that employ specific algorithms to achieve the minimization goal.

Pytorch also offers many optimization algorithms for different scenarios [[4]](#References).

The following block shows an example:
``` python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

You may explore different combinations of loss functions, optimizers, and other hyperparameters based on your needs, and compare the results.

#### Observe the performances of some different combinations of components, and find out the best one. Are there logical relations between the classification performances and the models of your design?

#### Explain your choice of loss function and optimizer.

#### Observe the performances of some different combinations of optimizer and loss function, and find out the best one. Are there logical relations between the classification performances and the models of your design?

### Part B.3 - Train the network [2 pt]

Train your first network on your training set.


Adjust the following codes to train the model:
``` python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # TODO

print('Finished Training')
```

The number of epochs is obviously an influential factor that affects the final outcome of the classification. You should try different alternatives and observe the corresponding results.

## Part C. Building the embedding projector 

### Part C.1 - Build an embedding projector [5 pt]

In this assignment, we use [TensorBoard](https://www.tensorflow.org/tensorboard/) as our visualization tool for projecting the embeddings.

Run the following example:
``` python
import keyword
import torch
from torch.utils.tensorboard import SummaryWriter

# The writer will output to ./runs/ directory by default.
writer = SummaryWriter()
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist # get some strings
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100.0

writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.close()
```


Then navigate to your command line and execute the following command to open TensorBoard:
``` bash
$ tensorboard --logdir=runs --bind_all
```

#### Paste your results here.

### Part C.2 - Project the Embeddings [5 pt]


Adjust the following codes to write the embeddings to the TensorBoard:
``` pyhton
correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # stack your embeddings and other data 
        # TODO

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    
# write the embeddings to the tensorboard
# TODO
```

#### Paste your results on TensorBoard here.

## Part D. Analysing the embeddings

### Part D.1 - Analysing the embeddings with PCA - [2 pts]

Principal component analysis finds a new coordinate system for a given dataset, trying to minimize the variance among the data from the perspective of the created new system. The first few components are usually more important for capturing the distributional characteristics of the given dataset. Encoding the data based on the first few principal components is a common heuristic for reducing dimensionality while minimizing the loss of information about the data.

#### Choose different combinations of principal components in the following panel (the red box on the left-hand side) and observe the distribution of the data in the chosen coordinate system.


![](https://i.imgur.com/9pEZV7W.png)

Which combination of three components seems to separate the images of different classes best visually?

#### Explain the meanings of variances in PCA when setting different principal components.

### Part D.2 - Analysing with different models [3 pts]

Adjust your model architecture, project the new embeddings, and compare with the results from the previous one.

Following is a sample:
``` python
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn

class Net(ResNet):
    def __init__(self):
        # Based on ResNet18
        super(Net, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

model = Net()
```

## Additionals

### Fashion-MNIST

Pytorch provides a variety of different and organized datasets that can be used quickly [[5]](#References). Following is an example of modifying the program in A.1 from the MNIST dataset to the Fashion-MNIST dataset:

``` python
transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 4

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

```

\
Retrain the model and project it to get different results than the MNIST dataset:

\
![](https://i.imgur.com/qXX7OpT.png)



## References


[1] D. Smilkov, N. Thorat, C. Nicholson, E. Reif, F. B. Viégas, and M. Wattenberg, “Embedding Projector: Interactive Visualization and Interpretation of Embeddings,” arXiv:1611.05469 [cs, stat], Nov. 2016, Accessed: Sep. 09, 2021. [Online]. Available: http://arxiv.org/abs/1611.05469

[2] Pytorch Transforms - https://pytorch.org/vision/stable/transforms.html

[3] Pytorch Loss Functions - https://pytorch.org/docs/stable/nn.html#loss-functions

[4] Pytorch Optimization Algorithms - https://pytorch.org/docs/stable/optim.html

[5] Pytorch Datasets - https://pytorch.org/vision/stable/datasets.html