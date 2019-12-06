import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import torchvision
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import pathlib
import torchvision.transforms as transforms

#def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
def inception_score(data_loader, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(data_loader) * batch_size

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch, _) in enumerate(data_loader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def load_dataset(batch_size):
    #data_path = '~/tmpdev0/text-to-image-gen-model-acgan/output/samples_final'
    data_path = '/home/ooo/Documents/CS236/text-to-image-gen-model-acgan/output/samples_gen'
    eval_dataset = torchvision.datasets.ImageFolder(
        root = data_path,
        transform = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size = batch_size,
            num_workers = 1,
            shuffle=False
    )
    return eval_loader

if __name__ == '__main__':

    batch_size = 50
    eval_loader = load_dataset(batch_size)

    print ("Calculating Inception Score...")
    print (inception_score(eval_loader, cuda=True, batch_size=batch_size, resize=True, splits=10))

