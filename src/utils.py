import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # The target is a constant value and should not be back-propagated through
        self.target = target.detach()
        self.loss = 0

    def forward(self, x_input):
        
        # Duplicate the tensor in case we have a batch of inputs
        if self.target.size() != x_input.size():
            target = self.target.expand(x_input.size())
        else:
            target = self.target
        
        self.loss = F.mse_loss(x_input, target)

        # The loss is saved and the input is forwarded through 
        return x_input


def gram_matrix(x_input, T):
    a, b, c, d = x_input.size()  # a=batch size(=1)
    m, cp, dp = T.size()

    # Make sure we have the right dimensions 
    assert cp == c and dp == d

    # Produce a spatial mask for the filter activations
    masked_input = torch.einsum('abcd,mcd->ambcd', [x_input, T])
    masked_features = masked_input.view(a, m, b, c * d)

    # Compute the gram product
    G = torch.einsum('amik,amjk->amij', [masked_features, masked_features])

    # We 'normalize' the values of the gram matrix
    # by dividing by the number of elements in each feature maps.
    # Let gram_matrix output: batch x masks x spatial x spatial
    return G.div(b * c * d)


# Extra function for computing the constant target Gram-matrix
def target_gram(target_feature, T):
    m, b, c, d = target_feature.size()  # a=batch size(=1)
    mp, cp, dp = T.size()

    # Make sure we have the right dimensions
    assert mp == m and cp == c and dp == d

    # Produce a spatial mask for the filter activations
    masked_input = torch.einsum('mbcd,mcd->mbcd', [target_feature, T])
    masked_features = masked_input.view(m, b, c * d)

    # Compute the gram product
    G = torch.einsum('mik,mjk->mij', [masked_features, masked_features])

    # We 'normalize' the values of the gram matrix
    # by dividing by the number of elements in each feature maps.
    # Let gram_matrix output: batch x masks x spatial x spatial
    return G.div(b * c * d).unsqueeze(0)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, mask):
        super(StyleLoss, self).__init__()
        self.target = []
        self.mask = mask
        self.target = target_gram(target_feature, mask).detach()
        self.loss = 0

    def forward(self, x_input):

        G = gram_matrix(x_input, self.mask)

        # Make sure the target fits the batch size
        if self.target.size() != G.size():
            target = self.target.expand(G.size())
        else:
            target = self.target

        self.loss = F.mse_loss(G, target)

        # The loss is saved and the input is forwarded through
        return x_input
    

# Create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean=None, std=None):
        super(Normalization, self).__init__()

        # Set default mean and std
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        # Reshape the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    

def get_image_loader(imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    
    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)

        return image.to(device, torch.float)

    # Returns a function that can load and transform an image from file
    return image_loader


unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None):
    plt.ion()  # Make sure we can plot right away

    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    plt.imshow(image)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
