import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        if self.target.size() != input.size():
            target = self.target.expand(input.size())
        else:
            target = self.target
        
        
        self.loss = F.mse_loss(input, target)
        return input


def gram_matrix(input, T):
    a, b, c, d = input.size()  # a=batch size(=1)
    m, cp, dp = T.size()

    assert cp == c and dp == d
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)


    # Produce a spatial mask for the filter activations
    #m1 = torch.ones((a * b, math.floor(c * d / 2)))
    #m2 = torch.zeros((a * b, math.ceil(c * d / 2)))
    #T = torch.cat((m1, m2), dim=1)
    masked_input = torch.einsum('abcd,mcd->ambcd', [input, T])
    masked_features = masked_input.view(a, m, b, c * d)

    #features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    #masked_features = torch.mul(T.view(a * b, c * d), features)

    #G = torch.mm(masked_features, masked_features.t())  # compute the gram product
    G = torch.einsum('amik,amjk->amij', [masked_features, masked_features])

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.

    # Let gram_matrix output: batch x masks x spatial x spatial
    return G.div(b * c * d)

def target_gram(target_feature, T):
    m, b, c, d = target_feature.size()  # a=batch size(=1)
    mp, cp, dp = T.size()

    assert mp == m and cp == c and dp == d
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)


    # Produce a spatial mask for the filter activations
    #m1 = torch.ones((a * b, math.floor(c * d / 2)))
    #m2 = torch.zeros((a * b, math.ceil(c * d / 2)))
    #T = torch.cat((m1, m2), dim=1)
    masked_input = torch.einsum('mbcd,mcd->mbcd', [target_feature, T])
    masked_features = masked_input.view(m, b, c * d)

    #features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    #masked_features = torch.mul(T.view(a * b, c * d), features)

    #G = torch.mm(masked_features, masked_features.t())  # compute the gram product
    G = torch.einsum('mik,mjk->mij', [masked_features, masked_features])


    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.

    # Let gram_matrix output: batch x masks x spatial x spatial
    return G.div(b * c * d).unsqueeze(0)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, mask):
        super(StyleLoss, self).__init__()
        self.target = []
        self.mask = mask
        #tf = torch.stack(target_feature).squeeze(1) # will have one too much dim

        self.target = target_gram(target_feature, mask).detach()


        #for k in range(len(target_feature)):
        #    self.target.append(gram_matrix(target_feature[k], mask[k]).detach())
        #    self.mask.append(mask[k])

    def forward(self, input):
        self.loss=0
        G = gram_matrix(input, self.mask)
        #print(G.size())
        #print(self.target.size())

        if self.target.size() != G.size():
            target = self.target.expand(G.size())
        else:
            target = self.target
        self.loss = F.mse_loss(G, target)

        return input
    
# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
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
    
    return image_loader


unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    plt.ion() # Make sure we can plot right away
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
