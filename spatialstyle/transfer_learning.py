import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy



class StyleTransfer():

    def __init__(self, pretrained_model=None):
        pass

    def transfer(self, content_file_path, style_file_paths, masks=None):
        pass



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


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

def get_style_model_and_losses_lists(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default, spatial_mask=None):
    
    if spatial_mask is None:
        raise ValueError("A spatial mask must be provided")
        
    if len(spatial_mask) != len(style_img):
        raise ValueError("Number of spatial masks and number of style images must be equal")
    
    
    style_img = torch.stack(style_img).squeeze(1)
    
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    
    # We create a receptive averaging network whose input is the mask
    #receptive_averaging = nn.Sequential()#mask)
    
    # A list with the spatial masks for the style loss
    receptive_masks = [ [] for k in range(len(style_img)) ]
    receptive_masks = []
    layer_mask = spatial_mask
    receptive_masks.append(layer_mask)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        r_layer = None
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            
            #r_name = 'conv_{}'.format(i)
            #r_layer = nn.Conv2d()
            #nn.AvgPool2d(layer.kernel_size,layer.stride)
            #layer_mask = get_new_mask(layer_mask, 'Conv', layer.kernel_size, layer.stride, layer.padding)
            
### how did you find that?
            r_avg = nn.AvgPool2d(layer.kernel_size,layer.stride, layer.padding)
            #layer_mask = r_avg(layer_mask.unsqueeze(0)).squeeze(0)
            layer_mask = r_avg(layer_mask)
            
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            
            #r_name = 'pool_{}'.format(i)
            #r_layer = layer
            #layer_mask = get_new_mask(layer_mask, 'MaxPooling', layer.kernel_size, layer.stride, layer.padding)
            r_max = nn.MaxPool2d(layer.kernel_size,layer.stride, layer.padding)
            #layer_mask = r_max(layer_mask.unsqueeze(0)).squeeze(0)
            layer_mask = r_max(layer_mask)
            
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        
        #if r_layer is not None:
        #    layer_mask = r_layer(layer_mask)
        #    receptive_masks.append(layer_mask)
            #receptive_averaging.add_module(r_name, r_layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:            
            
            # add style loss:
            target_feature = []
            
            target_feature = model(style_img).detach()
            
            #for k in range(len(style_img)):
            #    target_feature.append(model(style_img[k]).detach())
            
            style_loss = StyleLoss(target_feature, layer_mask)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]


    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1, spatial_mask=None):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses_lists(cnn,
        normalization_mean, normalization_std, style_img, content_img, spatial_mask=spatial_mask)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img





