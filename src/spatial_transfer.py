import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import copy

from utils import get_image_loader, StyleLoss, ContentLoss, Normalization


class StyleTransfer:
    """A class that transfers multiple styles into one image with the styles spatial intensity given by masks.

    Attributes
    ----------
    device : torch.device
        The device on which the model will be used
    cnn : torch.nn.modules.container.Sequential
        The used torch module as sequential
    tractable_layer_names : set of str
        A set of strings which indicates which layer is a convolution
    content_layers : list of str
        The list of which convolutional layers shall be considered in the content loss calculation
    style_layers : list of str
        The list of which convolutional layers shall be considered in the style loss calculation

    Methods
    -------
    run_style_transfer(content_path, style_paths, spatial_mask=None, imsize=128, num_steps=300, style_weight=1000000,
                        content_weight=1)
        Executes style transfer on given image with given style images and masks.
    get_style_model_and_losses_lists(style_img, content_img, spatial_mask=None)
        Returns the model, content and style losses with implemented receptive averaging.
    modify_layers(content_layers, style_layers)
        Sets up the convolutional layers in the network which are used for content and style loss calculation.
    get_tractable_layer_names():
        Returns a set of strings which give information which layer are convolutions
    """

    def __init__(self, pretrained_model=None, device=None, layers=None):
        
        if device is None:
            self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        else: 
            self.device = device
        
        if pretrained_model is None:
            
            print("Using default pre-trained network")
            
            self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        else:
            self.cnn = pretrained_model.to(self.device).eval()
            
        self.tractable_layer_names = self.get_tractable_layer_names()
        
        if layers is None:
            
            print("Using default style and content layers")
            
            content_layers = ['conv_4']
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

            #  Set the style and content layers
            self.style_layers = []
            self.content_layers = []
            self.modify_layers(content_layers, style_layers)
            
        else:
            self.modify_layers(layers[0], layers[1])
    
    def run_style_transfer(self, content_path, style_paths, spatial_mask=None, imsize=128, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Executes style transfer on given image with given style images and masks.

        Parameters
        ----------
        content_path : str
            The path to the image on which the style transfer shall be executed
        style_paths : list of str
            The paths to the images which style shall be transfered
        spatial_mask : list torch.tensor, optional
            The masks which determine in which area of the image which masks shall be applied (default is None)
        imsize : int, optional
            The image size (default is 128)
        num_steps : int, optional
            The number of steps which shall be executed for the style transfer (default is 300)
        style_weight : int, optional
            The weight of how much the loss in style transfer contributes to the overall loss (default is 1000000)
        content_weight : int, optional
            The weight of how much the loss in content preservation contributes to the overall loss (default is 1)
        """
        
        image_loader = get_image_loader(imsize, self.device)

        style_img = [image_loader(img_pth) for img_pth in style_paths]
        content_img = image_loader(content_path) 

        input_img = content_img.clone()

        for k in range(len(style_img)):
            assert style_img[k].size() == content_img.size()
        
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses_lists(style_img, content_img,
                                                                                    spatial_mask=spatial_mask)
        
        optimizer = optim.LBFGS([input_img.requires_grad_()])

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

    def get_style_model_and_losses_lists(self, style_img, content_img, spatial_mask=None):
        """Returns the model, content and style losses with implemented receptive averaging.

        Parameters
        ----------
        style_img : list of torch.tensor
            The images whose style shall be transfered
        content_img : torch.tensor
            The image on which the style transfer shall be executed
        spatial_mask : list torch.tensor, optional
            The masks which determine in which area of the image which masks shall be applied (default is None)
        """
    
        if spatial_mask is None:
            raise ValueError("A spatial mask must be provided")

        if len(spatial_mask) != len(style_img):
            raise ValueError("Number of spatial masks and number of style images must be equal")

        style_img = torch.stack(style_img).squeeze(1)

        cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = Normalization().to(self.device)

        # just in order to have an iterable access to or list of content/syle losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        # A list with the spatial masks for the style loss
        receptive_masks = []
        layer_mask = spatial_mask
        receptive_masks.append(layer_mask)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)

                r_avg = nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
                layer_mask = r_avg(layer_mask)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)

                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

                r_max = nn.MaxPool2d(layer.kernel_size, layer.stride, layer.padding)

                layer_mask = r_max(layer_mask)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:

                target_feature = model(style_img).detach()

                style_loss = StyleLoss(target_feature, layer_mask)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]

        return model, style_losses, content_losses
    
    def modify_layers(self, content_layers, style_layers):
        """Sets up which convolutional layers in the network which are used for content and style loss calculation.

        Parameters
        ----------
        content_layers : list of str
            The list of which convolutional layers shall be regarded in the content loss calculation
        style_layers : list of str
            The list of which convolutional layers shall be regarded in the style loss calculation

        Raises
        ------
        ValueError
            If any of specified content or style layers is not in the network
        """
        
        if any([l not in self.tractable_layer_names for l in content_layers]):
            raise ValueError("One of content layers not in network")
        if any([l not in self.tractable_layer_names for l in style_layers]):
            raise ValueError("One of style layers not in network")
        
        self.content_layers = content_layers
        self.style_layers = style_layers
        
    def get_tractable_layer_names(self):
        """Returns a set of strings which give information which layers in the network
        can be  used for style transfer """

        names = set()
        
        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                names.add(name)
            
        return names
