import torch
from math import ceil, floor


def get_circular_mask(imsize, center_x, center_y, radius):
    """Returns a mask in form of a PyTorch tensor which has ones in the circle at the specified position.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    center_x : int
        The x position of the center of the circle
    center_y : int
        The y position of the center of the circle
    radius : int
        The radius of the circle    
    """
    
    mask = torch.zeros(imsize, imsize)
    
    for x in range(imsize):
        for y in range(imsize):
            if (x - center_x)**2 + (y - center_y)**2 <= (radius**2):
                # The first dim of a tensor is vertical and the second horizontal
                mask[y, x] = 1
                
    return mask


def get_corner_mask(imsize, corner='upper_right'):
    """Returns a mask in form of a PyTorch tensor which has ones in the corner specified by 'corner'.
    
    If imagesize is odd, the upper right corner will have size: ceil(imsize/2) by ceil(imsize/2).
    
    Parameters
    ----------
    imsize : int
        The size of the image
    corner : str
        The corner in which to apply the mask:  'upper_right', 'upper_left',  'lower_left', 'lower_right'
    """
    
    m1 = torch.ones((ceil(imsize/2), ceil(imsize/2)))
    m2 = torch.zeros((ceil(imsize/2), ceil(imsize/2)))
    
    mask1 = torch.cat((m2, m1), dim=1)
    # if imsize is odd, need to cut off first column to match dimension
    if imsize % 2 == 1:
        mask1 = mask1.narrow(1, 1, imsize)
    
    mask2 = torch.zeros((floor(imsize/2), imsize))

    mask = torch.cat((mask1, mask2), dim=0)

    # Rotate to the right corner
    if corner == 'upper_right':
        pass
    elif corner == 'upper_left':
        mask = mask.flip(dims=1)
    elif corner == 'lower_left':
        mask = mask.transpose(0, 1)
    elif corner == 'lower_right':
        mask = mask.flip(dims=0)
    else:
        raise ValueError(f'Corner: {corner} not supported')

    return mask


def get_side_mask(imsize, side='left'):
    """Returns a mask in form of a PyTorch tensor which has ones on the side specified by 'side'.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    side : str
        The side on which to apply the mask: 'left', 'right', 'up', 'down'
    """
    
    mask1 = torch.ones(imsize, ceil(imsize/2))
    mask2 = torch.zeros(imsize, ceil(imsize/2))

    mask = torch.cat((mask1, mask2), dim=1)

    if side == 'left':
        pass
    elif side == 'right':
        mask = mask.flip(dims=1)
    elif side == 'up':
        mask = mask.transpose(0, 1)
    elif side == 'down':
        mask = mask.transpose(0, 1).flip(dims=0)
    else:
        raise ValueError(f'Side: {side} not supported')
    
    return mask

    
def get_elliptic_mask(imsize, center_x, center_y, width, height):
    """Returns a mask in form of a PyTorch tensor which has ones in the specified ellipse.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    center_x : int
        The x position of the center of the ellipse
    center_y : int
        The y position of the center of the ellipse
    width : int
        The width of the ellipse
    height : int
        The height of the ellipse
    """
    
    mask = torch.zeros(imsize, imsize)
    
    for x in range(imsize):
        for y in range(imsize):
            if ((x-center_x)/(width/2))**2 + ((y-center_y)/(height/2))**2 <= 1:
                mask[y, x] = 1
    
    return mask


def get_rectangular_mask(imsize, upper_left_point_x, upper_left_point_y, length_x, length_y):
    """Returns a mask in form of a PyTorch tensor which has ones in the specified rectangle.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    upper_left_point_x : int
        The x position of the upper left corner of the rectangle
    upper_left_point_y : int
        The y position of the upper left corner of the rectangle
    length_x : int
        The length along the x-axis of the rectangle
    length_y : int
        The length along the y-axis of the rectangle
    """

    mask = torch.zeros(imsize, imsize)

    for x in range(upper_left_point_x, upper_left_point_x+length_x):
        for y in range(upper_left_point_y, upper_left_point_y+length_y):
            mask[y, x] = 1  # since first dimension is x-axis and second dimension is y-axis
    
    return mask
