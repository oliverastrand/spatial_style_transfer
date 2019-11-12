import torch
import math

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
    
    for i in range(imsize):
        for j in range(imsize):
            if (i - center_x)**2 + (j - center_y)**2 <= (radius**2):
                mask[j,i] = 1
                
    return mask

def get_upper_right_corner_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the upper right corner.
    
    If imagesize is odd, the upper right corner will have size: math.ceil(imsize/2) by math.ceil(imsize/2).
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
    
    m1 = torch.ones(( math.ceil(imsize/2) , math.ceil(imsize/2) ))
    m2 = torch.zeros(( math.ceil(imsize/2), math.ceil(imsize/2) ))
    
    mask1 = torch.cat((m2, m1), dim=1)
    # if imsize is odd, need to cut off first column to match dimension
    if imsize % 2 ==1:
        mask1=mask1.narrow(1,1,imsize)
    
    mask2 = torch.zeros(( math.floor(imsize/2) , imsize ))
    
    return torch.cat((mask1,mask2),dim=0)

def get_upper_left_corner_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the upper left corner.
    
    If imagesize is odd, the upper left corner will have size: math.ceil(imsize/2) by math.ceil(imsize/2).
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
            
    m1 = torch.ones(( math.ceil(imsize/2) , math.ceil(imsize/2) ))
    m2 = torch.zeros(( math.ceil(imsize/2), math.ceil(imsize/2) ))
    
    mask1 = torch.cat((m1, m2), dim=1)
    # if imsize is odd, need to cut off last column to match dimension
    if imsize % 2 == 1:
        mask1 = mask1.narrow(1,0,imsize)
    
    mask2 = torch.zeros(( math.floor(imsize/2) , imsize ))
    
    return torch.cat((mask1,mask2),dim=0)


def get_lower_left_corner_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the lower left corner.
    
    If imagesize is odd, the lower left corner will have size: math.ceil(imsize/2) by math.ceil(imsize/2).
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
    
    m1 = torch.ones(( math.ceil(imsize/2) , math.ceil(imsize/2) ))
    m2 = torch.zeros(( math.ceil(imsize/2), math.ceil(imsize/2) ))
    
    mask1 = torch.cat((m1, m2), dim=1)
    # if imsize is odd, need to cut off last column to match dimension
    if imsize % 2 == 1:
        mask1 = mask1.narrow(1, 0, imsize)
    
    mask2 = torch.zeros(( math.floor(imsize/2) , imsize ))

    return torch.cat((mask2,mask1),dim=0)

def get_lower_right_corner_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the lower right corner.
    
    If imagesize is odd, the lower right corner will have size: math.ceil(imsize/2) by math.ceil(imsize/2).
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
        
    m1 = torch.ones(( math.ceil(imsize/2) , math.ceil(imsize/2) ))
    m2 = torch.zeros(( math.ceil(imsize/2), math.ceil(imsize/2) ))
    
    mask1 = torch.cat((m2, m1), dim=1)
    # if imsize is odd, need to cut off first column to match dimension
    if imsize % 2 == 1:
        mask1 = mask1.narrow(1,1,imsize)

    mask2 = torch.zeros(( math.ceil(imsize/2) , imsize ))

    return torch.cat((mask2,mask1),dim=0)


def get_left_side_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the left side.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
    
    mask1 = torch.ones(imsize, math.ceil(imsize/2))
    mask2 = torch.zeros(imsize, math.ceil(imsize/2))
    
    return torch.cat((mask1,mask2),dim=1)

def get_right_side_mask(imsize):
    """Returns a mask in form of a PyTorch tensor which has ones in the right side.
    
    Parameters
    ----------
    imsize : int
        The size of the image
    """
    
    mask1 = torch.ones(imsize, math.ceil(imsize/2))
    mask2 = torch.zeros(imsize, math.ceil(imsize/2))
    
    return torch.cat((mask2,mask1),dim=1)
    
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
    
    for i in range(imsize):
        for j in range(imsize):
            if ((i-center_x)/(width/2))**2 + ((j-center_y)/(height/2))**2 <= 1:
                mask[j,i] = 1
    
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
    for i in range(upper_left_point_x, upper_left_point_x+length_x):
        for j in range(upper_left_point_y, upper_left_point_y+length_y):
            mask[j,i] = 1                  # since first dimension is x-axis and second dimension is y-axis
    
    return mask
