# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:12:51 2019

@author: AmP
"""


import cv2
import numpy as np
 


def merge_pics(fg, bg=None, fg_opa=1):
    
    def add_alpha_channel(img):
        b = img[:, :, 0].astype(float)/255
        g = img[:, :, 1].astype(float)/255
        r = img[:, :, 2].astype(float)/255
        o = np.zeros(b.shape)
        o[:] = 1.
        img_out = cv2.merge((b, g, r, o))
        return img_out
    
    # Read the images
    b = fg[:, :, 0].astype(float)/255
    ones = np.zeros(b.shape)
    ones[:] = 1.
    foreground = add_alpha_channel(fg*(1-fg_opa))
    
    if type(bg) is not type(None):
        if bg.shape[2] == 3:
            background = add_alpha_channel(bg)
        elif bg.shape[2] == 4:
            background = bg
        else:
            raise NotImplementedError
    else:
        shape = foreground.shape
        background = np.zeros(shape)

    # alpha mask
    tmp = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 150, 255, cv2.THRESH_BINARY)
    
    
    ## Normalize the alpha mask to keep intensity between 0 and 1
    mask = mask.astype(float)/255
    mask = cv2.merge((mask, mask, mask, mask))
    
    # 
    ## Multiply the foreground with the alpha matte
    foreground = cv2.multiply(1-mask, foreground)
    #foreground = cv2.add(foreground, mask)
    # 
    ## Multiply the background with ( 1 - alpha )
    background = cv2.multiply(mask, background)
    # 
    ## Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    return outImage



mode = 'straight_1'
bg = cv2.imread('pics/'+mode+'/'+mode+'_(1).jpg', 1)
fg = cv2.imread('pics/'+mode+'/'+mode+'_(2).jpg', 1)


outImage = merge_pics(bg, fg_opa=.0001)

outImage = merge_pics(fg, outImage, fg_opa=.9)


# Display image
#cv2.imshow("out", outImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite('test.png', outImage*255)
