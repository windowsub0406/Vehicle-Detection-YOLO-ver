import numpy as np
import cv2

def sobel_xy(img, orient='x', thresh=(20, 100)):
    """
    Define a function that applies Sobel x or y.
    The gradient in the x-direction emphasizes edges closer to vertical.
    The gradient in the y-direction emphasizes edges closer to horizontal.
    """
    # img = exposure.equalize_hist(img)
    # adaptive histogram equalization
    # img = exposure.equalize_adapthist(img, clip_limit=0.01)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Define a function to return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    """

    # adaptive histogram equalization
    # img = exposure.equalize_adapthist(img, clip_limit=0.01)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):
    """
    computes the direction of the gradient
    """
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    # Return the binary image
    return binary_output.astype(np.uint8)

def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
    return binary

def gradient_combine(img, th_x, th_y, th_mag, th_dir):
    """
    Find lane lines with gradient information of Red channel
    """
    rows, cols = img.shape[:2]
    R = img[220:rows - 12, 0:cols, 2]

    sobelx = sobel_xy(R, 'x', th_x)
    #cv2.imshow('sobel_x', sobelx)
    sobely = sobel_xy(R, 'y', th_y)
    #cv2.imshow('sobel_y', sobely)
    mag_img = mag_thresh(R, 3, th_mag)
    #cv2.imshow('sobel_mag', mag_img)
    dir_img = dir_thresh(R, 15, th_dir)
    #cv2.imshow('result5', dir_img)

    # combine gradient measurements
    gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
    gradient_comb[((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))] = 255

    return gradient_comb

def hls_combine(img, th_h, th_l, th_s):
    # convert to hls color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2]
    #R = img[220:rows - 12, 0:cols, 2]
    #_, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    #cv2.imshow('red!!!',R)
    H = hls[220:rows - 12, 0:cols, 0]
    L = hls[220:rows - 12, 0:cols, 1]
    S = hls[220:rows - 12, 0:cols, 2]

    h_img = ch_thresh(H, th_h)
    #cv2.imshow('HLS (H) threshold', h_img)
    l_img = ch_thresh(L, th_l)
    #cv2.imshow('HLS (L) threshold', l_img)
    s_img = ch_thresh(S, th_s)
    #cv2.imshow('HLS (S) threshold', s_img)

    # Two cases - lane lines in shadow or not
    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 # | (R > 1)] = 255
    #hls_comb[((s_img > 1) & (h_img > 1)) | (R > 1)] = 255
    return hls_comb

def comb_result(grad, hls):
    """ give different value to distinguish them """
    result = np.zeros_like(hls).astype(np.uint8)
    #result[((grad > 1) | (hls > 1))] = 255
    result[(grad > 1)] = 100
    result[(hls > 1)] = 255

    return result
