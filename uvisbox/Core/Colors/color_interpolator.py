from skimage.color import rgb2lab, lab2rgb

def interpolate_lab(rgb1, rgb2, v, v0, v1):
    lab1 = rgb2lab(rgb1)
    lab2 = rgb2lab(rgb2)
    return lab2rgb(lab1 + (lab2 - lab1) * (v - v0) / (v1 - v0))