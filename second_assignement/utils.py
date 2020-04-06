import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from display_caronthehill import *
import imageio


def save_GIF(ht, name="trajectory"):
    """
    Save an animation of the trajectory ht in a gif format in the current directory
    :param ht: a list representing the trajectory to be displayed
    :param name: the name of the gif file
    :return:
    """
    # Generation of images
    counter = 0
    images = []
    for e in range(0, len(ht), 3):
        p = ht[e][0]
        s = ht[e][1]
        save_caronthehill_image(p, s, "image\\state" + str(counter) + ".png")
        images.append(imageio.imread("image\\state" + str(counter) + ".png"))
        counter += 1
    imageio.mimsave("{}.gif".format(name), images)

