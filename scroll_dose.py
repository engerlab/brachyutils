"""
Date 
        2021/9/9
Purpose
        a function to generate a scrollable plot of 3D dose matricies. scrolling happens along the third axis (z-axis)
Author
        Hossein Jafarzadeh
            Enger Lab
            McGill University
Inputs
    1. "dose": 3D numpy matrix containing dose values
    2. "fileType": a string that describes the source of the dose (simulate vs dicom vs measured)

Dependencies
    1. numpy
    2. matplotlib.pyplot

Outputs
    a scrollable plot of the 3D dose
"""
import numpy as np
import matplotlib.pyplot as plt
# a function to scroll through dose slices: one can improve this section by adding units to axis and dose value
def plot_scrollable(dose, fileType):
    print("------------------")
    scroll = input("whould u like to scroll through " + fileType +" files? yes, or no \n")
    if scroll=="yes":
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, dose)

        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        return fig
            
    elif scroll=="no":
            print("------------------")
            print("not scrolling through the 3ddose file and moving on")
            return 0
    else:
            print("------------------")
            print("invalid input, rerun the script and pick either yes or no")
            return 0

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X 
        rows, cols, self.slices = X.shape 
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()
    
    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind -1) % self.slices
        self.update()

