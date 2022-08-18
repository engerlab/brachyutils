"""
Date 
        2021/9/9
Purpose
        a function to generate a scrollable plot of 3D dose matricies. scrolling happens along the third axis (z-axis)
Author
        Hossein Jafarzadeh
            Enger Lab
            McGill University
        code is mostly taken from "https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html"

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
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, dose, fileType)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    return fig

    
class IndexTracker(object):
    def __init__(self, ax, X, fileType):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate ' + fileType + ' images.')

        self.X = X 
        self.slices, rows, cols = X.shape 
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :])
        plt.colorbar(self.im, use_gridspec = True)
        self.update()
    
    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind -1) % self.slices
        self.update()

