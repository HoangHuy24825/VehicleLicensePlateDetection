# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog as tkFileDialog
import os
import cv2
import numpy as np
from tensorflow import keras

IMAGE_SIZE = 224
IMAGE_SIZE_VIEW = 500
dir_path = os.path.dirname(os.path.realpath(__file__))


def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)

        # Resize Image
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # crate test
        image_prediction = [np.array(image)]

        # Transforming in array
        image_prediction = np.array(image_prediction)

        # Renormalisation
        image_prediction = image_prediction / 255

        # Load model
        model = keras.models.load_model(os.path.join(dir_path, "model"))

        # Prediction
        y_cnn = model.predict(image_prediction)

        ny = y_cnn[0] * 255
        image_after_prediction = cv2.rectangle(image_prediction[0], (int(ny[0]), int(ny[1])), (int(ny[2]), int(ny[3])),
                                               (0, 255, 0))

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        image_after_prediction = Image.fromarray((image_after_prediction * 255).astype(np.uint8))

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        image_after_prediction = ImageTk.PhotoImage(image_after_prediction)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=image_after_prediction)
            panelB.image = image_after_prediction
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=image_after_prediction)
            panelA.image = image
            panelB.image = image_after_prediction
        # initialize the window toolkit along with the two image panels


root = Tk()
panelA = None
panelB = None

width = 600  # Width
height = 300  # Height
screen_width = root.winfo_screenwidth()  # Width of the screen
screen_height = root.winfo_screenheight()  # Height of the screen

# Calculate Starting X and Y coordinates for Window
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)

root.geometry('%dx%d+%d+%d' % (width, height, x, y))

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
root.title("Vehicle License Plate Detection")
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
