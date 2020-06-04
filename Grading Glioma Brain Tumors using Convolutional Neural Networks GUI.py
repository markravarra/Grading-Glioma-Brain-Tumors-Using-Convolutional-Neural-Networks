LARGE_FONT = ("Verdana", 12)
SMALL_FONT = ("Verdana", 8)
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, StringVar, messagebox # CSS for tkinter
import sys
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

from keras.preprocessing.image import ImageDataGenerator  # used for data augmentation (helps prevent overfitting)
from keras.optimizers import Adam  # imports the Adam  optimizer, the optimizer method used to train our network.
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer  # transform a class label string to an integer and vice versa.
from sklearn.model_selection import train_test_split  # Used to create training and testing splits
from cnnmastercode.smallervggnet import SmallerVGGNet
from keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import imageio
import imutils
from pathlib import Path

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import itertools


class PracClass(tk.Tk):
    def __init__(self, *args, **kwargs): # args - pass variables, kwargs - pass dictionaries
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="icon.ico")
        tk.Tk.wm_title(self, "GRADING GLIOMA BRAIN TUMORS USING CNN")

        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=5)
        container.grid_columnconfigure(0, weight=5)

        self.frames = {}

        for F in (StartPage, Detect, ClassifyImg, BatchClassify, PreProcessImg, Training):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise() # raise it to the front

def qf(param):
    print(param)

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="MAIN MENU", font = LARGE_FONT) # initialize the label
        label.grid(row=0, pady=10, padx=10, sticky="ew")

        button3 = ttk.Button(self, text="Preprocessing",
                             command=lambda: controller.show_frame(PreProcessImg))
        button3.state(['disabled'])
        button3.grid(row=1, ipadx=200, padx=10, sticky="ew")

        button4 = ttk.Button(self, text="Training Phase",
                             command=lambda: controller.show_frame(Training))
        button4.state(['disabled'])
        button4.grid(row=2, ipadx=200, padx=10, sticky="ew")

        button1A = ttk.Button(self, text="Detection",
                              command=lambda: controller.show_frame(Detect))
        button1A.grid(row=3, ipadx=200, padx=10, sticky="ew")

        button1 = ttk.Button(self, text="Classify an Image",
                            command=lambda: controller.show_frame(ClassifyImg))
        button1.grid(row=4, ipadx=200, padx=10, sticky="ew")

        button2 = ttk.Button(self, text="BatchClassify",
                            command=lambda: controller.show_frame(BatchClassify))
        button2.state(['disabled'])
        button2.grid(row=5, ipadx=200, padx=10, sticky="ew")

class ClassifyImg(tk.Frame):
    def __init__(self, parent, controller):
        # global panelA, panelB, panelC

        tk.Frame.__init__(self, parent)
        self.txtname = StringVar(None)
        label = tk.Label(self, text="CLASSIFY AN IMAGE", font=LARGE_FONT)  # initialize the label
        label.grid(padx=10, pady=10, columnspan=2)

        lbl = tk.Label(self, text="Pick Image File: ")
        lbl.grid(column=0, row=1, sticky="n")
        txt = tk.Entry(self, textvariable=self.txtname, width=20)
        txt.grid(column=1, row=1, sticky="we")
        txt.focus()  # Focus the cursor on the textbox
        button0 = ttk.Button(self, text="Open File",
                             command=lambda: self.inFile())
        button0.grid(column=2, row=1, padx=(5,5), sticky="ew")

        self.txt1 = scrolledtext.ScrolledText(self, width=40, height=5)
        self.txt1.grid(column=0, row=3, columnspan=2, padx=(10,0) , sticky="we")
        # Set scrolled text content
        # txt1.insert(tk.INSERT, "Your text goes hear")
        # sys.stdout = self

        button1 = ttk.Button(self, text="Back",
                            command=lambda: [self.ClearOut(),controller.show_frame(StartPage)])
        button1.grid(column=2, row=3, padx=(5,5), sticky="SEW")
        # button2 = ttk.Button(self, text="Batch Classification",
        #                     command=lambda: [self.ClearOut(),controller.show_frame(BatchClassify)])
        # button2.grid(column=0, row=2, sticky="ew")

        button2_1 = ttk.Button(self, text="Clear Outputs",
                             command=lambda: self.ClearOut())
        button2_1.grid(column=1, row=2, sticky="SEW")

        button3 = ttk.Button(self, text="Classify Image",
                             command=lambda: [self.ClearOut(),self.ClassifyImage(txt.get())])
        button3.grid(column=2, row=2, rowspan=3, padx=(5,5), sticky="ew")

        label_orig = tk.Label(self, text="Original", font=SMALL_FONT)  # initialize the label
        label_orig.grid(column=0, row=4, sticky="n")
        label_grey = tk.Label(self, text="Denoised", font=SMALL_FONT)  # initialize the label
        label_grey.grid(column=1, row=4, sticky="n")
        label_res = tk.Label(self, text="Classification", font=SMALL_FONT)  # initialize the label
        label_res.grid(column=2, row=4, sticky="n")

        self.panelA = None
        self.panelB = None
        self.panelC = None
        # panelA.grid(column=0, row=4)
        # panelB.grid(column=1, row=4)
        # panelC.grid(column=2, row=4)

    def write(self, txt):
        self.txt1.insert(tk.END, str(txt))
        #######This is the missing line!!!!:
        self.update_idletasks()

    def flush(self):
        pass

    def inFile(self):
        filename = filedialog.askopenfilename(title='Choose a file')
        self.txtname.set(filename)

    def ClearOut(self):
        try:
            # self.txtname.set("")
            self.txt1.delete(1.0, tk.END)
            self.panelA.destroy()
            self.panelB.destroy()
            self.panelC.destroy()
            self.panelA = None
            self.panelB = None
            self.panelC = None
        except:
            pass
    def ShowImages(self, fPath, result):
        # grab a reference to the image panels

        # open a file chooser dialog and allow the user to select an input
        # image
        path = fPath

        # ensure a file path was selected
        if len(path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            image = cv2.imread(path)
            denoised = cv2.medianBlur(image, 7)

            image = cv2.resize(image, (256, 256))
            denoised = cv2.resize(denoised, (256, 256))
            result = cv2.resize(result, (256, 256))

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # convert the images to PIL format...
            image = Image.fromarray(image)
            denoised = Image.fromarray(denoised)
            result = Image.fromarray(result)

            # ...and then to ImageTk format
            image = ImageTk.PhotoImage(image)
            denoised = ImageTk.PhotoImage(denoised)
            result = ImageTk.PhotoImage(result)

            # if the panels are None, initialize them
            if self.panelA is None or self.panelB is None or self.panelC is None:
                # the first panel will store our original image
                self.panelA = tk.Label(self, image=image)
                self.panelA.image = image
                self.panelA.grid(row=5, column=0)

                # while the second panel will store the edge map
                self.panelB = tk.Label(self, image=denoised)
                self.panelB.image = denoised
                self.panelB.grid(row=5, column=1)

                self.panelC = tk.Label(self, image=result)
                self.panelC.image = result
                self.panelC.grid(row=5, column=2)

            # otherwise, update the image panels
            else:
                # update the pannels
                self.panelA.configure(image=image)
                self.panelB.configure(image=denoised)
                self.panelC.configure(image=result)
                self.panelA.image = image
                self.panelB.image = denoised
                self.panelC.image = result

    def ClassifyImage(self, fileN):
        # import the necessary packages
        from keras.preprocessing.image import img_to_array
        from keras.models import load_model
        from sklearn import preprocessing
        import numpy as np
        import argparse
        import imutils
        import pickle
        import cv2
        import os
        file = fileN
        print(file)
        # load the image
        image = cv2.imread(file)
        outimg = image.copy()
        # image = cv2.medianBlur(image, 7)

        # pre-process the image for classification
        image = cv2.resize(image, (128, 128))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network and the label
        # binarizer
        print("[INFO] loading network...")
        model = load_model("Model Version 02-15-2020_132449")
        lb = pickle.loads(open("Labelbin Model Version 02-15-2020_132449.pickle", "rb").read())
        print(lb)
        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        print(proba)
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        print(label)

        # we'll mark our prediction as "correct"if the input image filename
        # contains the predicted label text (obviously this makes the
        # assumption that you have named your testing image files this way)
        filename = file[file.rfind(os.path.sep) + 1:]
        print(filename)
        correct = "correct" if filename.rfind(label) != -1 else "incorrect"

        # build the label and draw the label on the image
        label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
        outimg = imutils.resize(outimg, width=400)
        if correct == "correct":
            cv2.putText(outimg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        elif correct =="incorrect":
            cv2.putText(outimg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

        # show the output image
        print("[INFO] {}".format(label))
        # outResult = cv2.imshow("Output", outimg)
        image = cv2.resize(outimg, (256, 256))
        outResult = cv2.imwrite("Result.jpg", image)
        imgResult = cv2.imread("Result.jpg")
        self.ShowImages(file, imgResult)
        # cv2.waitKey(0)

class Detect(tk.Frame):
    def __init__(self, parent, controller):
        # global panelA, panelB, panelC

        tk.Frame.__init__(self, parent)
        self.txtname = StringVar(None)
        label = tk.Label(self, text="DETECT", font=LARGE_FONT)  # initialize the label
        label.grid(padx=10, pady=10, columnspan=2)

        lbl = tk.Label(self, text="Pick Image File: ")
        lbl.grid(column=0, row=1, sticky="n")
        txt = tk.Entry(self, textvariable=self.txtname, width=20)
        txt.grid(column=1, row=1, sticky="we")
        txt.focus()  # Focus the cursor on the textbox
        button0 = ttk.Button(self, text="Open File",
                             command=lambda: self.inFile())
        button0.grid(column=2, row=1, padx=(5,5), sticky="ew")

        self.txt1 = scrolledtext.ScrolledText(self, width=40, height=5)
        self.txt1.grid(column=0, row=3, columnspan=2, padx=(10,0) , sticky="we")
        # Set scrolled text content
        # txt1.insert(tk.INSERT, "Your text goes hear")
        sys.stdout = self

        button1 = ttk.Button(self, text="Back",
                            command=lambda: [self.ClearOut(),controller.show_frame(StartPage)])
        button1.grid(column=2, row=3, padx=(5,5), sticky="SEW")
        # button2 = ttk.Button(self, text="Batch Classification",
        #                     command=lambda: [self.ClearOut(),controller.show_frame(BatchClassify)])
        # button2.grid(column=0, row=2, sticky="ew")

        button2_1 = ttk.Button(self, text="Clear Outputs",
                             command=lambda: self.ClearOut())
        button2_1.grid(column=1, row=2, sticky="SEW")

        button3 = ttk.Button(self, text="Detect",
                             command=lambda: [self.ClearOut(),self.Detect(txt.get())])
        button3.grid(column=2, row=2, rowspan=3, padx=(5,5), sticky="ew")

        label_orig = tk.Label(self, text="Original", font=SMALL_FONT)  # initialize the label
        label_orig.grid(column=0, row=4, sticky="n")
        label_grey = tk.Label(self, text="Denoised", font=SMALL_FONT)  # initialize the label
        label_grey.grid(column=1, row=4, sticky="n")
        label_res = tk.Label(self, text="Detection", font=SMALL_FONT)  # initialize the label
        label_res.grid(column=2, row=4, sticky="n")

        self.panelA = None
        self.panelB = None
        self.panelC = None
        # panelA.grid(column=0, row=4)
        # panelB.grid(column=1, row=4)
        # panelC.grid(column=2, row=4)

    def write(self, txt):
        self.txt1.insert(tk.END, str(txt))
        #######This is the missing line!!!!:
        self.update_idletasks()

    def flush(self):
        pass

    def inFile(self):
        filename = filedialog.askopenfilename(title='Choose a file')
        self.txtname.set(filename)

    def ClearOut(self):
        try:
            # self.txtname.set("")
            self.txt1.delete(1.0, tk.END)
            self.panelA.destroy()
            self.panelB.destroy()
            self.panelC.destroy()
            self.panelA = None
            self.panelB = None
            self.panelC = None
        except:
            pass
    def ShowImages(self, fPath, result):
        # grab a reference to the image panels

        # open a file chooser dialog and allow the user to select an input
        # image
        path = fPath

        # ensure a file path was selected
        if len(path) > 0:
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            image = cv2.imread(path)
            denoised = cv2.medianBlur(image, 7)

            image = cv2.resize(image, (256, 256))
            denoised = cv2.resize(denoised, (256, 256))
            result = cv2.resize(result, (256, 256))

            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # convert the images to PIL format...
            image = Image.fromarray(image)
            denoised = Image.fromarray(denoised)
            result = Image.fromarray(result)

            # ...and then to ImageTk format
            image = ImageTk.PhotoImage(image)
            denoised = ImageTk.PhotoImage(denoised)
            result = ImageTk.PhotoImage(result)

            # if the panels are None, initialize them
            if self.panelA is None or self.panelB is None or self.panelC is None:
                # the first panel will store our original image
                self.panelA = tk.Label(self, image=image)
                self.panelA.image = image
                self.panelA.grid(row=5, column=0)

                # while the second panel will store the edge map
                self.panelB = tk.Label(self, image=denoised)
                self.panelB.image = denoised
                self.panelB.grid(row=5, column=1)

                self.panelC = tk.Label(self, image=result)
                self.panelC.image = result
                self.panelC.grid(row=5, column=2)

            # otherwise, update the image panels
            else:
                # update the pannels
                self.panelA.configure(image=image)
                self.panelB.configure(image=denoised)
                self.panelC.configure(image=result)
                self.panelA.image = image
                self.panelB.image = denoised
                self.panelC.image = result

    def Detect(self, fileN):
        # import the necessary packages
        from keras.preprocessing.image import img_to_array
        from keras.models import load_model
        from sklearn import preprocessing
        import numpy as np
        import argparse
        import imutils
        import pickle
        import cv2
        import os
        file = fileN
        print(file)
        # load the image
        image = cv2.imread(file)
        outimg = image.copy()
        # image = cv2.medianBlur(image, 7)

        # pre-process the image for classification
        image = cv2.resize(image, (128, 128))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network and the label
        # binarizer
        print("[INFO] loading network...")
        model = load_model("Model Version 02-15-2020_132449")
        lb = pickle.loads(open("Labelbin Model Version 02-15-2020_132449.pickle", "rb").read())
        print(lb)
        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        print(proba)
        idx = np.argmax(proba)
        comp_label = lb.classes_[idx]

        label = lb.classes_[idx]
        if label == "G1":
            label = "With Glioma"
        elif label == "G2":
            label = "With Glioma"
        elif label == "G3":
            label = "With Glioma"
        elif label == "G4":
            label = "With Glioma"
        elif label == "G0":
            label = "Normal MRI"
        print("IDX: ",idx)
        print("LABELS : ", label)

        # we'll mark our prediction as "correct"if the input image filename
        # contains the predicted label text (obviously this makes the
        # assumption that you have named your testing image files this way)
        filename = file[file.rfind(os.path.sep) + 1:]
        print(filename)
        correct = "correct" if filename.rfind(label) != -1 else "incorrect"


        # build the label and draw the label on the image
        label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
        outimg = imutils.resize(outimg, width=400)
        if correct == "correct":
            cv2.putText(outimg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(outimg, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            
        # show the output image
        print("[INFO] {}".format(label))
        # outResult = cv2.imshow("Output", outimg)
        image = cv2.resize(outimg, (256, 256))
        outResult = cv2.imwrite("Result.jpg", image)
        imgResult = cv2.imread("Result.jpg")
        self.ShowImages(file, imgResult)
        # cv2.waitKey(0)

class BatchClassify(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="BATCH CLASSIFICATION", font=LARGE_FONT)  # initialize the label
        label.grid(row=0, pady=10, padx=10, sticky="ew")

        # self.BCL()
        self.button0 = ttk.Button(self, text="Classify Test Set", command=lambda: [self.ShowBCL()])
        self.button0.grid(row=1, ipadx=200, padx=10, sticky="ew")
        labelsep = tk.Label(self, text="", font=LARGE_FONT)
        labelsep.grid(row=2, pady=10, padx=10, sticky="ew")
        labelsep1 = tk.Label(self, text="", font=LARGE_FONT)
        labelsep1.grid(row=3, pady=10, padx=10, sticky="ew")
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: [controller.show_frame(StartPage), self.ClearOut()])
        button1.grid(row=5, column=0, sticky="e")

    def ClearOut(self):
        try:
            # self.txtname.set("")
            self.title.destroy()
            self.filler.destroy()
            self.class_rep.destroy()
            self.showCM.destroy()
        except:
            pass

    def BCL(self):
        # from keras.models import load_model
        # from sklearn.metrics import classification_report, confusion_matrix
        # import itertools

        IMAGE_DIMS = (128, 128, 3)  # Supplying the spatial dimensions of our input images.
        # In this case we are using 96 x 96 pixels with 3 channels (RGB)
        # For the Thesis, Thesis we will be using, ## x ## with 2 channels (Grayscale)

        # initialize the data and labels
        Testdata = []
        Testlabels = []

        # grab the image paths and randomly shuffle them
        print("[INFO] loading images...")
        dataset = "TEST SETV2"
        imagePaths = sorted(list(paths.list_images(dataset)))
        random.seed(42)
        random.shuffle(imagePaths)

        # loop over the input images
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            # image = cv2.medianBlur(image, 7)
            image = img_to_array(image)
            Testdata.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]  # os.path.sep is for the windows pathname separator "\"
            Testlabels.append(label)


        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(Testdata, dtype="float") / 255.0
        labels = np.array(Testlabels)
        print("[INFO] data matrix: {:.2f}MB".format(
            data.nbytes / (1024 * 1000.0)))

        # binarize the labels
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        print("[INFO] loading network...")
        modelPath = "Model Version 02-15-2020_132449"
        model = load_model(modelPath)

        predictions = model.predict(data, batch_size=10, verbose=0)
        self.rounded_predictions = model.predict_classes(data, batch_size=10, verbose=0)
        self.y_label = np.argmax(labels, axis=1)
        self.cm = confusion_matrix(self.y_label, self.rounded_predictions)

        self.cm_plot_labels = ["GRADE 0", "GRADE 1", "GRADE 2", "GRADE 3", "GRADE 4"]

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix',
                              cmap=plt.cm.Blues):
        # this function prints the confusion matrix

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        # true labels on y axis
        plt.ylabel("True Label")

        # predicted labels on x axis
        plt.xlabel("Predicted Label")

        plt.ylim(4.5, -0.5)
        plt.tight_layout()
        plt.show()

    # plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")

    # buttons

    def ShowBCL(self):
        from sklearn.metrics import classification_report

        self.BCL()
        self.button0.destroy()
        self.showCM = ttk.Button(self, text="View Confusion Matrix",
                        command=lambda: self.plot_confusion_matrix(self.cm, self.cm_plot_labels, title="Confusion Matrix"), width=40)
        # label
        self.title = tk.Label(self, text="Classification Report", font=('Helvetica', 20, 'bold'))
        self.class_rep = tk.Text(self, height=15, width=65)
        self.class_rep.insert(tk.END, classification_report(self.y_label, self.rounded_predictions))
        self.class_rep.configure(state='disabled')
        self.filler = tk.Label(self, text="", padx=50)
        # grid
        self.title.grid(column=0, row=1, padx=10, sticky="NSEW")
        # self.filler.grid(column=0, row=2)
        self.class_rep.grid(column=0, row=2, padx=20, sticky="NEWS")
        self.showCM.grid(column=0, row=3, pady=20)

class PreProcessImg(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="PREPROCESSING", font=LARGE_FONT)  # initialize the label
        label.grid(pady=10, padx=10, columnspan=2)

        # textbox
        e = tk.Entry(self, width=35, borderwidth=3)

        # Buttons
        chooseImgFolder = ttk.Button(self, text="Choose Image Folder", command=lambda: self.open_file(e))
        preprocess = ttk.Button(self, text="Start Preprocessing", command=lambda: self.preprocess(e))

        e.grid(row=1, column=1, padx=(80,10), pady=10, columnspan=2, sticky="ew")
        chooseImgFolder.grid(row=1, column=3, columnspan=2, sticky="ew")
        preprocess.grid(row=2, column=2, columnspan=3, sticky="ew")

        button1 = ttk.Button(self, text="Back",
                             command=lambda: [ self.ClearOut(), controller.show_frame(StartPage)])
        button1.grid(row=1, column=5, rowspan=3, sticky="NS", padx=(15,10))

    def open_file(self, e):
        inpath = filedialog.askdirectory()
        e.delete(0, tk.END)
        e.insert(0, inpath)
    def ClearOut(self):
        try:
            # self.txtname.set("")
            self.sep.destroy()
            self.img1Label.destroy()
            self.result0.destroy()
            self.img2Label.destroy()
            self.result2.destroy()
            self.img3Label.destroy()
            self.result3.destroy()
            self.img4Label.destroy()
            self.result4.destroy()
            self.img5Label.destroy()
            self.result5.destroy()
            self.img6Label.destroy()
            self.result6.destroy()
            self.img7Label.destroy()
            self.result7.destroy()
        except:
            pass
    def preprocess(self, e):
        try:
            inPath = os.path.join(str(Path(e.get())) + '\G0 JPG')
            print(inPath)
            outPath = os.path.join(str(Path(e.get())) + '\Denoised\G0')
        except OSError:
            if not os.path.exists(inPath):
                messagebox.showerror("Invalid File Path", "Please re-enter a valid file path or choose an image folder"
                                                          " using the Choose Image Folder Button.")
        try:
            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_denoise = cv2.imread(full_inpath, 0)
                img_median = cv2.medianBlur(mri_to_denoise, 7)  # Add median filter to image
                counter += 1

                fullpath = os.path.join(outPath, 'TEST_G0_denoised7_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath), img_median)

            inPath = os.path.join(str(Path(e.get())) + '\G1 JPG')
            outPath = str(Path(e.get())) + '\Denoised\G1'

            counter=0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_denoise = cv2.imread(full_inpath, 0)
                img_median = cv2.medianBlur(mri_to_denoise, 7)  # Add median filter to image
                counter += 1

                fullpath = os.path.join(outPath, 'TEST_G1_denoised7_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath), img_median)

            inPath = os.path.join(str(Path(e.get())) + '\G2 JPG')
            outPath = str(Path(e.get())) + '\Denoised\G2'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_denoise = cv2.imread(full_inpath, 0)
                img_median = cv2.medianBlur(mri_to_denoise, 7)  # Add median filter to image
                counter += 1

                fullpath = os.path.join(outPath, 'TEST_G2_denoised7_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath), img_median)

            inPath = os.path.join(str(Path(e.get())) + '\G3 JPG')
            outPath = str(Path(e.get())) + '\Denoised\G3'

            counter = 0

            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_denoise = cv2.imread(full_inpath, 0)
                img_median = cv2.medianBlur(mri_to_denoise, 7)  # Add median filter to image
                counter += 1

                fullpath = os.path.join(outPath, 'TEST_G3_denoised7_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath), img_median)

            inPath = os.path.join(str(Path(e.get())) + '\G4 JPG')
            outPath = str(Path(e.get())) + '\Denoised\G4'

            counter = 0

            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_denoise = cv2.imread(full_inpath, 0)
                img_median = cv2.medianBlur(mri_to_denoise, 7)  # Add median filter to image
                counter += 1

                fullpath = os.path.join(outPath, 'TEST_G4_denoised7_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath), img_median)

            inPath = str(Path(e.get())) + '\Denoised\G0'
            outPath = str(Path(e.get())) + '\Rotated\G0'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_rotate = cv2.imread(full_inpath, 0)
                counter += 1

                rotated_mri_45 = imutils.rotate_bound(mri_to_rotate, 45)
                rotated_mri_n45 = imutils.rotate_bound(mri_to_rotate, -45)
                rotated_mri_90 = imutils.rotate_bound(mri_to_rotate, 90)
                rotated_mri_180 = imutils.rotate_bound(mri_to_rotate, 180)
                rotated_mri_270 = imutils.rotate_bound(mri_to_rotate, 270)
                rotated_mri_360 = imutils.rotate_bound(mri_to_rotate, 360)

                fullpath1 = os.path.join(outPath, 'TEST_G0_rotated45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath1), rotated_mri_45)

                fullpath2 = os.path.join(outPath, 'TEST_G0_rotatedn45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath2), rotated_mri_n45)

                fullpath3 = os.path.join(outPath, 'TEST_G0_rotated90_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath3), rotated_mri_90)

                fullpath4 = os.path.join(outPath, 'TEST_G0_rotated180_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath4), rotated_mri_180)

                fullpath5 = os.path.join(outPath, 'TEST_G0_rotated270_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath5), rotated_mri_270)

                fullpath6 = os.path.join(outPath, 'TEST_G0_rotated360_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath6), rotated_mri_360)

            inPath = str(Path(e.get())) + '\Denoised\G1'
            outPath = str(Path(e.get())) + '\Rotated\G1'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_rotate = cv2.imread(full_inpath, 0)
                counter += 1

                rotated_mri_45 = imutils.rotate_bound(mri_to_rotate, 45)
                rotated_mri_n45 = imutils.rotate_bound(mri_to_rotate, -45)
                rotated_mri_90 = imutils.rotate_bound(mri_to_rotate, 90)
                rotated_mri_180 = imutils.rotate_bound(mri_to_rotate, 180)
                rotated_mri_270 = imutils.rotate_bound(mri_to_rotate, 270)
                rotated_mri_360 = imutils.rotate_bound(mri_to_rotate, 360)

                fullpath1 = os.path.join(outPath, 'TEST_G1_rotated45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath1), rotated_mri_45)

                fullpath2 = os.path.join(outPath, 'TEST_G1_rotatedn45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath2), rotated_mri_n45)

                fullpath3 = os.path.join(outPath, 'TEST_G1_rotated90_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath3), rotated_mri_90)

                fullpath4 = os.path.join(outPath, 'TEST_G1_rotated180_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath4), rotated_mri_180)

                fullpath5 = os.path.join(outPath, 'TEST_G1_rotated270_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath5), rotated_mri_270)

                fullpath6 = os.path.join(outPath, 'TEST_G1_rotated360_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath6), rotated_mri_360)

            inPath = str(Path(e.get())) + '\Denoised\G2'
            outPath = str(Path(e.get())) + '\Rotated\G2'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_rotate = cv2.imread(full_inpath, 0)
                counter += 1

                rotated_mri_45 = imutils.rotate_bound(mri_to_rotate, 45)
                rotated_mri_n45 = imutils.rotate_bound(mri_to_rotate, -45)
                rotated_mri_90 = imutils.rotate_bound(mri_to_rotate, 90)
                rotated_mri_180 = imutils.rotate_bound(mri_to_rotate, 180)
                rotated_mri_270 = imutils.rotate_bound(mri_to_rotate, 270)
                rotated_mri_360 = imutils.rotate_bound(mri_to_rotate, 360)

                fullpath1 = os.path.join(outPath, 'TEST_G2_rotated45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath1), rotated_mri_45)

                fullpath2 = os.path.join(outPath, 'TEST_G2_rotatedn45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath2), rotated_mri_n45)

                fullpath3 = os.path.join(outPath, 'TEST_G2_rotated90_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath3), rotated_mri_90)

                fullpath4 = os.path.join(outPath, 'TEST_G2_rotated180_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath4), rotated_mri_180)

                fullpath5 = os.path.join(outPath, 'TEST_G2_rotated270_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath5), rotated_mri_270)

                fullpath6 = os.path.join(outPath, 'TEST_G2_rotated360_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath6), rotated_mri_360)

            inPath = str(Path(e.get())) + '\Denoised\G3'
            outPath = str(Path(e.get())) + '\Rotated\G3'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_rotate = cv2.imread(full_inpath, 0)
                counter += 1

                rotated_mri_45 = imutils.rotate_bound(mri_to_rotate, 45)
                rotated_mri_n45 = imutils.rotate_bound(mri_to_rotate, -45)
                rotated_mri_90 = imutils.rotate_bound(mri_to_rotate, 90)
                rotated_mri_180 = imutils.rotate_bound(mri_to_rotate, 180)
                rotated_mri_270 = imutils.rotate_bound(mri_to_rotate, 270)
                rotated_mri_360 = imutils.rotate_bound(mri_to_rotate, 360)

                fullpath1 = os.path.join(outPath, 'TEST_G3_rotated45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath1), rotated_mri_45)

                fullpath2 = os.path.join(outPath, 'TEST_G3_rotatedn45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath2), rotated_mri_n45)

                fullpath3 = os.path.join(outPath, 'TEST_G3_rotated90_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath3), rotated_mri_90)

                fullpath4 = os.path.join(outPath, 'TEST_G3_rotated180_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath4), rotated_mri_180)

                fullpath5 = os.path.join(outPath, 'TEST_G3_rotated270_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath5), rotated_mri_270)

                fullpath6 = os.path.join(outPath, 'TEST_G3_rotated360_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath6), rotated_mri_360)

            inPath = str(Path(e.get())) + '\Denoised\G4'
            outPath = str(Path(e.get())) + '\Rotated\G4'

            counter = 0
            for image_path in os.listdir(inPath):
                full_inpath = os.path.join(inPath, image_path)
                mri_to_rotate = cv2.imread(full_inpath, 0)
                counter += 1

                rotated_mri_45 = imutils.rotate_bound(mri_to_rotate, 45)
                rotated_mri_n45 = imutils.rotate_bound(mri_to_rotate, -45)
                rotated_mri_90 = imutils.rotate_bound(mri_to_rotate, 90)
                rotated_mri_180 = imutils.rotate_bound(mri_to_rotate, 180)
                rotated_mri_270 = imutils.rotate_bound(mri_to_rotate, 270)
                rotated_mri_360 = imutils.rotate_bound(mri_to_rotate, 360)

                fullpath1 = os.path.join(outPath, 'TEST_G4_rotated45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath1), rotated_mri_45)

                fullpath2 = os.path.join(outPath, 'TEST_G4_rotatedn45_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath2), rotated_mri_n45)

                fullpath3 = os.path.join(outPath, 'TEST_G4_rotated90_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath3), rotated_mri_90)

                fullpath4 = os.path.join(outPath, 'TEST_G4_rotated180_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath4), rotated_mri_180)

                fullpath5 = os.path.join(outPath, 'TEST_G4_rotated270_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath5), rotated_mri_270)

                fullpath6 = os.path.join(outPath, 'TEST_G4_rotated360_' + str(counter) + '.jpeg')
                imageio.imsave(Path(fullpath6), rotated_mri_360)




        except OSError:
            if not os.path.exists(inPath):
                messagebox.showerror("Invalid File Path", "Please re-enter a valid file path or choose an image folder"
                                                          " using the Choose Image Folder Button.")
        except ValueError:
            messagebox.showerror("Invalid Folder", "Please choose an image folder.")

        else:
            # filler = tk.Label(self, text="")
            # filler.grid(row=3, column=0, columnspan=3, pady=10)
            self.sep = tk.Label(self, text="")
            self.sep.grid(row=4, columnspan=4)
            self.img1Label = tk.Label(self, text="Denoised")
            self.img1Label.grid(row=5, column=1, columnspan=2, sticky="ew")

            img1 = Image.open(str(Path(e.get())) + '\Denoised\G0\TEST_G0_denoised7_1.jpeg')
            img1 = img1.resize((250, 250), Image.ANTIALIAS)
            new_img1 = ImageTk.PhotoImage(img1)
            self.result0 = tk.Label(self, image=new_img1)
            self.result0.image = new_img1
            self.result0.grid(row=6, column=1, columnspan=2, rowspan=4, padx=(20,10), pady=(0,15), sticky="w")

            self.img2Label = tk.Label(self, text="45° Rotated")
            self.img2Label.grid(row=5, column=3, columnspan=2, sticky="ew")

            img2 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotated45_1.jpeg')
            img2 = img2.resize((98, 98), Image.ANTIALIAS)
            new_img2 = ImageTk.PhotoImage(img2)
            self.result2 = tk.Label(self, image=new_img2)
            self.result2.image = new_img2
            self.result2.grid(row=6, column=3, columnspan=2, sticky="ew")

            self.img3Label = tk.Label(self, text="-45° Rotated")
            self.img3Label.grid(row=5, column=5, columnspan=2, padx=(0,20), sticky="ew")

            img3 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotatedn45_1.jpeg')
            img3 = img3.resize((98, 98), Image.ANTIALIAS)
            new_img3 = ImageTk.PhotoImage(img3)
            self.result3 = tk.Label(self, image=new_img3)
            self.result3.image = new_img3
            self.result3.grid(row=6, column=5, columnspan=2, padx=(0,20), sticky="ew")



            self.img4Label = tk.Label(self, text="90° Rotated")
            self.img4Label.grid(row=7, column=3, columnspan=2, sticky="ew")

            img4 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotated90_1.jpeg')
            img4 = img4.resize((98, 98), Image.ANTIALIAS)
            new_img4 = ImageTk.PhotoImage(img4)
            self.result4 = tk.Label(self, image=new_img4)
            self.result4.image = new_img4
            self.result4.grid(row=8, column=3, columnspan=2, sticky="ew")

            self.img5Label = tk.Label(self, text="180° Rotated")
            self.img5Label.grid(row=7, column=5, columnspan=2, padx=(0,20), sticky="ew")

            img5 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotated180_1.jpeg')
            img5 = img5.resize((98, 98), Image.ANTIALIAS)
            new_img5 = ImageTk.PhotoImage(img5)
            self.result5 = tk.Label(self, image=new_img5)
            self.result5.image = new_img5
            self.result5.grid(row=8, column=5, columnspan=2, padx=(0,20), sticky="ew")

            self.img6Label = tk.Label(self, text="270° Rotated")
            self.img6Label.grid(row=5, column=7, columnspan=2, padx=(0, 20), sticky="ew")

            img6 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotated270_1.jpeg')
            img6 = img6.resize((98, 98), Image.ANTIALIAS)
            new_img6 = ImageTk.PhotoImage(img6)
            self.result6 = tk.Label(self, image=new_img6)
            self.result6.image = new_img6
            self.result6.grid(row=6, column=7, columnspan=2, padx=(0, 20), sticky="ew")

            self.img7Label = tk.Label(self, text="360° Rotated")
            self.img7Label.grid(row=7, column=7, columnspan=2, padx=(0, 20), sticky="ew")

            img7 = Image.open(str(Path(e.get())) + '\Rotated\G0\TEST_G0_rotated360_1.jpeg')
            img7 = img7.resize((98, 98), Image.ANTIALIAS)
            img7 = img7.resize((98, 98), Image.ANTIALIAS)
            new_img7 = ImageTk.PhotoImage(img7)
            self.result7 = tk.Label(self, image=new_img7)
            self.result7.image = new_img7
            self.result7.grid(row=8, column=8, columnspan=2, padx=(0, 20), sticky="ew")

class Training(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="TRAINING PHASE", font=LARGE_FONT)  # initialize the label
        label.grid(pady=10, padx=10, sticky="WE")

        self.txtname = StringVar(None)
        lbl = tk.Label(self, text="Directory of Data Set")
        lbl.grid(column=0, row=1)
        txt = tk.Entry(self, textvariable=self.txtname, width=40)
        txt.grid(column=1, row=1, columnspan=3, sticky="WE")
        txt.focus()  # Focus the cursor on the textbox
        button0 = ttk.Button(self, text="Open Directory",
                             command=lambda: self.inDir())
        button0.grid(column=5, row=1, padx=(10,10), sticky="WE")

        button2 = ttk.Button(self, text="Train Model",
                            command=lambda: self.TrainModel(txt.get()))
        button2.grid(column=1, row=2, columnspan=3, sticky="WE")
        # self.mpb = ttk.Progressbar(self.mpb_frame, orient='horizontal', length=400, mode='determinate')
        # self.mpb.grid(row=3)
        # sys.stdout = self
        # self.canvas = tk.Canvas(self, width=100, height=100, bg="white")
        # self.canvas.grid(row=3, column=0, columnspan=6, padx=(10,10), sticky="WE")
        button2 = ttk.Button(self, text="Back",
                             command=lambda: controller.show_frame(StartPage))
        button2.grid(row=4, column=5,padx=(10,10) ,sticky="WE")

    def update_result(self, data):
        self.result.delete(0.0, tk.END)  # clear all the old data out
        self.result.insert(0.0, data)  # put new data in
        
    def inDir(self):
        filedir = filedialog.askdirectory(title='Choose a Directory')
        self.txtname.set(filedir)

    # def write(self, txt):
    #     self.txttrain.insert(tk.END, str(txt))
    #     #######This is the missing line!!!!:
    #     self.update_idletasks()

    def flush(self):
        pass

    def TrainModel(self, DataDir):
        import matplotlib

        matplotlib.use("Agg")

        # import the necessary packages
        # from keras.preprocessing.image import \
        #     ImageDataGenerator  # used for data augmentation (helps prevent overfitting)
        # from keras.optimizers import \
        #     Adam  # imports the Adam  optimizer, the optimizer method used to train our network.
        # from keras.preprocessing.image import img_to_array
        # from sklearn.preprocessing import LabelBinarizer  # transform a class label string to an integer and vice versa.
        # from sklearn.model_selection import train_test_split  # Used to create training and testing splits
        # from TestGuiClassify.cnnmastercode.smallervggnet import SmallerVGGNet
        # from keras.callbacks import TensorBoard
        # from datetime import datetime
        # import matplotlib.pyplot as plt
        # from imutils import paths
        # import numpy as np
        # import random
        # import pickle
        # import cv2
        # import os

        # initialize the number of epochs to train for, initial learning rate,
        # batch size, and image dimensions
        EPOCHS = 20  # EPOCHS:  The total number of epochs we will be training our network for
        # (i.e., how many times our network “sees” each training example and learns patterns from it).
        INIT_LR = 1e-3  # INIT_LR:  The initial learning rate — a value of 1e-3 is the default value for the Adam optimizer,
        # the optimizer we will be using to train the network.
        BS = 32  # We will be passing batches of images into our network for training.
        # There are multiple batches per epoch. The BS  value controls the batch size.
        IMAGE_DIMS = (96, 96, 1)  # Supplying the spatial dimensions of our input images.
        # In this case we are using 96 x 96 pixels with 3 channels (RGB)
        # For the Thesis, Thesis we will be using, ## x ## with 2 channels (Grayscale)

        # initialize the data and labels
        data = []
        labels = []

        # grab the image paths and randomly shuffle them
        print("[INFO] loading images...")
        dataset = DataDir
        print(DataDir)
        imagePaths = sorted(list(paths.list_images(dataset)))
        random.seed(42)
        random.shuffle(imagePaths)

        # loop over the input images
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]  # os.path.sep is for the windows pathname separator "\"
            labels.append(label)

        # scale the raw pixel intensities to the range [0, 1]
        # print(labels)
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        # print(labels)
        print("[INFO] data matrix: {:.2f}MB".format(
            data.nbytes / (1024 * 1000.0)))

        # binarize the labels
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        def TrainModel():
            # initialize the model
            print("[INFO] compiling model...")
            now = datetime.now()
            timeStamp = now.strftime("%m-%d-%Y_%H%M%S")
            NAME = "Model Version {}".format(timeStamp)
            tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))
            model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                                        depth=IMAGE_DIMS[2], classes=len(lb.classes_))
            opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
            model.compile(loss="categorical_crossentropy", optimizer=opt,
                          metrics=["accuracy"])
            # compile  our model  with categorical cross-entropy since we have > 2 classes
            # For only two classes you should use binary cross-entropy as the loss.

            # train the network
            print("Layers of the CNN: ", len(model.layers))
            print("[INFO] training network...")
            H = model.fit_generator(
                aug.flow(trainX, trainY, batch_size=BS),
                validation_data=(testX, testY),
                steps_per_epoch=len(trainX) // BS,
                epochs=EPOCHS, verbose=1, callbacks=[tensorboard])

            # save the model to disk
            print("[INFO] serializing network...")
            model.save(NAME)

            # save the label binarizer to disk
            print("[INFO] serializing label binarizer...")
            f = open("Labelbin {}.pickle".format(NAME), "wb")
            f.write(pickle.dumps(lb))
            f.close()

            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            N = EPOCHS

            history_dict = H.history
            print(history_dict.keys())

            plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
            plt.title("Training Loss")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="upper left")
            plt.savefig("LOSSPlot {}".format(NAME))

            plt.style.use("ggplot")
            plt.figure()

            plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
            plt.title("Training Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend(loc="upper left")
            plt.savefig("ACCPlot {}".format(NAME))

        TrainModel()

app = PracClass()
app.mainloop()