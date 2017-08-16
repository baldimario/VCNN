from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K

import tkinter
from tkinter import *
from tkinter import messagebox as mbox
from tkinter import filedialog
from PIL import Image, ImageTk
import keras
from keras import *
from keras.models import Model
import keras.backend as K
#K.set_learning_phase(0)
import numpy as np
from pylab import cm

import matplotlib.pyplot as plt

from VCNN.Webcam import cam_shot
from VCNN.Core import predict, summary, load_model, get_activations, max_activation, get_activations_images, get_input_shape

class VCNN(Frame):
	w = 600
	h = 600
	h5 = '' #'models\\VGG16\\vgg16.h5'
	json = '' #'models\\VGG16\\vgg16.json'
	image = '' #'models\\VGG16\\cat.jpg'
	model = None
	layersMenu = None
	cmap = 'default'
	last_loaded_layer = ''
	images = []
	frame = None
	menubar = None
	network_input = None
	debug = False #True

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):
		self.master.title('VCNN')
		self.pack(fill=BOTH, expand=1)
		self.centerWindow()

		#content frame
		self.frame = Frame(self)
		self.frame.pack(side=TOP, fill=BOTH, expand=YES)

		#menu bar
		menubar = Menu(self.master)
		self.master.config(menu=menubar)

		#menu file
		fileMenu = Menu(menubar)

		#import submenu
		submenu = Menu(fileMenu)
		submenu.add_command(label="Json", command=self.loadJson)
		submenu.add_command(label="H5", command=self.loadH5)
		submenu.add_command(label="Image", command=self.loadImage)

		#menu file
		fileMenu.add_cascade(label='Import', menu=submenu)
		fileMenu.add_separator()
		fileMenu.add_command(label="Exit", command=self.onExit)

		menubar.add_cascade(label="File", menu=fileMenu)

		#menu net
		netMenu = Menu(menubar)
		netMenu.add_cascade(label='Load Model', command=self.loadModel)
		netMenu.add_cascade(label='Run', command=self.run)
		netMenu.add_cascade(label='Summary', command=self.summary)
		menubar.add_cascade(label='Net', menu=netMenu)

		#menu layers (it's a global variable 'cause it have to be dynamic)
		self.layersMenu = Menu()
		menubar.add_cascade(label='Layers', menu=self.layersMenu)

		#sources menu
		sourcesMenu = Menu()
		sourcesMenu.add_cascade(label='Webcam', command=self.camShot)
		menubar.add_cascade(label='Sources', menu=sourcesMenu)

		#color map menu
		cmapMenu = Menu(menubar)
		cmapMenu.add_command(label='Grey', command=lambda: self.setCmap('grey'))
		cmapMenu.add_command(label='Jet', command=lambda: self.setCmap('jet'))
		cmapMenu.add_command(label='Reds', command=lambda: self.setCmap('Reds'))
		cmapMenu.add_command(label='Greens', command=lambda: self.setCmap('Greens'))
		cmapMenu.add_command(label='Blues', command=lambda: self.setCmap('Blues'))
		cmapMenu.add_command(label='Default', command=lambda: self.setCmap('default'))
		menubar.add_cascade(label='Cmap', menu=cmapMenu)

		#globalize the main menubar
		self.menubar = menubar

		if(self.debug):
			self.loadModel(silent=True)
			self.run(silent=True)

	#webacm callback, it uses opencv to take a photo, square crop at center and resize as the input shape
	def camShot(self, mirror = True):
		#check loaded model
		if(self.model == None):
			mbox.showerror("Error", "Please load a model first")
			return None

		input_shape = get_input_shape(self.model)

		#storing the webcam image name
		self.image = 'cam_input.png'

		#take an image from the webcam
		img = cam_shot(input_shape, save=True, save_name=self.image)

		#self.showImage(img) #show the image

		self.cleanImages() #clean the content frame

		#run the prediction to get the activations
		self.run()
		return img

	#exit callback
	def onExit(self):
		self.quit()

	#set the colormap global variable and reload the layer
	def setCmap(self, cmap):
		self.cmap = cmap
		if(self.last_loaded_layer != ''):
			self.loadLayer(self.last_loaded_layer)

	#show the model summary
	def summary(self):
		#check if the model is loaded
		if(self.model == None):
			mbox.showerror("Error", "Please load the model first")
			return

		txt = summary(self.model)

		mbox.showinfo("Summary", txt)

	#run the neural network on the chosen input to prepare the activations and get the input shape
	def run(self, silent=False):
		#check if the model is loaded
		if(self.model == None):
			mbox.showerror("Error", "Please load the model first")
			return

		#check if the input impage is loaded
		if(self.image == ''):
			mbox.showerror("Error", "Please import an adeguate image file")
			return

		self.network_input = predict(self.model, self.image)

		#prompt the user
		if(not silent):
			mbox.showinfo("Info", 'Input predicted')

	#clear all the images in the main content frame
	def cleanImages(self):
		for img in self.images:
			img.destroy()

	#creates an immage and appends it to the main content frame
	def showImage(self, img, row = 1, col = 1, button=False, lambda_f=None):
		photo = ImageTk.PhotoImage(img)
		if(button):
			widget = Button(self.frame, image=photo, width=img.size[0], height=img.size[1], borderwidth=2, state='normal', padx=0, pady=0, highlightthickness=0, command=lambda_f)
		else:
			widget = Label(self.frame, image=photo, width=img.size[0], height=img.size[1], borderwidth=2, state='normal', padx=0, pady=0, highlightthickness=0)
		widget.image = photo
		widget.grid(row=row, column=col)

		#label.pack(anchor=NW, side=TOP|LEFT) #.pack(fill=BOTH, expand=1)
		self.images.append(widget)
		return widget

	#display the activation of a layer
	def displayActivtions(self, activation_maps, layer_name=None):
		import numpy as np
		import matplotlib.pyplot as plt
		import os

		self.cleanImages()

		if(layer_name != None):
			self.master.title('VCNN: '+layer_name)

		imgs = get_activations_images(self.model, activation_maps, layer_name, window=[self.winfo_width(), self.winfo_height()], cmap=self.cmap)

		batch_size = activation_maps[0].shape[0]
		assert batch_size == 1, 'One image at a time to visualize.'

		square_grid = int(np.ceil(np.sqrt(len(imgs))))
		row = 0
		for i, img in enumerate(imgs):
			col = i%square_grid
			if(i != 0 and col == 0):
				row += 1

			if len(imgs) > 1:
				lambda_f = lambda name_layer=layer_name, index_filter=i: self.maxActivation(name_layer, index_filter)
				self.showImage(img, row, col, button=True, lambda_f=lambda_f)
			else:
				self.showImage(img, row, col, button=False) #show single image

	#loads a layer and displays the activation
	def loadLayer(self, layer_name):
		self.last_loaded_layer = layer_name
		activations = get_activations(self.model, self.network_input, layer_name)
		self.displayActivtions(activations, layer_name)

	#load the model
	def loadModel(self, silent=False):
		#check if the json it's loaded
		if(self.json == ''):
			mbox.showerror("Error", "Please import the Json file")
			return

		#check if the weights are loaded
		if(self.h5 == ''):
			mbox.showerror("Error", "Please import the H5 file")
			return

		self.model = load_model(self.json, self.h5)

		#delete the layers menu of the previous loaded neural network
		self.layersMenu.delete(0, 'end')

		#create the new layers menu
		for layer in self.model.layers:
			self.layersMenu.add_command(label=layer.name, command=lambda name=layer.name: self.loadLayer(name))

		#prompt the user
		if(not silent):
			mbox.showinfo("Info", "Model Loaded")

	#loads the neural network weights
	def loadH5(self):
		#show the file chooser dialog
		ftypes = [('H5 files', '*.h5'), ('All files', '*')]
		dlg = filedialog.Open(self, filetypes = ftypes)
		fl = dlg.show()

		if fl != '':
			self.h5 = fl
			mbox.showinfo("Info", "H5 Loaded")
		else:
			mbox.showerror("Error", "Could not open file")

	#loads the neural network model
	def loadJson(self):
		#show the file chooser dialog
		ftypes = [('Json files', '*.json'), ('All files', '*')]
		dlg = filedialog.Open(self, filetypes = ftypes)
		fl = dlg.show()

		if fl != '':
			self.json = fl
			mbox.showinfo("Info", "Json Loaded")
		else:
			mbox.showerror("Error", "Could not open file")

	#loads the input image
	def loadImage(self):
		#show the file chooser dialog
		ftypes = [('Image files', '*.png'), ('Image files', '*.bmp'), ('Image files', '*.jpg'), ('Image files', '*.jpeg'), ('All files', '*')]
		dlg = filedialog.Open(self, filetypes = ftypes)
		fl = dlg.show()

		if fl != '':
			self.image = fl
			mbox.showinfo("Info", "Image Loaded")
			return 0
		else:
			mbox.showerror("Error", "Could not open file")

	#show the max activations for that filter
	def maxActivation(self, layer_name, filter_index, show=True):
		img = max_activation(self.model, layer_name, filter_index)

		if(show):
			plt.imshow(img.squeeze(), interpolation='None', cmap='gray')
			plt.show()

		return img

	#center the window and set the geometry
	def centerWindow(self):
		sw = self.master.winfo_screenwidth()
		sh = self.master.winfo_screenheight()

		x = (sw - self.w)/2
		y = (sh - self.h)/2
		self.master.geometry('%dx%d+%d+%d' % (self.w, self.h, x, y))

#init the application gui
def main():
	root = Tk()
	app = VCNN()
	root.mainloop()

#load the application
if __name__ == '__main__':
	main()
