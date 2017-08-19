#returns the input shape of the first layer that's the input of the entire neural network
def get_input_shape(model):
	return model.layers[0].input.shape

#redirect the stream of summary output
def summary(model):
	global txt
	txt = ''

	#concatenation routine for the summary stream redirection
	def conc(x):
		global txt
		txt += "\n"+x

	model.summary(print_fn=lambda x: conc(x))

	return txt

#run the neural network with given input
def get_network_input(model, image):
	from VCNN.ImageTools import is_grey_scale
	from PIL import Image
	import numpy as np

	img = Image.open(image)
	input_data = np.asarray(img)
	input_data = input_data.astype('float32') / 255.

	#get the neural network's input shape of the first layer
	input_shape = get_input_shape(model)
	lis = list(input_shape)

	if(input_data.shape != input_shape):
		input_data = np.reshape(input_data, (lis[1], lis[2], lis[3]))

	#if the image is grayscale get only the first channel (R=G=B)
	if(is_grey_scale(image)):
		input_data = input_data[:,:,0]

	#reshape the loaded image to fit the neural network's input dimensions
	network_input = np.reshape(input_data, (1, input_shape[1], input_shape[2], input_shape[3]))

	return network_input

#read the json file and import in a new model
def load_model(json, h5):
	from keras.models import Model, model_from_json

	json_file = open(json, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(h5)
	loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy')
	return loaded_model

#This function seek an image that maximize the activations of a specific filter in a specific layer
def max_activation(model, layer_name, filter_index, steps=50):
	from keras import backend as K
	import numpy as np
	from VCNN.ImageTools import deprocess_image

	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	input_img = model.input

	shape_input = get_input_shape(model)
	img_width = shape_input[1]
	img_height = shape_input[2]
	img_depth = shape_input[3]

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	loss = K.mean(layer_output[:, :, :, filter_index])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	# we start from a gray image with some noise
	input_img_data = np.random.random((1, img_width, img_height, img_depth)) * 20 + 128.

	step = 1.
	for i in range(steps):
		loss_value, grads_value = iterate([input_img_data])

		grads_value = np.clip(grads_value, 0, 255)
		input_img_data += grads_value * step


		if loss_value <= 0.:
			break

	img = input_img_data[0]
	img = deprocess_image(img)

	img = np.reshape(img, (shape_input[1], shape_input[2], shape_input[3]))

	return img

#get the model activations of a layer
def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
	from keras import backend as K

	activations = []
	inp = model.input

	model_multi_inputs_cond = True
	if not isinstance(inp, list):
		# only one input! let's wrap it in a list.
		inp = [inp]
		model_multi_inputs_cond = False

	outputs = []
	if layer_name != None:
		for l in model.layers:
			if l.name == layer_name:
				outputs.append(l.output)
	else:
		outputs = [layer.output for layer in model.layers if
			layer.name == layer_name or layer_name is None]  # all layer outputs

	funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

	if model_multi_inputs_cond:
		list_inputs = []
		list_inputs.extend(model_inputs)
		list_inputs.append(1.)
	else:
		list_inputs = [model_inputs, 1.]

	# Learning phase. 1 = Test mode (no dropout or batch normalization)
	# layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
	layer_outputs = [func(list_inputs)[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)

		print('LAS', layer_activations.shape)
		if print_shape_only:
			la = layer_activations
			las = list(layer_activations.shape)
			import numpy as np
			la = np.reshape(la, (las[1], las[2], las[3]))

			for i in range(0, las[3]):
				a = la[:,:,i]

				import matplotlib.pyplot as plt
				plt.imshow(a)
				plt.show()

		else:
			print(layer_activations)
	return activations

def get_activations_images(model, activation_maps, layer_name, window=[600, 600], cmap='default'):
	from VCNN.ImageTools import array_to_image
	import matplotlib.pyplot as plt
	from PIL import Image, ImageTk
	import numpy as np
	import os

	imgs = []

	layer = None
	for l in model.layers:
		if(l.name == layer_name):
			layer = l

	for i, activation_map in enumerate(activation_maps):
		if(model.layers[i].name == layer_name or len(activation_maps) == 1):
			shape = activation_map.shape
			if len(shape) == 4:
				activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
			elif len(shape) == 2:
				# try to make it square as much as possible. we can skip some activations.
				activations = activation_map[0]
				num_activations = len(activations)

				if num_activations > 1024:  # too hard to display it on the screen.
					square_param = int(np.floor(np.sqrt(num_activations)))
					activations = activations[0: square_param * square_param]
					activations = np.reshape(activations, (square_param, square_param))
				else:
					activations = np.expand_dims(activations, axis=0)
			else:
				raise Exception('len(shape) = 3 has not been implemented.')

			out_shape = list(shape)#model.layers[i].output.shape.as_list()
			if(len(out_shape) > 3): #if the shape is square (result of a convolution to get the filters)
				out_w = out_shape[1]
				out_h = out_shape[2]
				out_a = out_shape[3]

				#calculate the resizing values
				square_grid = int(np.ceil(np.sqrt(out_a)))

				border = 2
				sw = window[0]-2*border*square_grid
				sh = window[1]-2*border*square_grid

				sizeX = int(sw/square_grid)
				sizeY = int(sh/square_grid)

				#draw the images in a grid
				row = 0
				for i in range(0, out_a):
					col = i%square_grid
					if(i != 0 and col == 0):
						row += 1

					#cut the activations image to get the desired filter
					single_act = activations[:,i*out_w:(i+1)*out_w]
					img = array_to_image(single_act, cmap=cmap)
					#resize the filter to fit all the space
					img = img.resize((sizeX, sizeY))
					#show the image
					imgs.append(img)
			else: #otherwise the activations aren't a convolution results but it's fully connecter/flattened/softmax results show as it is
				img = array_to_image(activations, cmap=cmap)

				border = 2
				sw = window[0]-2*border
				sh = window[1]-2*border

				if(img.size[1] > img.size[0]):
					r = img.size[0]/img.size[1]
				else:
					r = img.size[1]/img.size[0]

				#if the height it's too little stretch the height to make the activations visible
				if(r < 0.01):
					r = 0.01

				sizeX = int(sw)
				sizeY = int(sw*r)

				#resize the image
				img = img.resize((sizeX, sizeY))

				imgs.append(img)

	return imgs
