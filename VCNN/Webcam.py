def cam_shot(input_shape, mirror=True, save=True, save_name='cam_input.png'):
	from PIL import Image
	import numpy as np
	import cv2

	x = input_shape[1]
	y = input_shape[2]

	cam = cv2.VideoCapture(0)
	ret_val, img = cam.read()

	shape = list(img.shape)
	width = shape[1]
	height = shape[0]
	delta = int((width-height)/2)

	#if the webcam it's mirrored flip the image
	if mirror:
		img = cv2.flip(img, 1)

	img = img[0:height, delta:width-delta] #crop the image
	img = cv2.resize(img, (x, y)) #resize the image

	img = np.asarray(img)
	img = Image.fromarray(img)

	if save: #if save flag it's true save the image
		cv2.imwrite(save_name, np.asarray(img))

	return img
