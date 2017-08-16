#check if the image is grayscale (R=G=B)
def is_grey_scale(img_path):
	from PIL import Image

	im = Image.open(img_path).convert('RGB')
	w,h = im.size
	for i in range(w):
		for j in range(h):
			r,g,b = im.getpixel((i,j))
			if r != g != b: return False

	return True

def array_to_image(myarray, cmap='default'):
	from PIL import Image, ImageTk
	from pylab import cm
	import numpy as np

	if(cmap == 'gray'):
		return Image.fromarray(np.uint8(cm.gray(myarray)*255))
	elif(cmap == 'jet'):
		return Image.fromarray(np.uint8(cm.jet(myarray)*255))
	elif(cmap == 'Reds'):
		return Image.fromarray(np.uint8(cm.Reds(myarray)*255))
	elif(cmap == 'Greens'):
		return Image.fromarray(np.uint8(cm.Greens(myarray)*255))
	elif(cmap == 'Blues'):
		return Image.fromarray(np.uint8(cm.Blues(myarray)*255))
	elif(cmap == 'default'):
		return Image.fromarray(np.uint8(myarray*255))
	else:
		return Image.fromarray(np.uint8(cm.gray(myarray)*255))
