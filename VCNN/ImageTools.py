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

def deprocess_image(x):
	import numpy as np
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	#clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	#convert to RGB array
	x *= 255
	#x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def array_to_image(myarray, cmap='default'):
	from PIL import Image, ImageTk
	from pylab import cm
	import numpy as np

	myarray = deprocess_image(myarray)

	f = 255
	if(cmap == 'grey'):
		return Image.fromarray(np.uint8(cm.gray(myarray)*f))
	elif(cmap == 'jet'):
		return Image.fromarray(np.uint8(cm.jet(myarray)*f))
	elif(cmap == 'Reds'):
		return Image.fromarray(np.uint8(cm.Reds(myarray)*f))
	elif(cmap == 'Greens'):
		return Image.fromarray(np.uint8(cm.Greens(myarray)*f))
	elif(cmap == 'Blues'):
		return Image.fromarray(np.uint8(cm.Blues(myarray)*f))
	elif(cmap == 'viridis'):
		return Image.fromarray(np.uint8(cm.viridis(myarray)*f))
	elif(cmap == 'default'):
		return Image.fromarray(np.uint8(myarray*f))
	#else:
	#	return Image.fromarray(np.uint8(cm.gray(myarray)*f))
