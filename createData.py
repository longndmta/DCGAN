import glob as gb
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def normalization(x):
    return x / 127.5 - 1


def format_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize((256, 128), Image.ANTIALIAS)
    image = image.convert('RGB')
    #image.show()
    # return the numpy arrays
    return np.array(image)

# convert images to hdf5 data

def build_hdf5(jpeg_dir, size=256):
	hdf5_file = os.path.join('data', 'data.h5')
	data_full = []
	data_blur = []
	with h5py.File(hdf5_file, 'w') as f:
		for data_type in tqdm(['sharp', 'blur'], desc = 'create HDF5 dataset from images'):
			data_path = jpeg_dir + '/%s/*.jpg' % data_type
			images_path = gb.glob(data_path)
			images_path.sort()
			#print(images_path)
			for image_path in images_path:
				if data_type == 'sharp':
					image_full = format_image(image_path, size)
					#print(image_full)
					data_full.append(image_full)
				elif data_type == 'blur':
					image_blur = format_image(image_path, size)
					#print(image_blur)
					data_blur.append(image_blur)

			data_path = jpeg_dir + '/%s/*.png' % data_type
			images_path = gb.glob(data_path)
			images_path.sort()
			#print(images_path)
			for image_path in images_path:
				if data_type == 'sharp':
					image_full = format_image(image_path, size)
					#print(image_full)
					data_full.append(image_full)
				elif data_type == 'blur':
					image_blur = format_image(image_path, size)
					#print(image_blur)
					data_blur.append(image_blur)
		#print((data_blur))
		f.create_dataset('_data_full' , data=data_full)
		f.create_dataset('_data_blur' , data=data_blur)

# load data by data type
def load_data():
    with h5py.File('data/data.h5', 'r') as f:
        data_full = f['_data_full'][:].astype(np.float32)
        data_full = normalization(data_full)

        data_blur = f['_data_blur'][:].astype(np.float32)
        data_blur = normalization(data_blur)

        return data_full, data_blur

def generate_image(full, blur, generated, path, epoch=None, index=None):
    full = full * 127.5 + 127.5
    blur = blur * 127.5 + 127.5
    generated = generated * 127.5 + 127.5
    for i in range(generated.shape[0]):
        image_full = full[i, :, :, :]
        image_blur = blur[i, :, :, :]
        image_generated = generated[i, :, :, :]
        image = np.concatenate((image_full, image_blur, image_generated), axis=1)
        if (epoch is not None) and (index is not None):
            Image.fromarray(image.astype(np.uint8)).save(path + str(epoch + 1) + '_' + str(index + 1) + '.png')
        else:
            Image.fromarray(image.astype(np.uint8)).save(path + str(i) + '.png')


if __name__ == '__main__':
    #format_image('data/small/test/301.jpg', size=256)
    build_hdf5('data')
    img_full, img_blur = load_data()
    print(len(img_full), '\n', len(img_blur))



