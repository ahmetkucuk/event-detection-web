import numpy as np
from tile_image import SolarImageLabeler
import tensorflow as tf
from PIL import Image
import lenet
import os


def get_image_filenames(directory):
	image_filenames = []
	for filename in os.listdir(directory):
		if ".jpg" in filename or ".png" in filename:
			path = os.path.join(directory, filename)
			image_filenames.append([filename, path])
	return image_filenames


def preprocess(image, image_size):
	image = image.astype(np.uint8)
	image = Image.fromarray(image)
	im = image.convert("L")
	im = im.resize(size=(image_size, image_size))
	im_array = np.array(im)
	im_array = np.expand_dims(im_array, axis=2)
	return im_array


class FullDiskDetector(object):

	def __init__(self, model_ckpt):

		self.image_size = 2048
		self.patch_size = 64
		self.input_image_size = 128

		self.session = tf.Session()
		self.images_placeholder = tf.placeholder("float", [None, self.input_image_size, self.input_image_size, 1])
		network_fn = lenet.lenet
		self.logits, self.end_points = network_fn(self.images_placeholder, num_classes=2, is_training=False)
		self.preds = tf.nn.softmax(self.logits)

		saver = tf.train.Saver()
		print(model_ckpt)
		saver.restore(self.session, model_ckpt)

	def label_image(self, input_image_path, output_image_path):

		labeler = SolarImageLabeler(input_image_path, self.patch_size)
		patch_count = int(self.image_size/self.patch_size)

		for i in range(patch_count):
			for j in range(patch_count):
				patch_image = labeler.get_patch(i, j)
				input_image = preprocess(patch_image, self.input_image_size)

				preds_results = self.session.run(self.preds, feed_dict={self.images_placeholder: [input_image]})

				labels = ['AR', 'QS']
				label_index = preds_results.argmax(1)[0]
				label = labels[label_index]
				if label == "AR":
					labeler.add_label(i, j, label)
		labeler.save_fig(output_image_path)


# if __name__ == "__main__":
# 	main(sys.argv[1:])

# args = ["data/20170211_001146_4096_0171.jpg", "test.png", "/Users/ahmetkucuk/Documents/log_test/"]
# args = ["data/unlabeled/", "data/labeled/", "/Users/ahmetkucuk/Documents/log_test/"]
# main(args)
# python detect-full-disk.py "/home/ahmet/workspace/ar-detection/data/unlabeled/" "data/labeled/" "/home/ahmet/workspace/tensorboard/lenet/"