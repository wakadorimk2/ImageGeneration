import tensorflow as tf
import cv2
import numpy as np
import pathlib
import matplotlib.pylab as plt

from conf import IMAGE_HEIGHT, IMAGE_WIDTH

def my_load_data(input_dir='../Datasets/kantoku'):
  """ load dataset from local, and convert to numpy array """
  images = []
  length = 0
  for path in pathlib.Path(input_dir).iterdir():
    path = str(path)
    image = cv2.imread(path)
    if image is None:
      continue
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    length += 1
  return np.asarray(images, dtype='int32').reshape(length, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

def generate_and_save_images(model, epoch, test_input):
  """ generate images from a model, and plot to figure """
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      #import pdb;pdb.set_trace()
      plt.imshow(predictions[i, :, :, :] /2 + 1, cmap='gist_rainbow')
      #plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5, cmap='gist_rainbow')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_crop(image):
  return tf.image.random_crop(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

def random_jitter(image, alpha=1.1):
  # resizing to 286 x 286 x 3
  size = [int(IMAGE_HEIGHT*alpha), int(IMAGE_WIDTH*alpha)]
  image = tf.image.resize(image, size,
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image):
  image = random_jitter(image)
  image = normalize(image)
  return image