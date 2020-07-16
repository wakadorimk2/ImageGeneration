import cv2
import numpy as np
import pathlib
import matplotlib.pylab as plt

def my_load_data(input_dir='../Datasets/kantoku'):
  images = []
  length = 0
  for path in pathlib.Path(input_dir).iterdir():
    path = str(path)
    image = cv2.imread(path)
    if image is None:
      continue
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    length += 1
  return np.asarray(images, dtype='float32').reshape(length, 128, 128, 3)

def generate_and_save_images(model, epoch, test_input):
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
