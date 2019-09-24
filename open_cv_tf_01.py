import tensorflow as tf
from tensorflow import keras
import pathlib
import random
import matplotlib.pyplot as plt
import cv2
tf.enable_eager_execution()

data_root ='framesdata'
data_root = pathlib.Path(data_root)
print(data_root)
for item in data_root.iterdir():
    print(item)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
print(all_image_paths[:10])

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

label_to_index = dict((name, index) for index,name in enumerate(label_names))
print(label_to_index)
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
print(img_path)

img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_jpeg(img_raw)
img_tensor = tf.reshape(img_tensor, [209, 171, 3])
print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize_images(img_tensor, [209, 171])
img_final = img_final/255
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [209, 171])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel('My XLabel')
plt.title(label_names[label])
print()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.contrib.data.AUTOTUNE)

plt.figure(figsize=(8,8))
for n,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('My XLabel')

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(2):
    print(label_names[label.numpy()])

image_label_ds_zip = tf.data.Dataset.zip((image_ds, label_ds))

print('image shape: ', image_label_ds_zip.output_shapes[0])
print('label shape: ', image_label_ds_zip.output_shapes[1])
print('types: ', image_label_ds_zip.output_types)
print()
print(image_label_ds_zip)


ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds_map = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds_map)


BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
prefetch_ds = image_label_ds.shuffle(buffer_size=image_count)
prefetch_ds = prefetch_ds.repeat()
prefetch_ds = prefetch_ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
prefetch_ds = prefetch_ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
print(prefetch_ds)

prefetch_ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
prefetch_ds = prefetch_ds.batch(BATCH_SIZE)
prefetch_ds = prefetch_ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
print(prefetch_ds)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(209, 171, 3)),
    keras.layers.LSTM(input_shape=103113, activation=tf.nn.tanh),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
