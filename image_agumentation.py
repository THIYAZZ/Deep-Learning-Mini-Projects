import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import PIL.Image


print("Loading model...")
generator = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']
print("Model loaded.")

noise = np.random.normal(size=[1, 512])


generated_image = generator(tf.constant(noise, dtype=tf.float32))['default']

generated_image = (generated_image.numpy().squeeze() + 1) * 127.5 
generated_image = np.uint8(generated_image)
img = PIL.Image.fromarray(generated_image)

img.save("new_face.png")
print("New image saved as 'new_face.png'")
