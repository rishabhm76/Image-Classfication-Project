# Import Libraries
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model('CIFAR10-pre_trained_model.h5')
print(model.summary())

# load and prepare image
'''ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']
'''
def load_img(filename):
    target_size = (32,32)
    img = image.load_img(filename, target_size=target_size)
    img = image.img_to_array(img)
    #reshaping image
    img = img.reshape(1,32,32,3)
    img = img.astype('float32')
    img = img / 255
    return img

def prediction():
    img = load_img('images.jpg')
    # predicting results
    result = model.predict_classes(img)
    print(result)

prediction()
