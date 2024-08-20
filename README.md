# disaster-image-classification
!pip install tensorflow_hub
!pip install tensorflow_datasets
!pip install lime
!pip install colorama
!pip install opencv-python
!git clone https://github.com/samson6460/tf_keras_gradcamplusplus.git ./assets/tf_keras_gradcamplusplus

import os
import sys
import re
import random
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub

from colorama import Fore, Style
from PIL import Image, ImageFile
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Identity, InputLayer, Rescaling
from keras.models import Sequential
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from assets.tf_keras_gradcamplusplus.utils import preprocess_image
from assets.tf_keras_gradcamplusplus.gradcam import grad_cam, grad_cam_plus
from IPython.display import clear_output

clear_output()
print(Fore.GREEN + u'\u2713 ' + 'Successfully downloaded dependencies.')
print(Style.RESET_ALL)

class IgnorePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

        #Global Variables
        DATASETS = ['disaster_images', 'medic']

DATASET = 'disaster_images'
# DATASET = 'medic'

CLASSES = []

SEED = 68765

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.2
TEST_SPLIT = 0.1

IMAGE_SHAPE_2D = (224, 224)
IMAGE_SHAPE_3D = (224, 224, 3)

SOURCE_DIRECTORY = './assets/disaster_data/'
REFACTORED_DIRECTORY = './assets/refactored_data/'
TRAIN_DIRECTORY = './assets/refactored_data/train/'
VALID_DIRECTORY = './assets/refactored_data/valid/'
TEST_DIRECTORY = './assets/refactored_data/test/'

EPOCHS = 1000
PATIENCE = 20
BATCH_SIZE = 128

# LEARNING_RATE = 0.01
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001

BASE_MODEL = ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.resnet50.preprocess_input

# BASE_MODEL = InceptionV3(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.inception_v3.preprocess_input

# BASE_MODEL = VGG19(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.vgg19.preprocess_input

# BASE_MODEL = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.efficientnet.preprocess_input

# BASE_MODEL = EfficientNetB7(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.efficientnet.preprocess_input

# BASE_MODEL = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input

# BASE_MODEL = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE_3D)
# PREPROCESSING_METHOD = preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input

# BASE_MODEL = hub.KerasLayer('https://tfhub.dev/sayakpaul/vit_b32_fe/1')
# PREPROCESSING_METHOD = None

# OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#helper variables
#load dataset
def load_dataset():
  global CLASSES
  assert DATASET in DATASETS, f'Input string \'{DATASET}\' must be one of following: {DATASETS}.'


  def _load_disaster_images_dataset():
    !git clone https://github.com/tariqshaban/disaster-classification-with-xai.git
    !mv -v ./disaster-classification-with-xai/assets/* ./assets
    !rm -rf disaster-classification-with-xai

  def _load_medic_dataset():
    !wget https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz
    !mkdir ./Medic
    !tar -xzvf ./MEDIC.tar.gz -C ./Medic

    !mkdir -p ./assets/disaster_data/_raw
    !mkdir ./assets/disaster_data/_metadata/
    !mv ./Medic/data/* ./assets/disaster_data/_raw
    !mv ./Medic/MEDIC_train.tsv ./assets/disaster_data/_metadata/
    !mv ./Medic/MEDIC_dev.tsv ./assets/disaster_data/_metadata/
    !mv ./Medic/MEDIC_test.tsv ./assets/disaster_data/_metadata/
    !rm -rf ./Medic

    tsv_files = !find ./assets/disaster_data/_metadata -name '*.tsv'

    dfs = []
    for file in tsv_files:
        dfs.append(pd.read_csv(file, sep='\t'))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['informative'] == 'informative']
    df = df[['disaster_types', 'image_path']]
    df['image_path'] = df['image_path'].str.split('/', n=1, expand=True)[1]

    for index, row in df.iterrows():
        image_path = row['image_path']
        class_name = row['disaster_types']
        destination_folder = f'./assets/disaster_data/{class_name}'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print(f'./assets/disaster_data/_raw/{image_path}')
        print(f'{destination_folder}/{os.path.basename(image_path)}')
        shutil.move(f'./assets/disaster_data/_raw/{image_path}', f'{destination_folder}/{os.path.basename(image_path)}')

    !rm -rf ./assets/disaster_data/_raw
    !rm -rf ./assets/disaster_data/_metadata

  if DATASET == 'disaster_images':
    _load_disaster_images_dataset()
  elif DATASET == 'medic':
    _load_medic_dataset()

  CLASSES = os.listdir('./assets/disaster_data')

  Prime Dataset
  def prime_dataset():
  if os.path.exists(REFACTORED_DIRECTORY):
    shutil.rmtree(REFACTORED_DIRECTORY)

  # Read Each Image With its Class Label
  images = []
  folders=CLASSES

  for folder in folders:
    t = folder
    x = !ls $SOURCE_DIRECTORY$t
    for i in x:
      for j in re.split(r'[-;,\t\s]\s*', i):
        if j == '':
          continue
        images.append({'Class':t,'Image':j})


  # Partition Images into Training, Validation, and Testing
  for c in folders:
      os.makedirs(f'{TRAIN_DIRECTORY}{c}', exist_ok=True)
      os.makedirs(f'{VALID_DIRECTORY}{c}', exist_ok=True)
      os.makedirs(f'{TEST_DIRECTORY}{c}', exist_ok=True)

  counter=0
  for c in folders:
      numOfFiles = len(next(os.walk(f'{SOURCE_DIRECTORY}{c}/'))[2])
      for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles*TRAIN_SPLIT)):
          shutil.move(files, f'{TRAIN_DIRECTORY}{c}')

      for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles*VALID_SPLIT)):
          shutil.move(files, f'{VALID_DIRECTORY}{c}')

      for files in glob(f'{SOURCE_DIRECTORY}{c}/*'):
          shutil.move(files, f'{TEST_DIRECTORY}{c}')
      counter+=1

  shutil.rmtree(SOURCE_DIRECTORY)

  def build_model(measure_performance:bool = True):
  ImageFile.LOAD_TRUNCATED_IMAGES = True

  train_batches = ImageDataGenerator(preprocessing_function=PREPROCESSING_METHOD).flow_from_directory(directory=TRAIN_DIRECTORY, target_size=IMAGE_SHAPE_2D, classes=CLASSES, batch_size=BATCH_SIZE)
  valid_batches = ImageDataGenerator(preprocessing_function=PREPROCESSING_METHOD).flow_from_directory(directory=VALID_DIRECTORY, target_size=IMAGE_SHAPE_2D, classes=CLASSES, batch_size=BATCH_SIZE, shuffle=False)
  test_batches =  ImageDataGenerator(preprocessing_function=PREPROCESSING_METHOD).flow_from_directory(directory=TEST_DIRECTORY, target_size=IMAGE_SHAPE_2D, classes=CLASSES, batch_size=BATCH_SIZE, shuffle=False)

  input_shape = IMAGE_SHAPE_3D
  nclass = len(CLASSES)
  epoch = EPOCHS
  base_model = BASE_MODEL
  base_model.trainable = False

  model = Sequential()
  if len(BASE_MODEL(tf.zeros((1, *IMAGE_SHAPE_3D))).shape) == 4:
      model.add(base_model)
      model.add(Identity())
      model.add(GlobalAveragePooling2D())
      model.add(Dropout(0.2))
      model.add(Dense(256, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(128, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(64, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(nclass, activation='softmax'))
  else:
      model.add(InputLayer(input_shape))
      model.add(Rescaling(scale=1/127.5, offset=-1))
      model.add(base_model)
      model.add(Dense(512, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(256, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(128, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(64, activation='relu'))
      model.add(Dropout(0.2))
      model.add(Dense(nclass, activation='softmax'))

  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
  es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1 ,  patience = PATIENCE)

  model.summary()

  fitted_model= model.fit(x=train_batches, validation_data=valid_batches, epochs=epoch, callbacks=[es])
  score, accuracy = model.evaluate(x=test_batches, batch_size=BATCH_SIZE)

  print('\n')
  print(Fore.GREEN + u'\n\u2713 ' + f'Loss ==> {score}')
  print(Fore.GREEN + u'\n\u2713 ' + f'Accuracy ==> {accuracy}')

  plt.rcParams["figure.figsize"] = (15,8)

  if measure_performance:
    plt.plot(fitted_model.history['accuracy'])
    plt.plot(fitted_model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    plt.plot(fitted_model.history['loss'])
    plt.plot(fitted_model.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    y_pred = model.predict(test_batches)

    ax = sns.heatmap(confusion_matrix(test_batches.classes, y_pred.argmax(axis=1)), annot=True, cmap='Blues', fmt='g')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(CLASSES)
    ax.yaxis.set_ticklabels(CLASSES)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
return model

def predict_image_class(img, model):
  img = np.expand_dims(img, axis=0)
  preprocessed_image = image_preprocess(img)
  tensor = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)

  print(Fore.GREEN + u'\n\u2713 ' + f'Model Output ==> {CLASSES[np.argmax(model.predict(tensor))]}')
  print(Style.RESET_ALL)

  #Show image
  def show_image(img):
  img = Image.fromarray(img)
  display(img)

  #url to image
  def url_to_image(url):
  image_url = tf.keras.utils.get_file(origin=url)
  img = image.load_img(image_url, target_size=IMAGE_SHAPE_2D)
  img = np.expand_dims(img, axis=0)
  return np.vstack([img])[0]

  #get image
  def path_to_image(image_name = None):
  for root, dirs, files in os.walk(REFACTORED_DIRECTORY):
          if image_name in files:
              image_name = os.path.join(root, image_name)
              break

  img = Image.open(image_name)
  img = img.resize(IMAGE_SHAPE_2D)
  img = np.expand_dims(img, axis=0)
  return np.vstack([img])[0]

  #implement image processing method
  def image_preprocess(img):
  return PREPROCESSING_METHOD(img)

  #Implement Lime XAI
  def explain_image_lime(img, model):
  # Temporarily disable output stream, preventing unnecesarry output
  with IgnorePrints():
      preprocessedImage = image_preprocess(img)

      explainer = lime_image.LimeImageExplainer()
      explanation = explainer.explain_instance(np.asanyarray(preprocessedImage).astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)

      temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

  return mark_boundaries(img, mask)

def show_imgwithheat(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img

    def explain_image_grad_cam(img, model):
  assert len(model.layers[1].get_output_at(0).get_shape().as_list()) == 4, \
    f'Expected the number of dimensions in the layer to be 4, ' \
    f'found {len(model.layers[1].get_output_at(0).get_shape().as_list())}.'

  image_path = './assets/buffer.jpg'

  Image.fromarray(img).save(image_path)

  img = preprocess_image(image_path, target_size=IMAGE_SHAPE_2D)
  preprocessed_image = image_preprocess(img)

  heatmap = grad_cam(
    model, preprocessed_image,
    layer_name = model.layers[1].name,
  )

  img_arr = show_imgwithheat(image_path, heatmap)

  os.remove(image_path)

  return img_arr

  def explain_image_grad_cam_plus_plus(img, model):
  assert len(model.layers[1].get_output_at(0).get_shape().as_list()) == 4, \
    f'Expected the number of dimensions in the layer to be 4, ' \
    f'found {len(model.layers[1].get_output_at(0).get_shape().as_list())}.'

  image_path = './assets/buffer.jpg'

  Image.fromarray(img).save(image_path)

  img = preprocess_image(image_path, target_size=IMAGE_SHAPE_2D)
  preprocessed_image = image_preprocess(img)

  heatmap = grad_cam_plus(
    model, preprocessed_image,
    layer_name = model.layers[1].name,
  )

  show_imgwithheat(image_path, heatmap)

  img_arr = show_imgwithheat(image_path, heatmap)

  os.remove(image_path)

  return img_arr

  def plot_XAI(img, model):
  plt.rcParams["figure.figsize"] = (10,10)
  fig, ax = plt.subplots(2,2)
  ax[0,0].imshow(img)
  ax[0,1].imshow(explain_image_lime(img, model))
  ax[1,0].imshow(explain_image_grad_cam(img, model))
  ax[1,1].imshow(explain_image_grad_cam_plus_plus(img, model))

  ax[0, 0].set_title("Original Image")
  ax[0, 1].set_title("LIME")
  ax[1, 0].set_title("Grad-CAM")
  ax[1, 1].set_title("Grad-CAM++")

  fig.tight_layout()
  plt.show()
  plt.rcParams["figure.figsize"] = (15,8)

  os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

load_dataset()
prime_dataset()
model = build_model(measure_performance=True)

img = url_to_image('https://www.enr.com/ext/resources/News/2016/September/north_carolina_hurricane_matthew.jpg')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('05_01_1225.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('04_01_0005.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('06_02_2615.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('06_03_1742.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('06_04_0780.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('01_01_0363.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('03_0005.png')
plot_XAI(img, model)
predict_image_class(img, model)

img = path_to_image('01_02_0471.png')
plot_XAI(img, model)
predict_image_class(img, model)


    

  





 
