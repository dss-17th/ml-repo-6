import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_base_model(model, input_shape=(224, 224, 3), trainable=False):
    
    if model == "mobilenet":
        url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        base_model = hub.KerasLayer(url, input_shape=input_shape)
    
    elif model == "resnet":
        url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
        base_model = hub.KerasLayer(url, input_shape=input_shape)
        
    elif model == "inception_resnet":
        url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5"
        base_model = hub.KerasLayer(url, input_shape=input_shape)
        
    elif model == "inception_v3":
        url = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5'
        base_model = hub.KerasLayer(url, input_shape=input_shape)
        
    elif model == "efficientnet":
        url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
        base_model = hub.KerasLayer(url, input_shape=input_shape)
        
    if not trainable:
        base_model.trainable = False
        
    return base_model


def setting_callback(name, monitors, min_delta=0.0001, acc_patience=10, loss_patience=7):
    save_path = "/content/drive/MyDrive/datas/model_result/"
    
    callbacks = []
    if "val_accuracy" in monitors:
        earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=min_delta,patience=acc_patience)
        cp_callback = ModelCheckpoint(save_path+f'{name}_acc.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
        callbacks.append(earlystop_callback)
        callbacks.append(cp_callback)
        
    if "val_loss" in monitors:
        earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta,patience=loss_patience)
        cp_callback = ModelCheckpoint(save_path+f'{name}_loss.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        callbacks.append(earlystop_callback)
        callbacks.append(cp_callback)
        
    return callbacks
    
    
def make_network(base_model, name):
  if name == "inception_resnet":
    model = tf.keras.Sequential([
          base_model,
          tf.keras.layers.Dense(768, activation='elu'),
          tf.keras.layers.Dense(384, activation='elu'),
          tf.keras.layers.Dense(192, activation='elu'),
          tf.keras.layers.Dense(96, activation='elu'),
          tf.keras.layers.Dense(32, activation='elu'),
          tf.keras.layers.Dense(8, activation='elu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])
  elif name == "resnet":
    model = tf.keras.Sequential([
          base_model,
          tf.keras.layers.Dense(256, activation='elu'),
          tf.keras.layers.Dense(128, activation='elu'),
          tf.keras.layers.Dense(64, activation='elu'),
          tf.keras.layers.Dense(32, activation='elu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])
  elif name == "efficientnet":
    model = tf.keras.Sequential([
          base_model,
          tf.keras.layers.Dense(512, activation='elu'),
          tf.keras.layers.Dense(256, activation='elu'),
          tf.keras.layers.Dense(128, activation='elu'),
          tf.keras.layers.Dense(64, activation='elu'),
          tf.keras.layers.Dense(32, activation='elu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])
  elif name == "inception_v3":
    model = tf.keras.Sequential([
          base_model,
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(16, activation='relu'),
          tf.keras.layers.Dense(4, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
      ])
  else:
    model = tf.keras.Sequential([
          base_model,
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(16, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])
    
  return model