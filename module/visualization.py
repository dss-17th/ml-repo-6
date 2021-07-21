import matplotlib.pyplot as plt
import numpy as np


def make_scores_graph(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history.history['accuracy'], 'b-', label='accuracy')
    plt.plot(history.history['val_accuracy'], 'r-', label='val_accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r-', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.show()