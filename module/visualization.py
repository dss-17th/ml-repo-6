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


def show_predict(model, weight, X_valid, y_valid):
    model.load_weights(weight)
    
    idx = np.random.choice(range(len(X_valid)), 16, replace=False)
    datas = X_valid[idx]
    labels = y_valid[idx]
    
    pred = model.predict(datas).squeeze(-1)
    pred_label = [(1 if data >= 0.5 else 0) for data in pred]
    
    fig, ax = plt.subplots(4, 4, figsize=(16, 12))
    
    num = 0
    for row in range(4):
        for col in range(4):
          ax[row, col].axis("off")
          ax[row, col].imshow(datas[num])
          color = "green" if pred_label[num] == labels[num] else "red"
          title = "photo" if pred_label[num] == 1 else "painting"
          
          ax[row, col].set_title(title, fontsize=15)
          ax[row, col].title.set_color(color)
          num += 1
    
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)", fontsize=20)
    plt.show()