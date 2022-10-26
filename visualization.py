import matplotlib.pyplot as plt
import numpy as np 
import torch

#########################
#
# Plot curve
#
#########################

def plot_data_grid(data, index_data, nRow, nCol):
    
    size_col = 1.5
    size_row = 1.5

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(nCol * size_col, nRow * size_row))

    for i in range(nRow):
        for j in range(nCol):

            k       = i * nCol + j
            index   = index_data[k]

            axes[i, j].imshow(data[index], cmap='gray', vmin=0, vmax=1)
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)

    plt.show()
    
def plot_data_tensor_grid(data, index_data, nRow, nCol):
    
    size_col = 1.5
    size_row = 1.5

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(nCol * size_col, nRow * size_row))

    data = data.detach().cpu().squeeze(axis=1)

    for i in range(nRow):
        for j in range(nCol):

            k       = i * nCol + j
            index   = index_data[k]

            axes[i, j].imshow(data[index], cmap='gray', vmin=0, vmax=1)
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)

    plt.show()
    
def plot_curve_error(data_mean, data_std, x_label, y_label, title):

    plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3
    
    plt.plot(range(len(data_mean)), data_mean, '-', color = 'red')
    plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, facecolor = 'blue', alpha = alpha) 
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    plt.show()
    
def print_curve(data, index):
    
    for i in range(len(index)):

        idx = index[i]
        val = data[idx]

        print('index = %2d, value = %12.10f' % (idx, val))
        
def get_data_last(data, index_start):

    data_last = data[index_start:]

    return data_last

def get_max_last_range(data, index_start):

    data_range = get_data_last(data, index_start)
    value = data_range.max()

    return value

def get_min_last_range(data, index_start):

    data_range = get_data_last(data, index_start)
    value = data_range.min()

    return value

#########################
#
# Functions for presenting the results
#
#########################

def plot_input_images(data, text=None):

    if text:
        print(text)
        print('') 

    nRow = 8
    nCol = 6
    index_data  = np.arange(0, nRow * nCol)
    image,_ = data[index_data]
    image = image[0]
    
    plot_data_grid(image, index_data, nRow, nCol)
    
def plot_label_images(data, text=None):

    if text:
        print(text)
        print('') 

    nRow = 8
    nCol = 6
    index_data  = np.arange(0, nRow * nCol)
    _ , label = data[index_data]
    label = label[0]
    
    plot_data_grid(label, index_data, nRow, nCol)
    
def plot_prediction_images(model, data, text=None):
    if text:
        print(text)
        print('') 

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    nRow = 8
    nCol = 6
    index_data          = np.arange(0, nRow * nCol)
    image, _      = data[index_data] 
    image         = image[0].unsqueeze(dim=1).to(device)
    prediction_train    = model(image)
    
    plot_data_tensor_grid(prediction_train, index_data, nRow, nCol)

def plot_loss(loss_mean, loss_std, text=None):

    if text:
        print(text)
        print('') 

    plot_curve_error(loss_mean, loss_std, 'epoch', 'loss', text)