import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def plot_scatter_feature_2d(feature_gesture1, feature_gesture2, Feature_names):
    feature_gesture1_matrix_np = np.array(feature_gesture1)
    feature_gesture2_matrix_np = np.array(feature_gesture2)

    feature_gesture1_matrix_dp = pd.DataFrame(feature_gesture1_matrix_np)
    feature_gesture2_matrix_dp = pd.DataFrame(feature_gesture2_matrix_np)
    print(feature_gesture2_matrix_dp)
    print(feature_gesture1_matrix_dp)



    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
    axes = axes.flatten()  # reshape 2d => 1d
    for i, ax in enumerate(axes):
        print(f"next layer {i}")
        ax.scatter(feature_gesture1_matrix_np[:,i], feature_gesture1_matrix_np[:,i+1], s=20,
                    c="r", alpha=0.5, )
        ax.scatter(feature_gesture2_matrix_np[:, i], feature_gesture2_matrix_np[:, i+1], s=20,
                   c="b", alpha=0.5, )
        ax.set_title(f"feature {i}-{Feature_names[i]}")
        ax.set_xlabel(f"feature_gesture1")
        ax.set_ylabel(f"feature_gesture1")

    plt.tight_layout()

    plt.grid(True)
    plt.show()

def plot_scatter_feature_2d_single_2muliy2(feature_gesture1, feature_gesture2, Feature_names):
    '''
    i try to plot data as point so each feature is part of point ex: rms =\>x , iemg => y
    '''
    feature_gesture1_matrix_np = np.array(feature_gesture1)
    feature_gesture2_matrix_np = np.array(feature_gesture2)
    gesture1Label =np.zeros(shape=feature_gesture1_matrix_np.shape[0],dtype= np.int32)
    gesture2Label= np.ones(shape=feature_gesture2_matrix_np.shape[0],dtype=np.int32 )
    feature_gesture1_matrix_dp = pd.DataFrame(feature_gesture1_matrix_np)
    feature_gesture2_matrix_dp = pd.DataFrame(feature_gesture2_matrix_np)
    print(feature_gesture2_matrix_dp)
    print(feature_gesture1_matrix_dp)

    totalFeatures= np.concatenate((feature_gesture1_matrix_np,feature_gesture2_matrix_np),axis=0)
    cmapLabel=np.concatenate((gesture1Label,gesture2Label),axis=0)
    plt.figure(figsize=(10,10))
    plt.scatter(x=totalFeatures[:,0],y=totalFeatures[:,1],c=cmapLabel, alpha=0.7, cmap="RdYlBu")


    plt.tight_layout()

    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, x1, x2, y=[0, 1]):
  """
  Plots the decision boundary created by a model predicting on X.
  @ model classifier want draw
  @ x => one sample for have for each objects classified @type : numpy array shape (2,feature size) 2 for binary classification
  @ y => label vector this just for map color for scatter technique just give color for each value in matrix map with x and y ex: 0=> red 1=> blue
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  shapeForMatrix = x1.shape[1]
  realxx, realyy,mixYX= [],[],[]
  for i in range(shapeForMatrix): # move for each feature
      # Define the axis boundaries of the plot and create a meshgrid
      x_min, x_max = x1[:, i].min() - 0.1, x1[:, i].max() + 0.1
      y_min, y_max = x2[:, i].min() - 0.1, x2[:, i].max() + 0.1
      xx, yy = np.meshgrid(np.linspace(x_min, x_max, shapeForMatrix),np.linspace(y_min, y_max, shapeForMatrix))

      # Create X values (we're going to predict on all of these)
      ravelx= xx.ravel()
      ravely = yy.ravel()
      realxx.append(ravelx.tolist())
      realyy.append(ravely.tolist())
      x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html create matrix parse  [[x1,y1].....](13*13,2)
      x_in= x_in.ravel() #(13*13*2)[flatte]row vector
      mixYX.append(x_in)


  mixYX=np.array(mixYX).T
  # Make predictions using the trained model
  y_pred = model.predict(mixYX)
  # y_pred= np.random.rand(13*13,1)

  # Check for multi-class
  if model.output_shape[-1] > 1: # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
  # if False :
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(shapeForMatrix,shapeForMatrix)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1))# max just get same value but because axis one get max value for each row as squeeze
    y_pred= y_pred.reshape(-1,shapeForMatrix)

  # Plot decision boundary
  # plt.contourf(mixYX[0:(y_pred.shape[0]),:], mixYX[0:y_pred.shape[0],:], y_pred, cmap="RdYlBu", alpha=0.7)
  plt.contourf(realxx, realyy, y_pred, cmap="RdYlBu", alpha=0.7)
  # plt.scatter(x1[:, 0], x1[:, 0], c="b", s=40, cmap="RdYlBu")
  # plt.scatter(x1[:, 1], x1[:, 1], c="r", s=40, cmap="RdYlBu")
  plt.xlim(mixYX[0:(shapeForMatrix*2),:].min(), mixYX[0:(shapeForMatrix*2),:].max())
  plt.ylim(mixYX[0:(shapeForMatrix*2),:].min(), mixYX[0:(shapeForMatrix*2),:].max())
  plt.show()


def plot_decision_boundaryOriginal(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[-1] > 1:  # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap="RdYlBu", alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="RdYlBu")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
# x= np.c_[np.random.randint(1,10,1),np.random.randint(10,20,1)]
# plot_decision_boundary(model=None, X=x )