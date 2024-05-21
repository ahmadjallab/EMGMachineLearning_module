import numpy as np
import math
import tensorflow as tf
def setlable (data, label):
    """
    this finction add lable for all sup list of array 
    we will add vector for matrix  

    Args:
        data (tensorflow array ): _description_
        label (number ): output 

    Returns:
        tensorarray : appand the label to the data for each window 
        
        
        
    """
    
    for i in range(len(data)):
        data[i].append(label)
    return data




def split_data_label(data):
    """
    split the data to data and label 

    Args:
        data (tensorflow array ): _description_

    Returns:
        tensorarray : data and label 
    """
    data = np.array(data)
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y



def splite_data_to_train_test(data, train_size=0.8,test_size=0.2,random_state=42,shuffle=True ,predecitedFromData=False):
    """
    splite the data to train and test 
    and shuffle the data 

    Args:
        data (tensorflow array ): this data will be splited to train and test
        test_size (number ): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the  test split
        train_size (number ): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split . If None, the value is automatically set to the complement of the test size.
        random_state (number ): seed for the random number generator
        shuffle (bool ): whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        predecitedFromData : if you not have enough data so you can split 10 % for predecited data from data set 
    Returns: 
        x_train, x_test, y_train, y_test
        tensorarray : train and test data 
        the data will be splited to x_train, x_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    data_set =data
 
    if predecitedFromData:
        data_set = tf.random.shuffle(data, seed=42)   #shuffle the data for slicing 0 or 1 label
        predecited_ = data_set[:math.ceil(len(data) *0.1)]
        data_set = data_set[math.ceil(len(data) *0.1):]
        predecited__x,predicted_y = split_data_label(predecited_)
    else:
        predecited__x = []
        predicted_y = []
    
    x, y = split_data_label(data_set)
    y_int =np.array( y, dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(x, y_int, test_size=test_size,random_state=42,shuffle=True)
    return x_train, x_test, y_train, y_test ,predecited__x,predicted_y



