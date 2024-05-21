import wfdb
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd
from EMGPatternRecognition.MLUtilits import setlable, split_data_label, splite_data_to_train_test


def polt_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.figure()

    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()


def get_DataSet_fromURL(participant_number=1, gesture_number=16, trial_number=1,ploting=False ,channelPlotIndex=0):
    '''
    this function get all channels for signal fetching from data set  
    get just 5 sec from data frame  this mean 10240 samples from adc samplefequency => sample/sec  2048 * 5
    '''
    
    # Download the data
    record =wfdb.rdrecord(f'session1_participant{participant_number}_gesture{gesture_number}_trial{trial_number}',pn_dir=f'grabmyo/1.0.2/Session1/session1_participant{participant_number}'  ,sampto=10240)


    # Extract the datanm 
    data = record.p_signal  #[10240 rows x channels(32) columns ]
    ##clean channels free u-1234
    u_channel_index =[16,22,22,28]
    for i in u_channel_index:
        print(np.shape(data))
        data=np.hstack((data[:,:i],data[:,i+1:]))

    # Get the sampling frequency
    fs = record.fs
    
    #extract channels name 
    channelsName = record.sig_name
    
    # Plot the data
    if ploting :
        polt_signal(data[:,channelsName], fs, channelsName[channelPlotIndex])

    return np.array(data), fs ,channelsName


def Dissolves_array_reshape (array):
    '''
    reshape(-1, ndarray.shape[-1]): This method reshapes the array into a 2D array. 
    The -1 argument in the first dimension means that NumPy will automatically determine the number of rows based on total elements can contain in new matrix 
    so this mean rearrangement matrix elements to fit new shape

    dissolves the array into 2d array this exactly mean keep last array save  [*] for any change and merge others [ [[*],[*]] , [[*],[*]] ] => [[*],[*],[*],[*]]
    '''
    arrayMatrix= np.array(array)
    arrayMatrix= arrayMatrix.reshape(-1,arrayMatrix.shape[-1])# dissolves the array into 2d array this exactiy mean keep last array save  [*] for any change and merge others [ [[*],[*]] , [[*],[*]] ] => [[*],[*],[*],[*]]
    arrayMatrix = np.squeeze(arrayMatrix) # remove the single-dimensional entries from the shape of an array. not need but if no one`s in shape tuble no effect
    print(arrayMatrix.shape)
    return arrayMatrix.tolist()


def fetching_generate_channels_matrix(participant_list , gesture_list , trial_list):
    '''
    this function fetch all participant list for each gesture  then create matrix of channels for each participant
    so frame after that will be 10240 * size participant list 
    so will return matrix of channel or channels for all participants get in list for each separate   gesture    
    
    return: @type numpy.array with shape (participant_list.size, gesture_list.size, trial_list*row(samples 5 sec 10240),channels.size(max 32 )  )
    *we can make squeezing for remove any one`s in shape  if return one participant so will still have gestures row 3d or 4d array
    
    after study the data set we found that the data set have 3 session  43 participant and 16 gesture and 7 trial

    @return : np.array(participant_matrix),fs,_channelsName
    * participant_matrix is a matrix make squeezing for remove any one`s in shape if return one participant30

    '''
    participant_matrix = [] #shape (43,16,7,10240,32) 43 row inside each   16 row inside each  16 row have  7 row for 10240 row for 32 column
    gesture_matrix = []  #shape (16,7,10240,32)16 row for 7row for 10240 row for 32 column
    channels_matrix_trials =[]  #shape (7,10240,32) 7row for 10240 row for 32 column
    _channelsName =[]

    _fs=0
    for participant in participant_list:
        for gesture in gesture_list:
            for trial in trial_list:
                data , fs ,channelsName = get_DataSet_fromURL(participant, gesture, trial)
                channels_matrix_trials.append(data)
                _channelsName =  channelsName.copy()
                _fs=fs

                print(np.shape(channels_matrix_trials))

            channels_matrix_trials=Dissolves_array_reshape(channels_matrix_trials)
            gesture_matrix.append(channels_matrix_trials)
            channels_matrix_trials= []

            
            print(np.shape(gesture_matrix))
        participant_matrix.append(gesture_matrix)
                
        
            
    return np.array(participant_matrix),fs,_channelsName



'''

example for function 
matrix = fetching_generate_channels_matrix (participant_list=[1],gesture_list=[16,17],trial_list=[1,2,3,4,5,6,7])
print(matrix.shape)
#shape=(1, 2, 14, 10240, 32) so 1 row 2row for 14 row for 10240 row for 32 column so to print in table format we need to reshape it to 2d array
#so we will reshape it to 2d array row 2*14*10240 ,32 column
matrix1 = matrix.reshape(-1,matrix.shape[-1]) 

ndarray: This is the original NumPy array with shape (1, 2, 7, 10240, 32).
.reshape(-1, ndarray.shape[-1]): This method reshapes the array into a 2D array. The -1 argument in the first dimension means that NumPy will automatically determine the number of rows based on total elements can contain in new matrix so this mean rearrangement matrix elements to fit new shape 

#squeezematrix= np.squeeze(matrix[0,1,0:3,0:2,0:2])


'''


## machine learing model create

def startTraningModel_binaryClassification(objectsClassificationMatrix,model_name ,LEARNING_RATE = 0.01,EPOCHS = 200,BATCH_SIZE = 32):
    '''
    this function for build any model for classification between two type of object
    @para : objectsClassificationMatrix => 2d numpy array just must shape (any 2 gesture ,windows, feature  )

    @return model trand can make prediction for it
    '''
    #################################################################
    ##Machine learing  feacture extraction + encoding

    """
    for start ml i need matrix shape (n_window,n_feature )

    start the machine learning model 

    first prepare the data to be ready for the machine learning model
    """
    dataChannelOne =objectsClassificationMatrix[0]
    dataChannel2 = objectsClassificationMatrix[1]



    completmatrix = setlable(dataChannelOne, 0)

    print(type(dataChannelOne))
    print(pd.DataFrame(completmatrix))

    print("start the machine learning model")
    tensorMatrix = tf.Variable(completmatrix, name="tensorMatrix channel 1")
    print(tensorMatrix.shape)
    tensorpanda = pd.DataFrame(tensorMatrix)
    print(tensorpanda)

    print("start the second channel")
    print("/n")
    completmatrix2 = setlable(dataChannel2, 1)
    tensorMatrix2 = tf.Variable(completmatrix2, name="tensorMatrix channel 2  ")  # convert the data to tensor
    print(tensorMatrix2.shape)
    tensorpanda2 = pd.DataFrame(tensorMatrix2)
    print(tensorpanda2)

    """_summary_
    now after to equip each data with the label we need to to merge the two data to be one data 
    then we will shuffle the data to be random
    """

    mergedTensor = tf.concat([tensorMatrix, tensorMatrix2], axis=0)  # merge the two data
    print(mergedTensor.shape)
    print(pd.DataFrame(mergedTensor))
    # tensorShaffle = tf.random.shuffle(mergedTensor, seed=42)    #shuffle the data

    """_summary_
    after that we need to extract the data to be the input and the output split x and y 
    after that we need to split the data to be training and testing data and we will use 80% for training and 20% for testing
    """
    print("split the data to be input and output")

    x_train, x_test, y_train, y_test, predecited__x, predicted_y = splite_data_to_train_test(mergedTensor, train_size=0.8, test_size=0.2,random_state=42, shuffle=True,predecitedFromData=False)



    print("x_train")
    print(x_train.shape)
    print("y_train")
    print(y_train.shape)
    print("x_test")
    print(x_test.shape)
    print("y_test")
    print(y_test.shape)
    print(pd.DataFrame(y_test))
    # print(predecited__x.shape)
    # print(predicted_y.shape)
    print(pd.DataFrame(predecited__x))
    print(pd.DataFrame(predicted_y))
    # print( np.in1d(x_train, predecited__x))
    # print( "/n" )
    # print( np.in1d(x_test, x_train))

    '''
    all is done preparing data now we need to start the machine learning model

    first we need to create the model that just differentiate between the two channels signals 
    '''

    # Set the random seed
    tf.random.set_seed(42)
    # Create the model
    model_10 = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=(x_train.shape[1],)),  # input layer (the input shape is the number of features,)
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name=model_name)
    # Compile the model with the ideal learning rate
    model_10.compile(loss="binary_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                     # to adjust the learning rate, you need to use tf.keras.optimizers.Adam (not "adam")
                     metrics=["accuracy"])

    History = model_10.fit(x_train, y_train, epochs=EPOCHS ,batch_size=BATCH_SIZE,verbose=2)
    print("/n test value ")
    model_10.evaluate(x_test, y_test ,batch_size=BATCH_SIZE, verbose=2)
    #plot_decision_boundary(model_10,x_train,y_train )
    model_10.summary()
    print(pd.DataFrame(History.history))
    pd.DataFrame(History.history).plot()
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()
    # Fit the model for 20 epochs (5 less than before)

    return model_10


def generateTraningTestingData (objectsClassificationMatrix)  :
    '''
    this function for dived any feature matrix object classify to x_train, x_test, y_train, y_test, predecited__x, predicted_y
     @para : objectsClassificationMatrix => 2d numpy array just must shape (any 2 gesture ,windows, feature  )
     @return  x_train, x_test, y_train, y_test, predecited__x, predicted_y
    '''
    dataGestureOne = objectsClassificationMatrix[0]
    dataGestureTwo = objectsClassificationMatrix[1]

    completmatrix = setlable(dataGestureOne, 0)

    print(type(dataGestureOne))
    print(pd.DataFrame(completmatrix))

    print("start the machine learning model")
    tensorMatrix = tf.Variable(completmatrix, name="tensorMatrix channel 1")
    print(tensorMatrix.shape)
    tensorpanda = pd.DataFrame(tensorMatrix)
    print(tensorpanda)

    print("start the second channel")
    print("/n")
    completmatrix2 = setlable(dataGestureTwo, 1)
    tensorMatrix2 = tf.Variable(completmatrix2, name="tensorMatrix channel 2  ")  # convert the data to tensor
    print(tensorMatrix2.shape)
    tensorpanda2 = pd.DataFrame(tensorMatrix2)
    print(tensorpanda2)

    """_summary_
    now after to equip each data with the label we need to to merge the two data to be one data 
    then we will shuffle the data to be random
    """

    mergedTensor = tf.concat([tensorMatrix, tensorMatrix2], axis=0)  # merge the two data
    print(mergedTensor.shape)
    print(pd.DataFrame(mergedTensor))
    # tensorShaffle = tf.random.shuffle(mergedTensor, seed=42)    #shuffle the data

    """_summary_
    after that we need to extract the data to be the input and the output split x and y 
    after that we need to split the data to be training and testing data and we will use 80% for training and 20% for testing
    """
    print("split the data to be input and output")

    x_train, x_test, y_train, y_test, predecited__x, predicted_y = splite_data_to_train_test(mergedTensor,
                                                                                             train_size=0.8,
                                                                                             test_size=0.2,
                                                                                             random_state=42,
                                                                                             shuffle=True,
                                                                                             predecitedFromData=False)

    return  x_train, x_test, y_train, y_test, predecited__x, predicted_y