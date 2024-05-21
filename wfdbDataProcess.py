import wfdb
import numpy as np
import matplotlib.pyplot as plt
from EMGPatternRecognition.feature_extraction import features_estimation ,extract_features_toTensorflow ,Handel_timeD_featuresEngeenring_withReshape
from EMGPatternRecognition.digital_processing import bp_filter,bp_filter_ndimSignalCOlume
import pandas as pd

# Specify the path to your .dat file and .hea file
dat_file_path = 'session1_participant1_gesture10_trial1'
hea_file_path = 'session1_participant1_gesture10_trial1'


def polt_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.figure()

    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()



#recordHeader = wfdb.rdheader(hea_file_path)
#print (recordHeader.sig_name )
#print (recordHeader.e_d_signal)
#print (recordHeader.units)
#print(recordHeader.fs)
#print(recordHeader.__dict__)
#signal,field = wfdb.rdsamp(dat_file_path, sampfrom=0, channels=[0],return_res=16)
#record = wfdb.rdrecord(dat_file_path)
#recod =wfdb.wrsamp(dat_file_path, recordHeader.fs, units=recordHeader.units,sig_name=recordHeader.sig_name)
#ecg_record = wfdb.rdheader('100')8]]]]]

## /**************************** this section work 

recordurl10  =wfdb.rdrecord('session1_participant1_gesture16_trial1',pn_dir='grabmyo/1.0.2/Session1/session1_participant1' ,channels=[0,15,2] ,sampto=10240)
print("channels matrix")
print (pd.DataFrame(recordurl10.p_signal[:,0:2],columns=recordurl10.sig_name[0:2]))
recordurl17  =wfdb.rdrecord('session1_participant1_gesture17_trial1',pn_dir='grabmyo/1.0.2/Session1/session1_participant1' ,channels=[3,9,2] ,sampto=10240)
#print(recordurl.__dict__)
print("\n")
#print(recordurl)
#wfdb.plot_wfdb(recordurl, title='Record 100 from Physionet Challenge 2015',time_units='seconds')
print(recordurl10.__dict__)
print("\n")
print(recordurl17.__dict__)

#we are take there 3 channels so we need divide the signal into 3 channels

#first = np.concatenate((recordurl10.p_signal[:,0],recordurl10.p_signal[:,1],recordurl10.p_signal[:,2]) ,axis=0)
first = recordurl10.p_signal[:,0:2]
#second =np.concatenate((recordurl17.p_signal[:,0],recordurl17.p_signal[:,1],recordurl10.p_signal[:,2]) ,axis=0)
second =recordurl17.p_signal[:,1:2]

print("compare the two channels")
print(np.array(first)==np.array(second))

#plot_signal(first, recordurl.fs, recordurl.sig_name[0])
print("\n")
#print("First Channel")
#print (first)


## plot the signal befor filter
#np.arange(start, stop, step): This NumPy function creates an array of evenly spaced values within the given range.
#start: The starting value of the sequence, which is 0 in this case.
#stop: The end value of the sequence, which is the length of first divided by the sampling frequency.
#step: The step size between each value in the sequence, calculated as 1/recordurl.fs.
#Dividing by the sampling frequency is necessary to convert the number of samples into time units. 

#polt_signal(first, recordurl.fs, "First Channel")
#print("this give mean for first channel")
#print (np.mean(first))





print("\n")
plt.figure()
channel_1_filter_matrix = bp_filter_ndimSignalCOlume(first,20, 450, recordurl10.fs)

channel_1_filter= bp_filter(first[:,0],20, 450, recordurl10.fs, plot=False)
channel_2_filter = bp_filter(second[:,0], 20, 450, recordurl17.fs, plot=False)
#channel_3_filter = bp_filter(third, 20, 450, recordurl.fs, plot=False)
# # Assuming signal is your array of data

# frame = 10  # Size of each window
# step = 5    # Step size for sliding window

# for i in range(frame, len(chennel_1_filter), step):
#     window_data = chennel_1_filter[i - frame:i]  # Extract the data within the window
    
#     # Plotting
#     plt.figure()  # Create a new figure for each window
#     plt.plot(window_data)  # Plot the windowed data
#     plt.title('Window {}'.format(i))  # Set title indicating the window index
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.show()  # Display the plot

# Assuming signal is your array of data
signal = channel_1_filter
frame = 250   # Size of each window
step  = 250 # Step size for sliding window

# for i in range(frame, len(signal), step):
#     window_data = signal[i - frame:i]  # Extract the data within the window
    
#     # Create a DataFrame for the windowed data
#     df = pd.DataFrame(window_data, columns=['Value'])
#     df.index.name = 'Index'
    
#     # Display the DataFrame as a table
#     print(f"Window {i}:")
#     print(df)
#     print('-' * 20)
# print(list(range(frame, signal.size,step)))
#    for i in range(frame, signal.size, step):




total_feature_matrix_channel1, features_names_channel2 , _ = features_estimation(channel_1_filter,recordurl10.sig_name[0], recordurl10.fs,frame, step,plot=False)
print ("total_feature_matrix_channel1")
print (total_feature_matrix_channel1.shape)
time_features_channel1 = total_feature_matrix_channel1.loc[['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'WL','ACC','M2','DVARV', 'DASDV', 'WAMP', 'MYOP','IE']]
print (pd.DataFrame(time_features_channel1))


total_feature_matrix_ch2, features_names_channel2,_ = features_estimation(channel_2_filter,recordurl17.sig_name[1], recordurl17.fs,frame, step,plot=False)
time_features_channel2 = total_feature_matrix_ch2.loc[['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'WL','ACC','M2','DVARV', 'DASDV', 'WAMP', 'MYOP','IE' ]]
print ("time_features_channel2")
print (pd.DataFrame(time_features_channel2))

#reshape the data to be to window per features 
dataChannelOne = extract_features_toTensorflow(time_features_channel1)
#print (time_features)
#print (features_names)

dataChannel2 = extract_features_toTensorflow(time_features_channel2)


print("comper the two channels features")
print (dataChannelOne == dataChannel2)

#/**** matrix features
features_estimation_2dMatrix , _ = Handel_timeD_featuresEngeenring_withReshape (channel_1_filter_matrix,recordurl10.sig_name, recordurl10.fs,frame, step,plot=False)

#################################################################
##Machine learing  feacture extraction + encoding

import tensorflow as tf 

from EMGPatternRecognition.MLUtilits import setlable ,split_data_label , splite_data_to_train_test

"""
for start ml i need matrix shape (n_window,n_feature )

start the machine learning model 

first prepare the data to be ready for the machine learning model
"""

completmatrix= setlable(dataChannelOne, 0)

print(type(dataChannelOne))
print(pd.DataFrame(completmatrix))

print("start the machine learning model")
tensorMatrix = tf.Variable(completmatrix ,name="tensorMatrix channel 1")
print(tensorMatrix.shape)
tensorpanda = pd.DataFrame(tensorMatrix)
print(tensorpanda)


print ("start the second channel")
print("/n")
completmatrix2= setlable(dataChannel2, 1)
tensorMatrix2 = tf.Variable(completmatrix2 ,name="tensorMatrix channel 2  ")    #convert the data to tensor
print(tensorMatrix2.shape)
tensorpanda2 = pd.DataFrame(tensorMatrix2)
print(tensorpanda2)



"""_summary_
now after to equip each data with the label we need to to merge the two data to be one data 
then we will shuffle the data to be random
"""

mergedTensor = tf.concat([tensorMatrix, tensorMatrix2], axis=0)  #merge the two data
print(mergedTensor.shape)
print(pd.DataFrame(mergedTensor))
#tensorShaffle = tf.random.shuffle(mergedTensor, seed=42)    #shuffle the data

"""_summary_
after that we need to extract the data to be the input and the output split x and y 
after that we need to split the data to be training and testing data and we will use 80% for training and 20% for testing
"""
print("split the data to be input and output")

x_train, x_test, y_train, y_test, predecited__x , predicted_y = splite_data_to_train_test(mergedTensor, train_size=0.6,test_size=0.2,random_state=42,shuffle=True,predecitedFromData=False)

print( "x_train" )
print( x_train.shape)
print( "y_train" )
print( y_train.shape)
print( "x_test")
print( x_test.shape)
print( "y_test" )
print( y_test.shape )
print( pd.DataFrame(y_test) )
# print(predecited__x.shape)
# print(predicted_y.shape)
print( pd.DataFrame(predecited__x) )
print( pd.DataFrame(predicted_y))
#print( np.in1d(x_train, predecited__x))
#print( "/n" )
#print( np.in1d(x_test, x_train))


'''
all is done preparing data now we need to start the machine learning model

first we need to create the model that just differentiate between the two channels signals 
'''


# Set the random seed
tf.random.set_seed(42)  
# Create the model
model_10 = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(x_train.shape[1],)), # input layer (the input shape is the number of features,)
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid"),
], name="MODEL_10_SEQ")
# Compile the model with the ideal learning rate
model_10.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam( learning_rate=0.01 ), # to adjust the learning rate, you need to use tf.keras.optimizers.Adam (not "adam")
                metrics=["accuracy"])

model_10.fit(x_train, y_train, epochs=150)

model_10.summary()
# Fit the model for 20 epochs (5 less than before)


'''
now we need to evaluate the model 

'''


recordurl10_prediction  =wfdb.rdrecord('session1_participant2_gesture16_trial2',pn_dir='grabmyo/1.0.2/Session1/session1_participant2' ,channels=[0,15,2] ,sampto=10240)
first_prediction = recordurl10_prediction.p_signal[:,0]
channel_1_filter_prediction= bp_filter(first_prediction,20, 450, recordurl10_prediction.fs, plot=False)

total_feature_matrix_channel1_prediction, features_names_channel2_prediction = features_estimation(channel_1_filter_prediction,recordurl10_prediction.sig_name[0], recordurl10_prediction.fs,frame, step,plot=False)
print ("total_feature_matrix_channel1_prediction")
print (total_feature_matrix_channel1_prediction.shape)
time_features_channel1_prediction = total_feature_matrix_channel1_prediction.loc[['VAR', 'RMS', 'IEMG','SSI', 'MAV', 'WL','ACC','M2','DVARV', 'DASDV', 'WAMP', 'MYOP','IE']]
print (pd.DataFrame(time_features_channel1))
#reshape the data to be to window per features 
dataChannelOne_prediction = extract_features_toTensorflow(time_features_channel1_prediction)

##Machine learing  feacture extraction + encoding
#completmatrix_prediction= setlable(dataChannelOne_prediction, 0) # not need add the label this prediction data 
tensorMatrix_prediction = tf.Variable(dataChannelOne_prediction ,name="tensorMatrix channel 1 prediction")
print ("tensorMatrix_prediction")
print (tensorMatrix_prediction.shape)
tensorpanda_prediction = pd.DataFrame(tensorMatrix_prediction)
print(tensorpanda_prediction)

# Set the model's prediction
predictions = model_10.predict(tensorMatrix_prediction)
output = predictions > 0.5
print (predictions)
print (output)
