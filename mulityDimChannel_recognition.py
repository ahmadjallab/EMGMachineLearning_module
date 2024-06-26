import pandas as pd
from matplotlib import pyplot as plt

from EMGPatternRecognition.MLUtilits import splite_data_to_train_test
from EMGPatternRecognition.plot import plot_scatter_feature_2d, plot_decision_boundaryOriginal, \
    plot_scatter_feature_2d_single_2muliy2, plot_decision_boundaryOriginalsvm_knn, \
    plot_svm_knn_accuracies_withRegularization
from EMGPatternRecognition.ClassificationUtility import fetching_generate_channels_matrix, \
    startTraningModel_binaryClassification, polt_signal
import numpy as np
import tensorflow as tf
from EMGPatternRecognition.digital_processing import bp_filter_ndimSignalCOlume
from EMGPatternRecognition.feature_extraction import Handel_timeD_featuresEngeenring_withReshape
from sklearn.metrics import classification_report, accuracy_score

#global_var for traning testing

gesture_name=['thumb','thumb_middle']
numberOfChannels = 16
#'VAR', 'RMS', 'IEMG','SSI', 'MAV', 'LOG', 'WL', 'ACC','M2','DVARV', 'DASDV', 'ZC', 'WAMP', 'MYOP','IE', "FR", "MNP", "TP", "MNF", "MDF", "PKF", "WENT"
extraction_features=['VAR', 'RMS']
namemodefiy =str(extraction_features).replace("[", "").replace("]", "").replace("'","").replace(" ","").replace(",","_")
modelName= f"binaryClassify-{namemodefiy}"
modelNum=0
# Define training hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 50
BATCH_SIZE = 32

thresholdAccuracy=0.5 # this for round prediction value from ANN model to apply for  classification_report, accuracy_score evaluation

#first section prepare  data set n

emg_Data_signals,fs,channelName =  fetching_generate_channels_matrix(participant_list=[5],gesture_list=[10,7], trial_list=range(1,5,1))
emg_Data_signals= np.squeeze(emg_Data_signals) #remove any one`s dim
# filter layer muilty channals

gesture_16 = emg_Data_signals[0]
gesture_17 =emg_Data_signals[1]

# polt_signal(gesture_16,2048," signal for 16")
# polt_signal(gesture_17,2048," signal for 17")

emg_filter_gesture_16 = bp_filter_ndimSignalCOlume(gesture_16,20,450,fs)

emg_filter_gesture_17 = bp_filter_ndimSignalCOlume(gesture_17,20,450,fs)

# polt_signal(emg_filter_gesture_16,2048,"filter signal for 16")
# polt_signal(emg_filter_gesture_17,2048,"filter signal for 17")

# Decrease  number of channel
emg_filter_gesture_16= emg_filter_gesture_16[:, 0:numberOfChannels]
emg_filter_gesture_17= emg_filter_gesture_17[:, 0:numberOfChannels]



# feature extraction for multi dim channel
frame=500
step=500
objects_gestures_tran_model =[]# defined as reference variable any modify in preparing section will all to variable
total_feature_estimation_gesture_16,_,_= Handel_timeD_featuresEngeenring_withReshape(filter_emgMatrix=emg_filter_gesture_16,channels_name=channelName,fs=fs ,frame=frame,step=step,extraction_features=extraction_features)
total_feature_estimation_gesture_17,_,extraction_features_name= Handel_timeD_featuresEngeenring_withReshape(filter_emgMatrix=emg_filter_gesture_17,channels_name=channelName,fs=fs ,frame=frame,step=frame,extraction_features=extraction_features)

polt_signal(np.array(total_feature_estimation_gesture_16)[:,0],2048,f"{extraction_features[0]} for {gesture_name[0]}")
polt_signal(np.array(total_feature_estimation_gesture_16)[:,1],2048,f"{extraction_features[1]} for {gesture_name[0]}")
polt_signal(np.array(total_feature_estimation_gesture_17)[:,0],2048,f"{extraction_features[0]} for {gesture_name[1]}")
polt_signal(np.array(total_feature_estimation_gesture_17)[:,1],2048,f"{extraction_features[1]} for {gesture_name[1]}")

# plot_scatter_feature_2d(total_feature_estimation_gesture_16,total_feature_estimation_gesture_17,extraction_features_name)
plot_scatter_feature_2d_single_2muliy2(total_feature_estimation_gesture_16,total_feature_estimation_gesture_17,extraction_features_name,gesture_name)



#ML traing
objects_gestures_tran_model.append(total_feature_estimation_gesture_16)
objects_gestures_tran_model.append(total_feature_estimation_gesture_17)

model = startTraningModel_binaryClassification(objects_gestures_tran_model,modelName ,LEARNING_RATE ,EPOCHS ,BATCH_SIZE )

#*******save mode para
model.save(f"./modelSaveDNN/{len(extraction_features)}D_ANN_BinaryClassification_{modelNum}.h5")
'''
now we need to evaluate the model 

'''
##data collection fetching data
emg_Data_signals_prediction,fs,channelName =  fetching_generate_channels_matrix(participant_list=[2],gesture_list=[10,7], trial_list=range(1,5,1))
emg_Data_signals_prediction= np.squeeze(emg_Data_signals_prediction) #remove any one`s dim
# filter layer muilty channals

#divied objects classifed
gesture_16_prediction = emg_Data_signals_prediction[0]
gesture_17_prediction =emg_Data_signals_prediction[1]


emg_filter_gesture_16_prediction = bp_filter_ndimSignalCOlume(gesture_16_prediction,20,450,fs)

emg_filter_gesture_17_prediction = bp_filter_ndimSignalCOlume(gesture_17_prediction,20,450,fs)

#channel drop
emg_filter_gesture_16 = emg_filter_gesture_16[:, 0:numberOfChannels]
emg_filter_gesture_17 = emg_filter_gesture_17[:, 0:numberOfChannels]

# feature extraction for multi dim channel

total_feature_estimation_gesture_16_prediction,_,_= Handel_timeD_featuresEngeenring_withReshape(filter_emgMatrix=emg_filter_gesture_16_prediction,channels_name=channelName,fs=fs ,frame=frame,step=step,extraction_features=extraction_features)
total_feature_estimation_gesture_17_prediction,_,_= Handel_timeD_featuresEngeenring_withReshape(filter_emgMatrix=emg_filter_gesture_17_prediction,channels_name=channelName,fs=fs ,frame=frame,step=step ,extraction_features=extraction_features)

tensorMatrix_prediction16 = tf.Variable(total_feature_estimation_gesture_16_prediction ,name="tensorMatrix gesture 16 prediction")

tensorMatrix_prediction17 = tf.Variable(total_feature_estimation_gesture_17_prediction,name="tensorMatrix gesture 16 prediction")
tfeg16=np.array(total_feature_estimation_gesture_16_prediction)
tfeg17= np.array(total_feature_estimation_gesture_17_prediction)
total_feature_estimation_gestures_prediction=np.concatenate((tfeg16, tfeg17))
label=np.concatenate((np.zeros(tensorMatrix_prediction16.shape[0],np.int32),np.ones(tensorMatrix_prediction17.shape[0],np.int32)))
#plot
# plot_decision_boundary(model=model, x1= np.array(total_feature_estimation_gesture_16_prediction),x2=np.array(total_feature_estimation_gesture_17_prediction))
plot_decision_boundaryOriginal(model,total_feature_estimation_gestures_prediction,label,extraction_features_name,gesture_name)
# Set the model's prediction
predictions16 = model.predict(tensorMatrix_prediction16)#0
predictions17 = model.predict(tensorMatrix_prediction17)#1


predictions16= np.squeeze(predictions16)
predictions17 = np.squeeze(predictions17)
predictions16=(predictions16 >=thresholdAccuracy).astype(int)
predictions17=(predictions17 >= thresholdAccuracy).astype(int)

predictions16_real = np.zeros(len(predictions16),dtype=np.int32)
predictions17_real = np.ones(len(predictions17),dtype=np.int32)
output16 = predictions16 > 0.5
output17 = predictions17 > 0.5
# Evaluation
print ("DNN (deep neural network )")
print("Accuracy of 16:", accuracy_score(predictions16_real, predictions16))
print("Classification Report:")
print(classification_report(np.zeros(len(predictions16)), predictions16))
def calculate_bool_matrix_percentage(predictions_bool_matrix,name):
    numOFone =0
    numOFzero =0

    for i in predictions_bool_matrix:
        if i :
            numOFone +=1
        else:
            numOFzero +=1

    percent1 = (numOFone / len(predictions_bool_matrix)) * 100
    percent0 = (numOFzero / len(predictions_bool_matrix)) * 100
    print(f'accuracy zero prediction {name}:{percent0}')
    print(f'accuracy one prediction {name}:{percent1}')
    return percent0,percent1




calculate_bool_matrix_percentage(output16,name= "gesture 16")
calculate_bool_matrix_percentage(output17 ,name = "gesture 17")

## use svm classifier

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
objects_gestures_tran_model=np.array(objects_gestures_tran_model)
objects_gestures_tran_modelX =np.concatenate((objects_gestures_tran_model[0],objects_gestures_tran_model[1]),axis=0)
x_train, x_test, y_train, y_test, _,_ =splite_data_to_train_test(objects_gestures_tran_modelX, train_size=0.8, test_size=0.2,random_state=42,shuffle=True,predecitedFromData=False)
# Initialize and train classifier
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)

# Predictions
y_pred = clf.predict(x_test)

#plot classifier
plot_decision_boundaryOriginalsvm_knn(clf,total_feature_estimation_gestures_prediction,label,extraction_features_name,gesture_name,"svm")

# Evaluation
print ("SVM (Support Vector Machines )")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

#use knn classifier
# Initialize and train classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Predictions
y_pred = knn.predict(x_test)

#plot classifier
plot_decision_boundaryOriginalsvm_knn(knn,total_feature_estimation_gestures_prediction,label,extraction_features_name,gesture_name,"knn")

# Evaluation
print("knn (K-Nearest Neighbors)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


#plot_svm_knn_accuracies_withRegularization
plot_svm_knn_accuracies_withRegularization(x_train,y_train ,extraction_features)

#save models by joblib in folder  modelSaveSVM-Knn
# Save models
import pickle


pickle.dump(clf, open(f'modelSaveSVM-Knn/{len(extraction_features)}svm_model{modelNum}', 'wb'))
pickle.dump(knn, open(f'modelSaveSVM-Knn/{len(extraction_features)}knn_model{modelNum}',"wb"))

loadsvm =pickle.load( open(f'modelSaveSVM-Knn/{len(extraction_features)}svm_model{modelNum}', 'rb'))
loadknn = pickle.load( open(f'modelSaveSVM-Knn/{len(extraction_features)}knn_model{modelNum}',"rb"))
print ("SVM load (Support Vector Machines )")
print("Accuracy:", accuracy_score(y_test,loadsvm.predict(x_test)))
print("knn load (K-Nearest Neighbors)")
print("Accuracy:", accuracy_score(y_test, loadknn.predict(x_test)))
