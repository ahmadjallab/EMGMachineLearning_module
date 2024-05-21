import numpy as np

from tensorflow import keras
import pandas as pd
import os

from EMGPatternRecognition.feature_extraction import features_estimation


def pridectionModel(data_rowEMG, extraction_features, channelName='f1', modelNumber=0, fs=2048, frame=500, step=500):
    '''
    this function predict from model list
    collect list for all model file h5 this form para model number choose from it
    :para data_rowEMG =numpy array 2d gesture row and cal vector for one  channel for each gesture
    :para extraction_feature: list for feature for model predict
    @para : modelNumber : int value for model files list

    :return = list for prediction value for each gesture **shape (gesture , each window predict )
    '''
    ##just for collect files to save dev***
    modelFilesList = []
    for filename in os.listdir('./'):
        if filename.endswith(".h5"):
            namefile = filename.title()
            modelFilesList.append(filename)
    print(pd.DataFrame({"models ML ": modelFilesList}))

    ##feature extraction
    gestures_features_list = []  ##each row represent label for gesture

    for Gesture in data_rowEMG:
        total_feature_matrixpd_prediction, _, _, total_feature_matrix_np_prediction = features_estimation(
            signal=Gesture, channel_name=channelName, fs=fs, frame=frame, step=step,
            plot=False)
        gestures_features_list.append(total_feature_matrixpd_prediction.loc[extraction_features].T.to_numpy().tolist())


    #load model
    model_prediction = keras.models.load_model(modelFilesList[modelNumber])
    print(model_prediction)

    #prediction section
    predicton_list = []
    for indexLabel , gesture_ in enumerate( gestures_features_list):
        prediction_value = model_prediction.predict(np.array(gesture_))
        prediction_value= np.round(np.max(prediction_value,axis=1))
        predicton_list.append(prediction_value.tolist())  # 0

    return predicton_list


dummyData = np.random.rand(2,4000)*100
gesture_name=["openHand","closeHand"]
extraction_features=['VAR', 'RMS']
prediction_list = pridectionModel(dummyData,extraction_features=extraction_features)

print (pd.DataFrame(prediction_list,index=gesture_name))
