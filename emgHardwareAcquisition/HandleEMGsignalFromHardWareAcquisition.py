import numpy as np
import os
import pandas as pd


def find_csv_fileFor_emg_hardware(participants_number):
    '''
    Retrieves all CSV files from the specified participant's folder.

    Parameters:
    participants_number (int): The number of the participant whose data is being retrieved.

    Returns:
    tuple: A list of CSV file names and the path to the participant's folder.

    Example:
    >>> csv_files, emg_path = find_csv_fileFor_emg_hardware(1)
    >>> print(csv_files)
    ['file1.CSV', 'file2.CSV', ...]
    >>> print(emg_path)
    '/current_working_directory/EMG-Data/EMG_S1'
    '''
    csv_files = []

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    emg_path = os.path.join(current_file_dir, f'EMG-Data/EMG_S{participants_number}')
    if os.path.exists(emg_path):
        for file_name in os.listdir(emg_path):
            if file_name.endswith(".CSV"):
                csv_files.append(file_name)
    return csv_files, emg_path


def generate_gesture_matrix_from_csv_files(csv_files, emg_path, gesture_list):
    '''
    Generates gesture matrices from CSV files.

    Each set of 20 files represents one gesture:
    - First 20 files => rest
    - Second 20 files => close hand
    - Third 20 files => thumb
    - Fourth 20 files => close thumb little

    Parameters:
    csv_files (list): List of CSV file names.
    emg_path (str): Path to the folder containing CSV files.
    gesture_list (list of int): List representing the gestures to include (e.g., [1, 2, 3, 4]).

    Returns:
    tuple: A tuple containing training and prediction matrices with shape (number_of_gestures, 2, 40000).
           This means 10 files for each training and prediction vector.

    Example:
    >>> training_set, prediction_set = generate_gesture_matrix_from_csv_files(csv_files, emg_path, [1, 2])
    >>> print(training_set.shape)
    (2, 40000)
    >>> print(prediction_set.shape)
    (2, 40000)
    '''
    gesture_matrix = []

    for gesture_index in gesture_list:
        start_index = (gesture_index - 1) * 20
        end_index = start_index + 20

        gesture_files = csv_files[start_index:end_index]
        gesture_data = []

        for csv_file in gesture_files:
            file_path = os.path.join(emg_path, csv_file)
            df = pd.read_csv(file_path)

            emg_data = df.iloc[16:len(df), 0].values  # Adjust this if the actual data structure differs
            gesture_data.append(emg_data.flatten())

        gesture_data = np.array(gesture_data).reshape(2, -1,1)
        gesture_matrix.append(gesture_data)

    gesture_matrix = np.array(gesture_matrix, dtype=np.int32)
    training_set = gesture_matrix[:, 0, :]
    prediction_set = gesture_matrix[:, 1, :]

    return training_set, prediction_set


def Extract_emgRowData(participant, gesture_list):
    '''
    Extracts EMG raw data for each participant gesture from 20 CSV files for each gesture (hand movement).
    Each CSV file starts with a 16-row header, followed by 4000 sample values.

    Parameters:
    participant (int): The number of the participant.
    gesture_list (list of int): List representing the gestures to include (e.g., [1, 2, 3, 4]).

    Returns:
    tuple: Training and prediction matrices, and the sample frequency (4000 Hz).

    Example:
    >>> training_data, prediction_data, sf = Extract_emgRowData(participant=1, gesture_list=[1, 2])
    >>> print(training_data.shape)
    (2, 40000)
    >>> print(prediction_data.shape)
    (2, 40000)
    >>> print(sf)
    4000
    '''
    sf = 2500
    channelName=['F1']
    csv_files, emg_path = find_csv_fileFor_emg_hardware(participant)
    training_rowEMG_0_1, prediction_rowEMG_0_1 = generate_gesture_matrix_from_csv_files(csv_files, emg_path,
                                                                                        gesture_list)

    if csv_files:
        print(f"Found {len(csv_files)} folders")
        print(csv_files)
        df = pd.read_csv(os.path.join(emg_path, csv_files[0]))
        print(df.head(n=16))
        print(len(df))
    else:
        print(f"No CSV files found, participant {participant} not found")

    return training_rowEMG_0_1, prediction_rowEMG_0_1, sf,channelName


'''# Example usage:
training_rowEMG_0_1, prediction_rowEMG_0_1, sf = Extract_emgRowData(participant=1, gesture_list=[1, 2])
print(type(training_rowEMG_0_1))
print(pd.DataFrame(training_rowEMG_0_1))'''
