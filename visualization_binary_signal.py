import numpy as np
import json
import os
import matplotlib.pyplot as plt

'''
extract Row data from binary file extension dat  and save it as json file
metadata of this row data is in header file extension hea
frequency of this data is 2048 Hz  
'''

# Specify the path to your .dat file and .hea file
dat_file_path = 'session1_participant1_gesture10_trial1.dat'
hea_file_path = 'session1_participant1_gesture10_trial1.hea'

# Load metadata from the .hea file
with open(hea_file_path, 'r') as hea_file:
    metadata_lines = hea_file.readlines()

# Extract relevant information from the metadata
num_channels = int(metadata_lines[0].split()[1])
sampling_rate = int(metadata_lines[0].split()[2])
num_samples = int(metadata_lines[0].split()[3])

#Extract scale factors for each channel from the metadata
scale_factors = {}
#details_of_signal_extract 
dict_header_of_signal_extract = {}
for line in metadata_lines[1:]:
   # print(line)
    tokens = line.split()
    
   # print(tokens)
    channel_name = tokens[-1]  # Assuming the last token is the channel name (e.g., F1, F2, etc.)
    dict_header_of_signal_extract[channel_name] = tokens
    scale_factor = float(tokens[2].split('(')[1].split(')')[0])  # Extract the scale factor from the format (number)
    scale_factors[channel_name] = scale_factor


# Print the scale factors for each channel
for channel_name, scale_factor in scale_factors.items():
    pass
   # print(f"Channel {channel_name}: Scale Factor = {scale_factor}")
sum=0
for channel_name, tokens in dict_header_of_signal_extract.items():
    print(f"Channel {channel_name}: Scale Factor = {tokens[5]}")
    sum=sum+np.abs(int(tokens[5]))
    

# Print the sum of all scale factors for channels 
print(f"Sum of all scale factors: {sum}")







# Load binary data from the .dat file
data = np.fromfile(dat_file_path, dtype=np.int16)  # Adjust the data type based on your actual data format
print(data.size)

# Convert the data array to a list
data_list = data.tolist()
print(type(data_list) )
print (data_list[10220:10250])
# Save the data as JSON

json_file_path = 'data.json'

# Check if data.json already exists
if not os.path.exists(json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file)



# Reshape the data based on the number of channels and samples
# reshaped_data = data.reshape((num_samples, num_channels))

# Generate time values for the x-axis
time = np.arange(327664)/2048 

# Plot the reshaped data with time on the x-axis
plt.plot(time, data)

# Add labels and title to the plot
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Binary Signal Visualization')

# Display the plot
plt.show()

# Reshape the data based on the number of channels and samples
# reshaped_data = data.reshape((num_samples, num_channels))

# Print the reshaped data
# print(reshaped_data)
