
import os
import pickle

# Concatenate the dataset files
dataset = []
for i in range(3):
    file_name = f'dataset_3d_large_{i}.pickle'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            dataset.extend(pickle.load(f))

        #os.remove(file_name)  # Remove the partial dataset file

# Save the concatenated dataset to a file
with open('dataset_3d_large_new1.pickle', 'wb') as f:
    pickle.dump(dataset, f)