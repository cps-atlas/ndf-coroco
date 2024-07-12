# training Configuration parameters (for a single 3D link)
INPUT_SIZE = 2 + 3  # (theta, phi) + 3D point
HIDDEN_SIZE = 64    #512
OUTPUT_SIZE = 1    # Distance Value
NUM_LAYERS = 3

NUM_EPOCHS = 100   
LEARNING_RATE = 0.002   #0.0015
BATCH_SIZE = 128   #256