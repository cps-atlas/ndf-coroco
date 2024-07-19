# training Configuration parameters (for a single 3D link)
INPUT_SIZE = 2 + 3  # (theta, phi) + 3D point
HIDDEN_SIZE = 32   #64
OUTPUT_SIZE = 1    # Distance Value
NUM_LAYERS = 3

NUM_EPOCHS = 30         #50  
LEARNING_RATE = 0.003   #0.0015
BATCH_SIZE = 256        #256