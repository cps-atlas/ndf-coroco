# training Configuration parameters
NUM_LINKS = 4
INPUT_SIZE = NUM_LINKS * 2  + 3  # (theta, phi) * 2 + 3D point
HIDDEN_SIZE = 512
OUTPUT_SIZE = 1    # NUM_LINKS
NUM_LAYERS = 5

NUM_EPOCHS = 8
LEARNING_RATE = 0.0013
BATCH_SIZE = 256