# training Configuration parameters (for a single 3D link)
INPUT_SIZE = 2 + 3  # (theta, phi) + 3D point
HIDDEN_SIZE = 16   # 16,32,64, ...
OUTPUT_SIZE = 1    # Distance Value
NUM_LAYERS = 4

NUM_EPOCHS = 50         #usually ~30 epochs sufficient
LEARNING_RATE = 0.003   #0.002
BATCH_SIZE = 256        