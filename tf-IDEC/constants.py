"""Provide constants used throughout DEC/IDEC."""
import os

# Stopping tolerance
tol = 1e-3
# Predefined number of clusters
k = 10
# Batch size
Nm = 256
# Gamma (weight for clustering loss in IDEC)
gamma = 0.1
# Number of image channels
NUM_CHANNELS = 1
IMAGE_SHAPE = (28, 28)
# Save time for pretraining
PRETRAIN_SAVE_EVERY_N_EPOCHS = 50
PRETRAIN_ITERATIONS = 5000
# Maximum number of DEC/IDEC epochs
MAX_TRAIN_ITERATIONS = 2e5
UPDATE_P_EVERY_N_EPOCHS = 280
TRAIN_SAVE_EVERY_N_EPOCHS = UPDATE_P_EVERY_N_EPOCHS * 2
AE_LOGDIR = os.path.join(os.path.dirname(__file__), "autoencoder_logdir")
IDEC_LOGDIR = os.path.join(os.path.dirname(__file__), "idec_logdir")
