import os, json, torch
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    """
    Configuration class for the entire training setup.

    This class collects all settings for data, model, training, and logging.
    It also saves the configuration to a JSON file inside the "runs/" folder.
    """

    # -------- DATA ----------
    train_path : str = "Dataset/train.h5"   # Path to training data
    val_path   : str = "Dataset/val.h5"     # Path to validation data
    test_path  : str = "Dataset/test.h5"    # Path to test data
    data_format: str = "channel_last"       # Input format: "channel_last" or "channel_first"
    pad_len    : int = 128                  # Max number of particles per event (padding)
    stream     : bool = False               # True = stream from disk, False = load entire dataset into RAM

    # -------- MODEL --------
    input_dims   : int = 15                 # Number of input features per particle
    num_classes  : int = 10                 # Number of output classes (e.g., QCD, Hbb, Hcc, ...)

    # -------- TRAINING -------
    batch_size   : int = 512                # Batch size
    num_workers  : int = 8                  # Number of DataLoader worker processes
    epochs       : int = 20                 # Number of epochs
    lr           : float = 3e-3             # Initial learning rate
    weight_decay : float = 1e-4             # Weight decay (L2 regularization)
    label_smooth : float = 0.05             # Label smoothing for cross-entropy

    # -------- LEARNING RATE / EARLY STOP ----------
    min_lr       : float = 1e-5             # Minimum learning rate at the end of CosineScheduler
    patience     : int   = 5                # Epochs without improvement before early stopping

    # -------- HARDWARE / LOGGING ----------
    device  : str = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    run_name: str = None                    # Name of the training run (generated if None)

    def __post_init__(self):
        """
        Runs automatically after initialization.
        Sets a run name, creates the logging folder,
        and saves the entire configuration to config.json.
        """
        if self.run_name is None:
            self.run_name = f"EAAN_{datetime.now():%Y-%m-%d_%H-%M-%S}"

        run_path = os.path.join("runs", self.run_name)
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)
