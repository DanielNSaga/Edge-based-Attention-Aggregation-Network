import logging
import numpy as np
import h5py
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def pad_array(a, maxlen, value=0., dtype='float32'):
    """
    Pad each array in a list to a fixed length.

    Parameters:
        a (list of arrays): Sequence of arrays, one per event (for example per particle list).
        maxlen (int): Maximum number of elements per event.
        value (float): Padding value to use when extending arrays.
        dtype (str): Data type of the returned array.

    Returns:
        np.ndarray: Array with shape (num_events, maxlen)
    """
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


# ---------------------------------------------------------------
class H5Dataset(Dataset):
    """
    Torch Dataset for the jet-tagging HDF5 files.
    Can load the entire dataset into memory or stream directly from disk.
    """

    def __init__(self, filepath, feature_dict=None, label='label',
                 pad_len=128, data_format='channel_last', stream=False):
        """
        Initialize the HDF5 dataset.

        Parameters:
            filepath (str): Path to the HDF5 file.
            feature_dict (dict): Mapping that specifies which columns belong to "points" and "features".
            label (str): Name of the label field.
            pad_len (int): Maximum number of particles per event.
            data_format (str): Either 'channel_first' or 'channel_last'.
            stream (bool): If True, read one event at a time from disk.
        """
        self.filepath = filepath
        self.label = label
        self._stream = stream
        self.stack_axis = 1 if data_format == 'channel_first' else -1

        # Fall back to the default feature layout if none is provided
        if feature_dict is None:
            feature_dict = {
                "points": ["part_delta_eta", "part_delta_phi"],
                "features": [
                    "part_log_pt", "part_log_energy", "part_log_ptrel",
                    "part_log_Erel", "part_deltaR", "part_charge",
                    "part_isElectron", "part_isMuon",
                    "part_isChargedHadron", "part_isNeutralHadron",
                    "part_isPhoton", "part_tanh_d0", "part_tanh_dz",
                    "part_sigma_d0", "part_sigma_dz",
                ]
            }
        self.feature_dict = feature_dict

        if not stream:
            logging.info(f"Loading {filepath} into memory...")
            with h5py.File(filepath, "r") as f:
                # Fetch labels
                self._label = f[label][:]
                self._values = {}
                # Fetch features and stack them in the correct order and layout
                for group, cols in feature_dict.items():
                    arrs = []
                    for col in cols:
                        x = f[col][:]
                        # Ensure that every array has three dimensions
                        if x.ndim == 2:
                            if self.stack_axis == -1:  # channel_last
                                x = x[..., None]
                            else:  # channel_first
                                x = x[:, None, :]
                        arrs.append(x)
                    self._values[group] = np.concatenate(arrs, axis=self.stack_axis)
            self._length = len(self._label)
            logging.info("Done.")
        else:
            # Streaming mode: only read the length
            with h5py.File(filepath, "r") as f:
                self._length = f[label].shape[0]
            self._label, self._values = None, None

    def __len__(self):
        """Return number of events."""
        return self._length

    def __getitem__(self, idx):
        """
        Return a single event formatted as:
            {"X": {"points": ..., "features": ...}, "y": label}
        """
        if self._stream:
            with h5py.File(self.filepath, "r") as f:
                sample = {}
                for group, cols in self.feature_dict.items():
                    arrs = []
                    for col in cols:
                        x = f[col][idx]
                        if x.ndim == 1:
                            x = x[:, None] if self.stack_axis == 1 else x[None, :]
                        arrs.append(x)
                    sample[group] = np.concatenate(arrs, axis=self.stack_axis)
                label = f[self.label][idx]
        else:
            sample = {k: v[idx] for k, v in self._values.items()}
            label = self._label[idx]

        return {"X": sample, "y": label}


# ---------------------------------------------------------------
def get_datasets(train_path, val_path, test_path, **kwargs):
    """
    Create H5Dataset instances for the train, validation, and test splits.

    Parameters:
        train_path (str): Path to the training file.
        val_path (str): Path to the validation file.
        test_path (str): Path to the test file.
        **kwargs: Extra arguments forwarded to H5Dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train, val, test
    """
    train = H5Dataset(train_path, **kwargs)
    val = H5Dataset(val_path, **kwargs)
    test = H5Dataset(test_path, **kwargs)
    return train, val, test
