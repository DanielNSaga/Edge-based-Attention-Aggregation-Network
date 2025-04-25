import logging
import numpy as np
import h5py
from torch.utils.data import Dataset

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def pad_array(a, maxlen, value=0., dtype='float32'):
    """
    Padder hver array i en liste til en fast lengde.

    Parametre:
        a (list av arrays): Liste med arrays, én per event (f.eks. per partikkelliste).
        maxlen (int): Maks antall elementer per event.
        value (float): Fyllverdi ved padding.
        dtype (str): Datatype på output-arrayen.

    Returnerer:
        np.ndarray: Array med form (antall_events, maxlen)
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
    Torch Dataset-klasse for HDF5-filer produsert for jet-tagging.
    Kan laste hele datasettet til minne eller bruke streaming fra disk.
    """

    def __init__(self, filepath, feature_dict=None, label='label',
                 pad_len=128, data_format='channel_last', stream=False):
        """
        Initialiserer HDF5-datasettet.

        Parametre:
            filepath (str): Sti til HDF5-filen.
            feature_dict (dict): Definisjon av hvilke features som tilhører "points" og "features".
            label (str): Navn på label-feltet.
            pad_len (int): Maks antall partikler per event.
            data_format (str): 'channel_first' eller 'channel_last'.
            stream (bool): Hvis True, last kun én event om gangen fra disk.
        """
        self.filepath = filepath
        self.label = label
        self._stream = stream
        self.stack_axis = 1 if data_format == 'channel_first' else -1

        # Hvis ingen feature_dict er spesifisert, brukes standardoppsett
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
            logging.info(f"Laster inn {filepath} i minnet …")
            with h5py.File(filepath, "r") as f:
                # Hent labels
                self._label = f[label][:]
                self._values = {}
                # Hent features og stack i riktig rekkefølge og format
                for group, cols in feature_dict.items():
                    arrs = []
                    for col in cols:
                        x = f[col][:]
                        # Sørg for at alle arrays har 3 dimensjoner
                        if x.ndim == 2:
                            if self.stack_axis == -1:  # channel_last
                                x = x[..., None]
                            else:  # channel_first
                                x = x[:, None, :]
                        arrs.append(x)
                    self._values[group] = np.concatenate(arrs, axis=self.stack_axis)
            self._length = len(self._label)
            logging.info("… ferdig.")
        else:
            # Streaming: bare les lengden
            with h5py.File(filepath, "r") as f:
                self._length = f[label].shape[0]
            self._label, self._values = None, None

    def __len__(self):
        """Returnerer antall eventer."""
        return self._length

    def __getitem__(self, idx):
        """
        Returnerer én event i format:
            {"X": {"points": …, "features": …}, "y": label}
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
    Initialiserer H5Dataset-instansene for trenings-, validerings- og testdata.

    Parametre:
        train_path (str): Sti til treningsfil.
        val_path (str): Sti til valideringsfil.
        test_path (str): Sti til testfil.
        **kwargs: Ekstra argumenter sendes videre til H5Dataset.

    Returnerer:
        Tuple[Dataset, Dataset, Dataset]: train, val, test
    """
    train = H5Dataset(train_path, **kwargs)
    val = H5Dataset(val_path, **kwargs)
    test = H5Dataset(test_path, **kwargs)
    return train, val, test
