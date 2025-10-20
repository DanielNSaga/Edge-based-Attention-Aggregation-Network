
import os, json, torch
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    """
    Konfigurasjonsklasse for hele treningsoppsettet.

    Denne klassen samler alle innstillinger for data, modell, trening og logging.
    Den lagrer også konfigurasjonen til en JSON-fil i "runs/"-mappen.
    """

    # -------- DATA ----------
    train_path : str = "Dataset/train.h5"     # Sti til treningsdata
    val_path   : str = "Dataset/val.h5"       # Sti til valideringsdata
    test_path  : str = "Dataset/test.h5"      # Sti til testdata
    data_format: str = "channel_last"      # Format på inputdata: "channel_last" eller "channel_first"
    pad_len    : int = 128                 # Maks antall partikler per event (padding)
    stream     : bool = False              # True = stream fra disk, False = last hele dataset i RAM

    # -------- MODELL --------
    input_dims   : int = 15                # Antall inputfeatures per partikkel
    num_classes  : int = 10                # Antall utklasser (f.eks. QCD, Hbb, Hcc, ...)

    # -------- TRENING -------
    batch_size   : int = 512               # Batch-størrelse
    num_workers  : int = 8                 # Antall prosesser i DataLoader
    epochs       : int = 20                # Antall epoker
    lr           : float = 3e-3            # Start learning rate
    weight_decay : float = 1e-4            # Weight decay (L2 regularisering)
    label_smooth : float = 0.05            # Label smoothing for cross entropy

    # -------- LÆRINGSRATE / EARLY STOP ----------
    min_lr       : float = 1e-5            # Minste learning rate ved slutten av CosineScheduler
    patience     : int   = 5               # Antall epoker uten forbedring før tidlig stopp

    # -------- HARDWARE / LOGGING ----------
    device  : str = "cuda" if torch.cuda.is_available() else "cpu"  # Bruk GPU hvis tilgjengelig
    run_name: str = None                   # Navn på treningskjøring (genereres hvis None)

    def __post_init__(self):
        """
        Kjøres automatisk etter initialisering.
        Setter navn og oppretter logging-mappe.
        Lagrer også hele configen til config.json.
        """
        if self.run_name is None:
            self.run_name = f"EAAN_{datetime.now():%Y-%m-%d_%H-%M-%S}"

        run_path = os.path.join("runs", self.run_name)
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)
