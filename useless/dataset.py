# data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class BPWaveformDataset(Dataset):
    """
    Dataset for BP waveform estimation from PPG.
    Expects data in the form of segments of PPG, corresponding BP waveforms, and optional morphological features.
    """
    def __init__(self, data_file, use_morph=True):
        """
        data_file: path to a file (e.g., NPZ or CSV) containing 'PPG', 'BP', and optionally 'features'.
                   Alternatively, data_file can be a dictionary with keys 'PPG', 'BP', 'features'.
        use_morph: whether to expect morphological features in data (True for training teacher, False for student-only usage).
        """
        # Load data
        if isinstance(data_file, str):
            # Assuming NPZ file with arrays
            data = np.load(data_file)
        else:
            # If data_file is already a dict or similar
            data = data_file
        self.ppg_segments = data['PPG']    # shape (N, seq_len)
        self.bp_segments = data['BP']      # shape (N, seq_len)
        # Normalize signals if not already normalized
        # (Assume data is already normalized as per paper; if not, we could do e.g. min-max normalization here.)
        # Denoising would also be done prior if needed.
        if use_morph:
            # Morphological features present
            if 'features' in data:
                self.morph_features = data['features']  # shape (N, feat_dim)
            else:
                raise ValueError("Morphological features not found in data but use_morph=True.")
        else:
            self.morph_features = None
        self.use_morph = use_morph
    def __len__(self):
        return len(self.ppg_segments)
    def __getitem__(self, idx):
        # Retrieve segment data
        ppg = self.ppg_segments[idx]      # (seq_len,)
        bp = self.bp_segments[idx]        # (seq_len,)
        # Convert to tensors
        ppg = torch.tensor(ppg, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        bp = torch.tensor(bp, dtype=torch.float32)  # (seq_len,)
        if self.use_morph:
            feat = self.morph_features[idx]
            feat = torch.tensor(feat, dtype=torch.float32)  # (feat_dim,)
            return ppg, feat, bp
        else:
            return ppg, bp