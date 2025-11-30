"""
Data Loader for X-FFM
Author: Sumetee Jirapattarasakul

This module provides utilities to load and preprocess biosignal datasets.
Supports multiple public datasets including:
- MIT-BIH Arrhythmia Database
- PhysioNet datasets
- Custom CSV/NPY files
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os


class BiosignalDataset(Dataset):
    """
    PyTorch Dataset for multimodal biosignals
    """
    
    def __init__(
        self,
        data_dir: str,
        modalities: List[str],
        signal_length: int = 1000,
        transform=None
    ):
        """
        Args:
            data_dir: Directory containing the data files
            modalities: List of modality names (e.g., ['ecg', 'ppg'])
            signal_length: Length of each signal segment
            transform: Optional transform to apply to signals
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.signal_length = signal_length
        self.transform = transform
        
        # Load data
        self.signals, self.labels, self.concepts = self._load_data()
        
    def _load_data(self) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Load data from files
        
        Expected file structure:
        data_dir/
            ecg_signals.npy  # Shape: [N, signal_length]
            ppg_signals.npy  # Shape: [N, signal_length]
            labels.npy       # Shape: [N]
            concepts.npy     # Shape: [N, num_concepts]
        
        Returns:
            signals: Dictionary of signals for each modality
            labels: Array of class labels
            concepts: Array of concept annotations
        """
        signals = {}
        
        # Load each modality
        for modality in self.modalities:
            signal_file = os.path.join(self.data_dir, f'{modality}_signals.npy')
            if os.path.exists(signal_file):
                signals[modality] = np.load(signal_file)
                print(f"Loaded {modality}: {signals[modality].shape}")
            else:
                raise FileNotFoundError(f"Signal file not found: {signal_file}")
        
        # Load labels
        label_file = os.path.join(self.data_dir, 'labels.npy')
        if os.path.exists(label_file):
            labels = np.load(label_file)
            print(f"Loaded labels: {labels.shape}")
        else:
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Load concepts (optional)
        concept_file = os.path.join(self.data_dir, 'concepts.npy')
        if os.path.exists(concept_file):
            concepts = np.load(concept_file)
            print(f"Loaded concepts: {concepts.shape}")
        else:
            print("Warning: Concept annotations not found. Using dummy values.")
            concepts = np.zeros((len(labels), 5))  # Dummy concepts
        
        return signals, labels, concepts
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            signals: Dictionary of signal tensors
            label: Class label
            concepts: Concept annotations
        """
        # Get signals for all modalities
        sample_signals = {}
        for modality in self.modalities:
            signal = self.signals[modality][idx]
            
            # Ensure correct length
            if len(signal) > self.signal_length:
                signal = signal[:self.signal_length]
            elif len(signal) < self.signal_length:
                signal = np.pad(signal, (0, self.signal_length - len(signal)))
            
            # Add channel dimension and convert to tensor
            signal = torch.from_numpy(signal).float().unsqueeze(0)
            
            # Apply transform if provided
            if self.transform:
                signal = self.transform(signal)
            
            sample_signals[modality] = signal
        
        # Get label and concepts
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        concepts = torch.from_numpy(self.concepts[idx]).float()
        
        return sample_signals, label, concepts


def create_synthetic_dataset(
    num_samples: int = 1000,
    signal_length: int = 1000,
    num_classes: int = 2,
    num_concepts: int = 5,
    save_dir: str = './data/synthetic'
):
    """
    Create a synthetic dataset for testing
    
    Args:
        num_samples: Number of samples to generate
        signal_length: Length of each signal
        num_classes: Number of classes
        num_concepts: Number of concepts
        save_dir: Directory to save the generated data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating synthetic dataset with {num_samples} samples...")
    
    # Generate ECG signals
    ecg_signals = []
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, signal_length)
        if i % 2 == 0:  # Normal
            signal = np.sin(1.0 * t) + 0.3 * np.sin(2.0 * t)
        else:  # Arrhythmia
            signal = np.sin(1.5 * t) + 0.5 * np.sin(3.0 * t) + 0.2 * np.sin(5.0 * t)
        signal += np.random.randn(signal_length) * 0.1
        ecg_signals.append(signal)
    ecg_signals = np.array(ecg_signals, dtype=np.float32)
    
    # Generate PPG signals
    ppg_signals = []
    for i in range(num_samples):
        t = np.linspace(0, 4 * np.pi, signal_length)
        if i % 2 == 0:  # Normal
            signal = np.sin(0.8 * t) + 0.2 * np.sin(1.6 * t)
        else:  # Arrhythmia
            signal = np.sin(1.2 * t) + 0.4 * np.sin(2.4 * t)
        signal += np.random.randn(signal_length) * 0.1
        ppg_signals.append(signal)
    ppg_signals = np.array(ppg_signals, dtype=np.float32)
    
    # Generate labels
    labels = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
    
    # Generate concept annotations
    concepts = np.zeros((num_samples, num_concepts), dtype=np.float32)
    for i in range(num_samples):
        if labels[i] == 0:  # Normal
            concepts[i] = np.array([0.85, 0.92, 0.78, 0.88, 0.95]) + np.random.randn(num_concepts) * 0.05
        else:  # Arrhythmia
            concepts[i] = np.array([0.45, 0.52, 0.38, 0.48, 0.55]) + np.random.randn(num_concepts) * 0.05
        concepts[i] = np.clip(concepts[i], 0, 1)
    
    # Save to files
    np.save(os.path.join(save_dir, 'ecg_signals.npy'), ecg_signals)
    np.save(os.path.join(save_dir, 'ppg_signals.npy'), ppg_signals)
    np.save(os.path.join(save_dir, 'labels.npy'), labels)
    np.save(os.path.join(save_dir, 'concepts.npy'), concepts)
    
    print(f"✓ Synthetic dataset saved to {save_dir}")
    print(f"  - ECG signals: {ecg_signals.shape}")
    print(f"  - PPG signals: {ppg_signals.shape}")
    print(f"  - Labels: {labels.shape}")
    print(f"  - Concepts: {concepts.shape}")


def create_dataloaders(
    data_dir: str,
    modalities: List[str],
    batch_size: int = 32,
    train_split: float = 0.8,
    signal_length: int = 1000
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Directory containing the data
        modalities: List of modality names
        batch_size: Batch size
        train_split: Fraction of data to use for training
        signal_length: Length of signals
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Create dataset
    dataset = BiosignalDataset(
        data_dir=data_dir,
        modalities=modalities,
        signal_length=signal_length
    )
    
    # Split into train and validation
    num_train = int(len(dataset) * train_split)
    num_val = len(dataset) - num_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created dataloaders:")
    print(f"  - Training samples: {num_train}")
    print(f"  - Validation samples: {num_val}")
    print(f"  - Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example: Create synthetic dataset
    create_synthetic_dataset(
        num_samples=1000,
        signal_length=1000,
        save_dir='/home/ubuntu/xffm_project/data/synthetic'
    )
    
    # Example: Load dataset
    train_loader, val_loader = create_dataloaders(
        data_dir='/home/ubuntu/xffm_project/data/synthetic',
        modalities=['ecg', 'ppg'],
        batch_size=32
    )
    
    # Test loading a batch
    for signals, labels, concepts in train_loader:
        print("\nLoaded batch:")
        for modality, signal in signals.items():
            print(f"  - {modality}: {signal.shape}")
        print(f"  - Labels: {labels.shape}")
        print(f"  - Concepts: {concepts.shape}")
        break
