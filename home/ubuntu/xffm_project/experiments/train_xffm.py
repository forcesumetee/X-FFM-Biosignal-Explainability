"""
Training Script for X-FFM
Author: Sumetee Jirapattarasakul

This script trains the X-FFM model on biosignal data.
"""

import sys
sys.path.insert(0, '/home/ubuntu/xffm_project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from models import create_multimodal_encoder, create_concept_bottleneck_model
from data.data_loader import create_dataloaders, create_synthetic_dataset


class XFFMTrainer:
    """Trainer for X-FFM model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        lambda_concept: float = 0.5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_concept = lambda_concept
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.concept_loss_fn = nn.BCELoss()
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for signals, labels, concepts in pbar:
            # Move to device
            signals = {k: v.to(self.device) for k, v in signals.items()}
            labels = labels.to(self.device)
            concepts = concepts.to(self.device)
            
            # Forward pass
            logits, pred_concepts, _ = self.model(
                signals,
                return_concepts=True
            )
            
            # Calculate losses
            classification_loss = self.classification_loss_fn(logits, labels)
            concept_loss = self.concept_loss_fn(pred_concepts, concepts)
            
            # Total loss
            loss = classification_loss + self.lambda_concept * concept_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels, concepts in self.val_loader:
                # Move to device
                signals = {k: v.to(self.device) for k, v in signals.items()}
                labels = labels.to(self.device)
                concepts = concepts.to(self.device)
                
                # Forward pass
                logits, pred_concepts, _ = self.model(
                    signals,
                    return_concepts=True
                )
                
                # Calculate losses
                classification_loss = self.classification_loss_fn(logits, labels)
                concept_loss = self.concept_loss_fn(pred_concepts, concepts)
                loss = classification_loss + self.lambda_concept * concept_loss
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, save_dir: str = './checkpoints'):
        """Train the model for multiple epochs"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0
        
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "=" * 80)
        print(f"Training Completed! Best Val Acc: {best_val_acc:.2f}%")
        print("=" * 80)
        
        return self.history


def main():
    print("=" * 80)
    print("X-FFM Training Script")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        'signal_length': 1000,
        'num_classes': 2,
        'concept_names': [
            'Regular_Rhythm',
            'Normal_Heart_Rate',
            'Low_Variability',
            'Stable_Amplitude',
            'No_Artifacts'
        ],
        'modalities': ['ecg', 'ppg'],
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'lambda_concept': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        if key != 'concept_names':
            print(f"  {key}: {value}")
    
    # Step 1: Create or load dataset
    print(f"\n[Step 1/4] Preparing dataset...")
    
    data_dir = '/home/ubuntu/xffm_project/data/synthetic'
    if not os.path.exists(data_dir):
        print("  Creating synthetic dataset...")
        create_synthetic_dataset(
            num_samples=1000,
            signal_length=CONFIG['signal_length'],
            save_dir=data_dir
        )
    else:
        print(f"  Using existing dataset at {data_dir}")
    
    # Step 2: Create dataloaders
    print(f"\n[Step 2/4] Creating dataloaders...")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        modalities=CONFIG['modalities'],
        batch_size=CONFIG['batch_size'],
        signal_length=CONFIG['signal_length']
    )
    
    # Step 3: Create model
    print(f"\n[Step 3/4] Creating X-FFM model...")
    
    modality_configs = {
        modality: {
            'signal_length': CONFIG['signal_length'],
            'in_channels': 1,
            'hidden_dim': 64,
            'num_layers': 3
        }
        for modality in CONFIG['modalities']
    }
    
    encoder = create_multimodal_encoder(modality_configs, fusion_dim=256)
    model = create_concept_bottleneck_model(
        encoder=encoder,
        concept_names=CONFIG['concept_names'],
        num_classes=CONFIG['num_classes']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {num_params:,} parameters")
    
    # Step 4: Train model
    print(f"\n[Step 4/4] Training model...")
    
    trainer = XFFMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        lambda_concept=CONFIG['lambda_concept']
    )
    
    history = trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_dir='/home/ubuntu/xffm_project/checkpoints'
    )
    
    # Save training history
    np.save('/home/ubuntu/xffm_project/checkpoints/training_history.npy', history)
    print(f"\n✓ Training history saved to checkpoints/training_history.npy")


if __name__ == "__main__":
    from typing import Tuple
    main()
