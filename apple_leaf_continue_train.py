# apple_leaf_continue_train.py (Fixed Version)
"""
Continue training apple leaf disease model from saved checkpoint
"""

# ============================================================================
# IMPORTS (same as before)
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import json
import time
from datetime import datetime
import csv
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFIED FOR CONTINUED TRAINING
# ============================================================================
class Config:
    """Configuration class for training parameters"""
    
    # Dataset path
    DATA_PATH = r"C:\Users\nimit\Music\.vscode\machine_leaning\AppleLeaf9-main"
    
    # Dataset parameters
    CLASS_NAMES = [
        'Alternaria leaf spot',
        'Brown spot',
        'Frogeye leaf spot',
        'Grey spot',
        'Health',
        'Mosaic',
        'Powdery mildew',
        'Rust',
        'Scab'
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Training parameters - CHANGED FOR CONTINUED TRAINING
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005  # Reduced for continued training
    EPOCHS = 40  # Total epochs (19 already + 21 new)
    
    # Model selection
    MODEL_NAME = 'resnet18'
    
    # Data splitting
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Output directories
    OUTPUT_DIR = "training_output_continued"
    MODELS_DIR = "saved_models_continued"
    RESULTS_DIR = "results_continued"
    TREATMENT_DIR = "treatment_database"
    
    # Checkpoint path - ADD THIS
    CHECKPOINT_PATH = r"C:\Users\nimit\Music\.vscode\machine_leaning\saved_models\best_model_with_treatment_20251201_174933.pth"

# Initialize config
config = Config()

# Create directories
for dir_name in [config.OUTPUT_DIR, config.MODELS_DIR, 
                 config.RESULTS_DIR, config.TREATMENT_DIR]:
    os.makedirs(dir_name, exist_ok=True)

# ============================================================================
# CHECKPOINT LOADING FUNCTION
# ============================================================================
def load_checkpoint_for_continued_training(checkpoint_path, device):
    """Load checkpoint for continued training"""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get actual classes from checkpoint
    actual_classes = checkpoint.get('actual_classes', config.CLASS_NAMES)
    print(f"Found {len(actual_classes)} classes in checkpoint: {actual_classes}")
    
    # Create model with correct number of classes
    num_classes = len(actual_classes)
    
    # Create model (same architecture as before)
    if config.MODEL_NAME == 'resnet18':
        model = models.resnet18(pretrained=False)  # Don't load pretrained weights
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Add other architectures if needed
        pass
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create optimizer and load its state
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Check if optimizer state exists in checkpoint
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Loaded optimizer state from checkpoint")
    else:
        print("⚠ No optimizer state found in checkpoint, using fresh optimizer")
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_val_acc = checkpoint.get('val_acc', 0.0)
    
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Starting epoch: {start_epoch}")
    print(f"  Previous best validation accuracy: {best_val_acc:.2f}%")
    
    return model, optimizer, start_epoch, best_val_acc, actual_classes

# ============================================================================
# DATASET CLASSES (same as before)
# ============================================================================
def find_class_folders():
    """Find actual class folder names"""
    if not os.path.exists(config.DATA_PATH):
        return []
    
    items = os.listdir(config.DATA_PATH)
    class_folders = []
    
    for item in items:
        item_path = os.path.join(config.DATA_PATH, item)
        if os.path.isdir(item_path):
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if images:
                class_folders.append(item)
    
    return class_folders

class AppleLeafDataset(Dataset):
    """Custom Dataset for Apple Leaf Disease images"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        class_folders = find_class_folders()
        
        if not class_folders:
            raise ValueError(f"No class folders with images found in {root_dir}")
        
        print(f"Found {len(class_folders)} classes: {class_folders}")
        
        # Create mapping from folder name to label index
        self.class_to_idx = {folder_name: idx for idx, folder_name in enumerate(class_folders)}
        
        for folder_name in class_folders:
            class_path = os.path.join(root_dir, folder_name)
            label_idx = self.class_to_idx[folder_name]
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(label_idx)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Total images loaded: {len(self.image_paths)}")
        
        self._split_data()
        self.class_folders = class_folders
    
    def _split_data(self):
        """Split data into train, validation, and test sets"""
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            self.image_paths, self.labels, 
            test_size=config.TEST_SIZE + config.VAL_SIZE, 
            random_state=42, 
            stratify=self.labels
        )
        
        val_ratio = config.VAL_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1-val_ratio,
            random_state=42,
            stratify=temp_labels
        )
        
        if self.mode == 'train':
            self.image_paths = train_paths
            self.labels = train_labels
        elif self.mode == 'val':
            self.image_paths = val_paths
            self.labels = val_labels
        elif self.mode == 'test':
            self.image_paths = test_paths
            self.labels = test_labels
        
        print(f"{self.mode.capitalize()} set: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='white')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_labels

# ============================================================================
# MAIN CONTINUED TRAINING FUNCTION
# ============================================================================
def main():
    """Main function for continued training"""
    
    print("=" * 70)
    print("APPLE LEAF DISEASE - CONTINUED TRAINING")
    print(f"Continuing from: {config.CHECKPOINT_PATH}")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Checkpoint not found at {config.CHECKPOINT_PATH}")
        print("Please check the path and try again.")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load checkpoint
    model, optimizer, start_epoch, best_val_acc, actual_classes = load_checkpoint_for_continued_training(
        config.CHECKPOINT_PATH, device
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    try:
        print("\n" + "-" * 50)
        print("Creating datasets...")
        
        train_dataset = AppleLeafDataset(config.DATA_PATH, train_transform, 'train')
        val_dataset = AppleLeafDataset(config.DATA_PATH, val_transform, 'val')
        test_dataset = AppleLeafDataset(config.DATA_PATH, val_transform, 'test')
        
        print(f"✓ Datasets created successfully")
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Validation: {len(val_dataset)} images")
        print(f"  Test: {len(test_dataset)} images")
        
    except Exception as e:
        print(f"\n❌ Error creating datasets: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler - FIXED: verbose parameter removed
    try:
        # Try with verbose parameter for newer PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=2,
            factor=0.5,
            verbose=True
        )
        print("✓ Learning rate scheduler created with verbose=True")
    except TypeError:
        # Fallback for older PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=2,
            factor=0.5
        )
        print("✓ Learning rate scheduler created (without verbose)")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epochs': [],
        'learning_rates': []
    }
    
    print("\n" + "=" * 50)
    print(f"STARTING CONTINUED TRAINING")
    print(f"Epochs: {start_epoch} to {config.EPOCHS} (Total: {config.EPOCHS - start_epoch} new epochs)")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()
        
        print(f"\n{'='*40}")
        print(f"EPOCH {epoch+1}/{config.EPOCHS}")
        print(f"{'='*40}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint regularly
        if (epoch + 1) % 5 == 0 or epoch == config.EPOCHS - 1:
            checkpoint_path = os.path.join(
                config.MODELS_DIR, 
                f'checkpoint_epoch_{epoch+1}_{timestamp}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'actual_classes': actual_classes,
                'learning_rate': current_lr,
                'config': vars(config)
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                config.MODELS_DIR, 
                f'best_model_continued_{timestamp}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'actual_classes': actual_classes,
                'learning_rate': current_lr,
                'total_epochs': epoch + 1,
                'previous_checkpoint': config.CHECKPOINT_PATH,
                'config': vars(config)
            }, best_model_path)
            print(f"  🏆 NEW BEST MODEL! Val Acc: {val_acc:.2f}%")
            print(f"    Saved to: {best_model_path}")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("CONTINUED TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Total epochs trained: {config.EPOCHS}")
    print(f"New epochs trained: {config.EPOCHS - start_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test evaluation
    print("\n" + "-" * 50)
    print("Evaluating on test set...")
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\n📊 TEST SET RESULTS:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    # Generate classification report
    print("\n📈 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=actual_classes))
    
    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, f'final_model_40_epochs_{timestamp}.pth')
    torch.save({
        'epoch': config.EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_acc': test_acc,
        'actual_classes': actual_classes,
        'total_epochs': config.EPOCHS,
        'history': history,
        'config': vars(config)
    }, final_model_path)
    
    print(f"✓ Final model saved: {final_model_path}")
    
    # Plot training history
    plot_training_history(history, timestamp)
    
    # Create summary report
    create_summary_report(history, test_acc, timestamp, config)
    
    print("\n" + "=" * 70)
    print("✅ CONTINUED TRAINING SUCCESSFUL!")
    print("=" * 70)
    print(f"\n📁 Outputs saved in:")
    print(f"  - Models: {config.MODELS_DIR}/")
    print(f"  - Results: {config.RESULTS_DIR}/")
    print(f"  - Training output: {config.OUTPUT_DIR}/")
    
    print(f"\n🎯 Best model: best_model_continued_{timestamp}.pth")
    print(f"📊 Test accuracy: {test_acc:.2f}%")
    print(f"⏱ Total epochs: {config.EPOCHS} (continued from epoch {start_epoch})")
    print(f"🔄 New epochs trained: {config.EPOCHS - start_epoch}")
    
    return model, history

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def plot_training_history(history, timestamp):
    """Plot and save training history"""
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['epochs'], history['train_loss'], label='Train Loss', marker='o')
        plt.plot(history['epochs'], history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['epochs'], history['train_acc'], label='Train Acc', marker='o')
        plt.plot(history['epochs'], history['val_acc'], label='Val Acc', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(config.RESULTS_DIR, f'training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved: {plot_path}")
    except Exception as e:
        print(f"⚠ Could not save plot: {e}")

def create_summary_report(history, test_acc, timestamp, config):
    """Create summary report"""
    try:
        report = {
            'training_summary': {
                'total_epochs': config.EPOCHS,
                'continued_from_checkpoint': config.CHECKPOINT_PATH,
                'best_val_accuracy': max(history['val_acc']) if history['val_acc'] else 0,
                'final_val_accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
                'final_train_accuracy': history['train_acc'][-1] if history['train_acc'] else 0,
                'test_accuracy': test_acc,
                'model_architecture': config.MODEL_NAME,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE
            },
            'history': history
        }
        
        # Save as JSON
        json_path = os.path.join(config.RESULTS_DIR, f'summary_report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Save as text
        txt_path = os.path.join(config.RESULTS_DIR, f'summary_report_{timestamp}.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("APPLE LEAF DISEASE - CONTINUED TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Continued from: {config.CHECKPOINT_PATH}\n")
            f.write(f"Total epochs: {config.EPOCHS}\n")
            f.write(f"New epochs trained: {len(history['epochs'])}\n")
            f.write(f"Best validation accuracy: {max(history['val_acc']) if history['val_acc'] else 0:.2f}%\n")
            f.write(f"Final validation accuracy: {history['val_acc'][-1] if history['val_acc'] else 0:.2f}%\n")
            f.write(f"Test accuracy: {test_acc:.2f}%\n")
            f.write(f"Model: {config.MODEL_NAME}\n")
            f.write(f"Learning rate: {config.LEARNING_RATE}\n")
            f.write(f"Batch size: {config.BATCH_SIZE}\n\n")
            
            if history['epochs']:
                f.write("Epoch-wise Performance:\n")
                f.write("-" * 60 + "\n")
                f.write("Epoch | Train Acc | Val Acc | Train Loss | Val Loss | LR\n")
                f.write("-" * 60 + "\n")
                for i, epoch in enumerate(history['epochs']):
                    f.write(f"{epoch:5d} | {history['train_acc'][i]:8.2f}% | {history['val_acc'][i]:7.2f}% | "
                           f"{history['train_loss'][i]:10.4f} | {history['val_loss'][i]:9.4f} | "
                           f"{history['learning_rates'][i]:.6f}\n")
        
        print(f"✓ Summary report saved: {txt_path}")
    except Exception as e:
        print(f"⚠ Could not save summary report: {e}")

# ============================================================================
# RUN THE CONTINUED TRAINING
# ============================================================================
if __name__ == "__main__":
    print("Starting Apple Leaf Disease Continued Training...")
    
    # Check requirements
    required_packages = ['torch', 'torchvision', 'PIL', 'sklearn', 'pandas']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        exit(1)
    
    print("✓ All required packages installed")
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_PATH):
        print(f"\n❌ Dataset path not found: {config.DATA_PATH}")
        print("Please check the path and try again.")
        exit(1)
    
    # Run main function
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()