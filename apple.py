# apple_leaf_training_with_treatment.py
"""
Apple Leaf Disease Classification + Treatment Recommendation System
Trains model and provides cure/prevention for detected diseases
"""

# ============================================================================
# IMPORTS
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
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration class for training parameters"""
    
    # Dataset path
    DATA_PATH = r"C:\Users\nimit\Music\.vscode\machine_leaning\AppleLeaf9-main"
    
    # Dataset parameters
    CLASS_NAMES = [
        'Altemaria leaf spot',
        'Brown spot',
        'Froggee leaf spot',
        'Grey spot',
        'Health',
        'Mosaic',
        'Powdery mildew',
        'Rust',  # Changed from 'Ratt' to 'Rust'
        'Scab'
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Training parameters
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 20
    
    # Model selection
    MODEL_NAME = 'resnet18'
    
    # Data splitting
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Output directories
    OUTPUT_DIR = "training_output"
    MODELS_DIR = "saved_models"
    RESULTS_DIR = "results"
    TREATMENT_DIR = "treatment_database"

# Initialize config
config = Config()

# Create directories
for dir_name in [config.OUTPUT_DIR, config.MODELS_DIR, 
                 config.RESULTS_DIR, config.TREATMENT_DIR]:
    os.makedirs(dir_name, exist_ok=True)

# ============================================================================
# TREATMENT DATABASE CREATION
# ============================================================================
def create_treatment_database():
    """Create comprehensive treatment and prevention database"""
    
    treatment_data = {
        'Altemaria leaf spot': {
            'description': 'Fungal disease causing dark spots with concentric rings',
            'chemical_control': [
                {'product': 'Mancozeb 80WP', 'rate': '2.5g/L', 'interval': '10-14 days', 'phi': '21 days'},
                {'product': 'Chlorothalonil 720SC', 'rate': '2ml/L', 'interval': '10-14 days', 'phi': '14 days'},
                {'product': 'Azoxystrobin 23% SC', 'rate': '1ml/L', 'interval': '14 days', 'phi': '14 days'}
            ],
            'organic_control': [
                {'product': 'Copper hydroxide', 'rate': '4g/L', 'interval': '7-10 days', 'phi': '1 day'},
                {'product': 'Bacillus subtilis', 'rate': '5g/L', 'interval': '7 days', 'phi': '0 days'}
            ],
            'cultural_practices': [
                'Remove fallen leaves in autumn',
                'Prune for better air circulation',
                'Avoid overhead irrigation',
                'Maintain tree spacing (15-20 feet)'
            ],
            'spray_schedule': [
                {'stage': 'Green tip', 'product': 'Mancozeb'},
                {'stage': 'Pink bud', 'product': 'Chlorothalonil'},
                {'stage': 'Petal fall', 'product': 'Azoxystrobin'},
                {'stage': 'Summer', 'product': 'Mancozeb (every 14 days)'}
            ],
            'resistant_varieties': ['Liberty', 'Freedom', 'Enterprise'],
            'severity_threshold': {
                'low': '<5% leaf area affected',
                'medium': '5-20% leaf area affected',
                'high': '>20% leaf area affected'
            }
        },
        'Brown spot': {
            'description': 'Circular brown spots with yellow halos',
            'chemical_control': [
                {'product': 'Myclobutanil 10% WP', 'rate': '1g/L', 'interval': '14 days', 'phi': '14 days'},
                {'product': 'Tebuconazole 25% EC', 'rate': '0.5ml/L', 'interval': '14 days', 'phi': '21 days'}
            ],
            'cultural_practices': [
                'Remove infected leaves and fruit',
                'Balanced fertilization (avoid excess nitrogen)',
                'Proper canopy management'
            ]
        },
        'Froggee leaf spot': {
            'description': 'Also known as Black Rot, causes frog-eye shaped spots',
            'chemical_control': [
                {'product': 'Thiophanate-methyl 70% WP', 'rate': '1.5g/L', 'interval': '10-14 days', 'phi': '14 days'},
                {'product': 'Pyraclostrobin 20% WG', 'rate': '0.5g/L', 'interval': '14 days', 'phi': '14 days'}
            ],
            'critical_period': '4-6 weeks after petal fall',
            'management': [
                'Prune out dead wood and cankers',
                'Remove mummified fruit',
                'Avoid wounding trees'
            ]
        },
        'Grey spot': {
            'description': 'Grayish spots with purple margins',
            'chemical_control': [
                {'product': 'Dodine 65% WP', 'rate': '1g/L', 'interval': '14 days', 'phi': '21 days'},
                {'product': 'Fenbuconazole 24% SC', 'rate': '0.5ml/L', 'interval': '14 days', 'phi': '21 days'}
            ],
            'organic_control': [
                {'product': 'Sulfur 80% WG', 'rate': '4g/L', 'interval': '7 days', 'phi': '1 day'}
            ]
        },
        'Health': {
            'description': 'Healthy apple leaf',
            'preventive_measures': [
                'Regular monitoring',
                'Apply preventive fungicides',
                'Maintain tree health',
                'Proper irrigation'
            ]
        },
        'Mosaic': {
            'description': 'Viral disease causing yellow mosaic patterns',
            'management': [
                'NO CHEMICAL CURE - Remove infected trees',
                'Use virus-free planting material',
                'Control aphid vectors (Imidacloprid)',
                'Disinfect tools (10% bleach solution)'
            ],
            'resistant_varieties': ['Some M-series rootstocks']
        },
        'Powdery mildew': {
            'description': 'White powdery growth on leaves',
            'chemical_control': [
                {'product': 'Myclobutanil 10% EW', 'rate': '1ml/L', 'interval': '14 days', 'phi': '14 days'},
                {'product': 'Triflumizole 30% EC', 'rate': '0.75ml/L', 'interval': '14 days', 'phi': '14 days'}
            ],
            'organic_control': [
                {'product': 'Potassium bicarbonate', 'rate': '5g/L', 'interval': '7 days', 'phi': '0 days'},
                {'product': 'Neem oil', 'rate': '5ml/L', 'interval': '7 days', 'phi': '0 days'}
            ],
            'cultural_control': [
                'Prune for open canopy',
                'Remove water sprouts',
                'Avoid excessive nitrogen'
            ]
        },
        'Rust': {
            'description': 'Orange or yellow rust pustules on leaves',
            'chemical_control': [
                {'product': 'Myclobutanil 10% WP', 'rate': '1g/L', 'interval': '14 days', 'phi': '14 days'},
                {'product': 'Triadimefon 25% WP', 'rate': '0.5g/L', 'interval': '21 days', 'phi': '21 days'}
            ],
            'critical_action': 'Remove alternate host (Juniper/Cedar trees within 2 miles)',
            'spray_schedule': [
                {'stage': 'Pink bud', 'action': 'First spray'},
                {'stage': 'Petal fall', 'action': 'Second spray'},
                {'stage': 'Summer', 'action': 'Every 10-14 days in wet weather'}
            ]
        },
        'Scab': {
            'description': 'Most common apple disease, causes olive-green spots',
            'chemical_control': [
                {'product': 'Mancozeb 75% WP', 'rate': '2g/L', 'interval': '7-10 days', 'phi': '21 days'},
                {'product': 'Dodine 65% WP', 'rate': '1g/L', 'interval': '10-14 days', 'phi': '21 days'},
                {'product': 'Trifloxystrobin 50% WG', 'rate': '0.3g/L', 'interval': '14 days', 'phi': '14 days'}
            ],
            'ipm_strategy': [
                'Monitor: Install weather stations',
                'Threshold: >10 hours leaf wetness at 10-25°C triggers spray',
                'Resistance Management: Rotate fungicide classes'
            ],
            'resistant_varieties': ['Liberty', 'Freedom', 'Goldrush', 'Enterprise']
        }
    }
    
    # Save as JSON
    treatment_json_path = os.path.join(config.TREATMENT_DIR, 'treatment_database.json')
    with open(treatment_json_path, 'w') as f:
        json.dump(treatment_data, f, indent=4)
    
    # Save as CSV for easier viewing
    csv_rows = []
    for disease, info in treatment_data.items():
        if disease == 'Health':
            continue
            
        for chem in info.get('chemical_control', []):
            csv_rows.append({
                'disease': disease,
                'control_type': 'Chemical',
                'product': chem['product'],
                'rate': chem.get('rate', ''),
                'interval': chem.get('interval', ''),
                'phi': chem.get('phi', ''),
                'description': info['description']
            })
        
        for org in info.get('organic_control', []):
            csv_rows.append({
                'disease': disease,
                'control_type': 'Organic',
                'product': org['product'],
                'rate': org.get('rate', ''),
                'interval': org.get('interval', ''),
                'phi': org.get('phi', ''),
                'description': info['description']
            })
    
    if csv_rows:
        csv_path = os.path.join(config.TREATMENT_DIR, 'treatment_recommendations.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['disease', 'control_type', 'product', 'rate', 'interval', 'phi', 'description'])
            writer.writeheader()
            writer.writerows(csv_rows)
    
    print(f"✓ Treatment database created in {config.TREATMENT_DIR}")
    return treatment_data

# ============================================================================
# DATASET AND MODEL CODE (Same as before)
# ============================================================================
def check_dataset_structure():
    """Check if dataset exists"""
    print(f"Checking dataset at: {config.DATA_PATH}")
    
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ ERROR: Dataset path does not exist!")
        return False
    
    items = os.listdir(config.DATA_PATH)
    folders = [item for item in items if os.path.isdir(os.path.join(config.DATA_PATH, item))]
    
    if not folders:
        print(f"❌ No folders found in {config.DATA_PATH}")
        return False
    
    print(f"Found {len(folders)} folders in dataset:")
    for folder in folders:
        folder_path = os.path.join(config.DATA_PATH, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"  {folder}: {len(images)} images")
    
    return True

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
        
        for label_idx, folder_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, folder_name)
            
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

def create_model(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES):
    """Create and initialize the model"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 28 * 28, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = SimpleCNN(num_classes)
    
    return model

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
            'Loss': f'{running_loss/len(dataloader):.4f}',
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
# TREATMENT RECOMMENDATION SYSTEM
# ============================================================================
class TreatmentRecommender:
    """Provides treatment recommendations based on disease prediction"""
    
    def __init__(self, treatment_database):
        self.treatment_db = treatment_database
    
    def get_recommendation(self, disease_name, confidence, severity='medium'):
        """Get comprehensive treatment recommendation"""
        
        if disease_name not in self.treatment_db:
            return self._get_default_recommendation(disease_name)
        
        disease_info = self.treatment_db[disease_name]
        recommendation = {
            'disease': disease_name,
            'description': disease_info['description'],
            'confidence': f"{confidence:.1f}%",
            'severity': severity,
            'immediate_actions': [],
            'chemical_treatments': [],
            'organic_treatments': [],
            'cultural_practices': [],
            'long_term_management': []
        }
        
        # Add treatments based on severity
        if severity == 'high':
            recommendation['immediate_actions'].append("Apply chemical fungicide immediately")
            if 'chemical_control' in disease_info:
                recommendation['chemical_treatments'] = disease_info['chemical_control'][:2]
        
        elif severity == 'medium':
            recommendation['immediate_actions'].append("Apply fungicide within 3 days")
            if 'chemical_control' in disease_info:
                recommendation['chemical_treatments'] = disease_info['chemical_control'][:1]
            if 'organic_control' in disease_info:
                recommendation['organic_treatments'] = disease_info['organic_control'][:1]
        
        else:  # low severity
            recommendation['immediate_actions'].append("Monitor and apply preventive measures")
            if 'organic_control' in disease_info:
                recommendation['organic_treatments'] = disease_info['organic_control'][:1]
        
        # Add cultural practices
        if 'cultural_practices' in disease_info:
            recommendation['cultural_practices'] = disease_info['cultural_practices']
        
        # Add resistant varieties if available
        if 'resistant_varieties' in disease_info:
            recommendation['long_term_management'].append(
                f"Consider planting resistant varieties: {', '.join(disease_info['resistant_varieties'])}"
            )
        
        # Add spray schedule if available
        if 'spray_schedule' in disease_info:
            schedule_str = "Spray schedule: " + "; ".join(
                [f"{s['stage']}: {s.get('product', s.get('action', ''))}" 
                 for s in disease_info['spray_schedule']]
            )
            recommendation['long_term_management'].append(schedule_str)
        
        # For healthy leaves
        if disease_name == 'Health':
            recommendation['immediate_actions'] = ["Maintain current practices"]
            recommendation['cultural_practices'] = disease_info.get('preventive_measures', [])
            recommendation['long_term_management'] = [
                "Continue regular monitoring",
                "Apply preventive fungicides as per schedule",
                "Maintain tree health with balanced fertilization"
            ]
        
        return recommendation
    
    def _get_default_recommendation(self, disease_name):
        """Default recommendation for unknown diseases"""
        return {
            'disease': disease_name,
            'description': 'Unknown disease - consult agricultural expert',
            'general_advice': [
                'Remove and destroy infected leaves',
                'Apply broad-spectrum fungicide',
                'Improve air circulation',
                'Avoid overhead watering',
                'Consult local agricultural extension service'
            ]
        }
    
    def generate_report(self, recommendation):
        """Generate formatted treatment report"""
        report = []
        report.append("=" * 60)
        report.append(f"APPLE LEAF DISEASE TREATMENT REPORT")
        report.append("=" * 60)
        report.append(f"Disease: {recommendation['disease']}")
        report.append(f"Confidence: {recommendation['confidence']}")
        report.append(f"Severity: {recommendation['severity'].upper()}")
        report.append(f"Description: {recommendation['description']}")
        
        if recommendation['immediate_actions']:
            report.append("\nIMMEDIATE ACTIONS:")
            for i, action in enumerate(recommendation['immediate_actions'], 1):
                report.append(f"  {i}. {action}")
        
        if recommendation['chemical_treatments']:
            report.append("\nCHEMICAL TREATMENTS:")
            for i, treatment in enumerate(recommendation['chemical_treatments'], 1):
                report.append(f"  {i}. {treatment['product']}")
                report.append(f"     Rate: {treatment.get('rate', 'N/A')}")
                report.append(f"     Interval: {treatment.get('interval', 'N/A')}")
                report.append(f"     PHI: {treatment.get('phi', 'N/A')} days")
        
        if recommendation['organic_treatments']:
            report.append("\nORGANIC TREATMENTS:")
            for i, treatment in enumerate(recommendation['organic_treatments'], 1):
                report.append(f"  {i}. {treatment['product']}")
                report.append(f"     Rate: {treatment.get('rate', 'N/A')}")
                report.append(f"     Interval: {treatment.get('interval', 'N/A')}")
        
        if recommendation['cultural_practices']:
            report.append("\nCULTURAL PRACTICES:")
            for i, practice in enumerate(recommendation['cultural_practices'], 1):
                report.append(f"  {i}. {practice}")
        
        if recommendation['long_term_management']:
            report.append("\nLONG-TERM MANAGEMENT:")
            for i, management in enumerate(recommendation['long_term_management'], 1):
                report.append(f"  {i}. {management}")
        
        report.append("\n" + "=" * 60)
        report.append("IMPORTANT NOTES:")
        report.append("- Always follow label instructions")
        report.append("- Rotate fungicide classes to prevent resistance")
        report.append("- Wear protective equipment when spraying")
        report.append("- Consult local regulations for pesticide use")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_recommendation(self, recommendation, image_path, output_dir):
        """Save recommendation to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"treatment_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(recommendation, f, indent=4)
        
        # Save as text report
        report = self.generate_report(recommendation)
        txt_path = os.path.join(output_dir, f"treatment_report_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save CSV entry
        csv_path = os.path.join(output_dir, "treatment_history.csv")
        csv_entry = {
            'timestamp': timestamp,
            'image': os.path.basename(image_path),
            'disease': recommendation['disease'],
            'confidence': recommendation['confidence'],
            'severity': recommendation['severity'],
            'treatment_applied': '; '.join([t['product'] for t in recommendation.get('chemical_treatments', [])]) or 
                              '; '.join([t['product'] for t in recommendation.get('organic_treatments', [])])
        }
        
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_entry)
        
        print(f"✓ Treatment recommendations saved to:")
        print(f"  - {json_path}")
        print(f"  - {txt_path}")
        
        return report

# ============================================================================
# ENHANCED PREDICTOR WITH TREATMENT
# ============================================================================
class AppleLeafPredictorWithTreatment:
    """Enhanced predictor with treatment recommendations"""
    
    def __init__(self, model_path=None, treatment_db_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load treatment database
        if treatment_db_path and os.path.exists(treatment_db_path):
            with open(treatment_db_path, 'r') as f:
                self.treatment_db = json.load(f)
        else:
            self.treatment_db = create_treatment_database()
        
        self.recommender = TreatmentRecommender(self.treatment_db)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("Model path not provided. Training required.")
            self.model = None
    
    def _load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint.get('actual_classes', config.CLASS_NAMES)
        
        num_classes = len(self.class_names)
        self.model = create_model(config.MODEL_NAME, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded from {model_path}")
        print(f"  Classes: {self.class_names}")
    
    def predict_with_treatment(self, image_path, severity='medium'):
        """Predict disease and provide treatment recommendations"""
        
        if self.model is None:
            return "Model not loaded", 0.0, "Train or load model first"
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return f"Error loading image: {e}", 0.0, ""
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        disease_name = self.class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        
        # Get treatment recommendation
        recommendation = self.recommender.get_recommendation(
            disease_name, confidence_percent, severity
        )
        
        # Generate report
        report = self.recommender.generate_report(recommendation)
        
        # Save to file
        output_dir = "treatment_recommendations"
        os.makedirs(output_dir, exist_ok=True)
        saved_report = self.recommender.save_recommendation(
            recommendation, image_path, output_dir
        )
        
        return disease_name, confidence_percent, report
    
    def batch_predict(self, image_folder, severity='medium'):
        """Predict for all images in a folder"""
        results = []
        
        for img_file in os.listdir(image_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(image_folder, img_file)
                try:
                    disease, confidence, report = self.predict_with_treatment(img_path, severity)
                    results.append({
                        'image': img_file,
                        'disease': disease,
                        'confidence': confidence,
                        'report_saved': True
                    })
                    print(f"✓ Processed: {img_file} -> {disease} ({confidence:.1f}%)")
                except Exception as e:
                    results.append({
                        'image': img_file,
                        'error': str(e)
                    })
                    print(f"✗ Error processing {img_file}: {e}")
        
        return results

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Main training pipeline with treatment system"""
    
    print("=" * 70)
    print("APPLE LEAF DISEASE CLASSIFICATION + TREATMENT SYSTEM")
    print("=" * 70)
    
    # Create treatment database first
    print("\nCreating treatment database...")
    treatment_db = create_treatment_database()
    
    # Setup
    start_time = time.time()
    
    # Check dataset
    if not check_dataset_structure():
        print(f"\n❌ ERROR: Dataset path '{config.DATA_PATH}' not found!")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Get transforms and create datasets
    train_transform, val_transform = get_transforms()
    
    try:
        train_dataset = AppleLeafDataset(config.DATA_PATH, train_transform, 'train')
        val_dataset = AppleLeafDataset(config.DATA_PATH, val_transform, 'val')
        test_dataset = AppleLeafDataset(config.DATA_PATH, val_transform, 'test')
        
        actual_classes = train_dataset.class_folders
        print(f"\nUsing {len(actual_classes)} actual classes: {actual_classes}")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
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
    
    # Create model
    num_classes = len(actual_classes)
    print(f"\nCreating model: {config.MODEL_NAME} for {num_classes} classes")
    
    model = create_model(config.MODEL_NAME, num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "-" * 50)
    print("Starting training...")
    print("-" * 50)
    
    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(config.MODELS_DIR, f'best_model_with_treatment_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'actual_classes': actual_classes,
                'treatment_database': treatment_db,
                'config': vars(config)
            }, best_model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test evaluation
    print("\n" + "-" * 50)
    print("Evaluating on test set...")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    
    # Create enhanced predictor
    print("\n" + "-" * 50)
    print("Creating enhanced predictor with treatment system...")
    
    predictor = AppleLeafPredictorWithTreatment(best_model_path)
    
    # Save complete system
    complete_system_code = f'''# apple_leaf_complete_system.py
"""
Complete Apple Leaf Disease Detection & Treatment System
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {config.MODEL_NAME}
Test Accuracy: {test_acc:.2f}%
Classes: {actual_classes}

Usage:
    python apple_leaf_complete_system.py predict "path/to/image.jpg"
    python apple_leaf_complete_system.py batch "path/to/folder"
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced predictor
try:
    # This assumes the current file is in the same directory
    from apple_leaf_training_with_treatment import AppleLeafPredictorWithTreatment
except ImportError:
    print("Please run this script from the same directory as the training script")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == 'predict' and len(sys.argv) >= 3:
        image_path = sys.argv[2]
        severity = sys.argv[3] if len(sys.argv) >= 4 else 'medium'
        
        predictor = AppleLeafPredictorWithTreatment()
        disease, confidence, report = predictor.predict_with_treatment(image_path, severity)
        
        print("\\n" + "="*60)
        print(f"PREDICTION RESULT")
        print("="*60)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Disease: {disease}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Severity: {severity}")
        print("\\nTreatment report has been saved to 'treatment_recommendations/' folder")
        print("="*60)
    
    elif command == 'batch' and len(sys.argv) >= 3:
        folder_path = sys.argv[2]
        severity = sys.argv[3] if len(sys.argv) >= 4 else 'medium'
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found!")
            return
        
        predictor = AppleLeafPredictorWithTreatment()
        results = predictor.batch_predict(folder_path, severity)
        
        print(f"\\nProcessed {len(results)} images")
        print("Results saved to 'treatment_recommendations/' folder")
    
    elif command == 'help':
        print(__doc__)
    
    else:
        print("Invalid command. Use 'predict', 'batch', or 'help'")

if __name__ == "__main__":
    main()
'''
    
    system_path = os.path.join(config.OUTPUT_DIR, 'apple_leaf_complete_system.py')
    with open(system_path, 'w') as f:
        f.write(complete_system_code)
    
    # Create simple test script
    test_script = '''# test_treatment_system.py
"""
Test the complete apple leaf disease treatment system
"""

import os
import sys

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import the main script
    exec(open('apple_leaf_training_with_treatment.py').read())
    
    print("=" * 70)
    print("TESTING TREATMENT RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Create a test image (blank) if none exists
    test_image_path = "test_apple_leaf.jpg"
    if not os.path.exists(test_image_path):
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='green')
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    # Test the treatment database
    print("\\n1. Testing Treatment Database...")
    treatment_db = create_treatment_database()
    print(f"   ✓ Created database with {len(treatment_db)} diseases")
    
    # Test recommender
    print("\\n2. Testing Treatment Recommender...")
    recommender = TreatmentRecommender(treatment_db)
    
    # Test for Scab disease
    recommendation = recommender.get_recommendation('Scab', 95.5, 'high')
    print(f"   ✓ Generated recommendation for 'Scab'")
    print(f"   - Immediate actions: {len(recommendation['immediate_actions'])}")
    print(f"   - Chemical treatments: {len(recommendation.get('chemical_treatments', []))}")
    
    # Test report generation
    report = recommender.generate_report(recommendation)
    print(f"   ✓ Generated report ({len(report.split(chr(10)))} lines)")
    
    print("\\n3. Testing Complete System Setup...")
    print("   To use the complete system:")
    print("   python apple_leaf_complete_system.py predict test_apple_leaf.jpg")
    print("   python apple_leaf_complete_system.py batch path/to/folder")
    
    print("\\n" + "=" * 70)
    print("SYSTEM READY!")
    print("=" * 70)
    
except Exception as e:
    print(f"Error during test: {e}")
    print("\\nMake sure to run the training first:")
    print("python apple_leaf_training_with_treatment.py")
'''
    
    test_path = os.path.join(config.OUTPUT_DIR, 'test_treatment_system.py')
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE SYSTEM SUMMARY")
    print("=" * 70)
    print(f"\n✅ Training Completed Successfully!")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Training Time: {total_time/60:.1f} minutes")
    
    print(f"\n✅ Treatment Database Created:")
    print(f"   Diseases covered: {len(treatment_db)}")
    print(f"   Location: {config.TREATMENT_DIR}/")
    
    print(f"\n✅ Files Generated:")
    print(f"   1. Enhanced Predictor: apple_leaf_training_with_treatment.py (this file)")
    print(f"   2. Complete System: training_output/apple_leaf_complete_system.py")
    print(f"   3. Test Script: training_output/test_treatment_system.py")
    print(f"   4. Best Model: saved_models/best_model_with_treatment_{timestamp}.pth")
    print(f"   5. Treatment Database: treatment_database/")
    
    print(f"\n🚀 HOW TO USE:")
    print(f"   1. Detect disease & get treatment:")
    print(f'      python training_output/apple_leaf_complete_system.py predict "image.jpg"')
    print(f"   2. Batch process folder:")
    print(f'      python training_output/apple_leaf_complete_system.py batch "folder_path"')
    print(f"   3. Test the system:")
    print(f"      python training_output/test_treatment_system.py")
    
    print(f"\n📋 TREATMENT OUTPUT:")
    print(f"   All recommendations saved to 'treatment_recommendations/' folder")
    print(f"   Includes: JSON, TXT report, and CSV history")
    
    print("\n" + "=" * 70)

# ============================================================================
# RUN THE SYSTEM
# ============================================================================
if __name__ == "__main__":
    print("Initializing Apple Leaf Disease Detection & Treatment System...")
    
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
    
    # Run main function
    main()