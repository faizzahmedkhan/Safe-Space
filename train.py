import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.video as video_models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001
CLIP_LENGTH = 16
IMG_SIZE = 112

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD)

# Ensure data directories exist
os.makedirs(os.path.join(DATA_DIR, "safe"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "unsafe"), exist_ok=True)

# ==========================================
# DATASET DEFINITION
# ==========================================
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        
        # Class 0: Safe, Class 1: Unsafe
        classes = {"safe": 0, "unsafe": 1}
        
        for cls_name, cls_idx in classes.items():
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(os.path.join(cls_path, fname))
                    self.labels.append(cls_idx)
                    
        print(f"Found {len(self.samples)} video samples.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and convert to RGB
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        # If video is shorter than CLIP_LENGTH, pad it by repeating the last frame
        if len(frames) == 0:
            # Fallback for broken video
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * CLIP_LENGTH
            
        while len(frames) < CLIP_LENGTH:
            frames.append(frames[-1])
            
        # Sample exactly CLIP_LENGTH frames evenly
        indices = np.linspace(0, len(frames) - 1, CLIP_LENGTH).astype(int)
        sampled_frames = [frames[i] for i in indices]
        
        # Stack and convert to PyTorch Tensor -> (T, C, H, W)
        frames_array = np.stack(sampled_frames)
        tensor = torch.from_numpy(frames_array.copy()).permute(0, 3, 1, 2).float()
        tensor.div_(255.0)
        tensor = normalize(tensor)
        
        # R3D_18 expects (C, T, H, W)
        tensor = tensor.permute(1, 0, 2, 3)
        
        return tensor, torch.tensor(label, dtype=torch.long)

# ==========================================
# TRAINING SCRIPT
# ==========================================
def train():
    dataset = VideoDataset(DATA_DIR)
    if len(dataset) == 0:
        print(f"Please add some .mp4 videos to '{DATA_DIR}/safe' and '{DATA_DIR}/unsafe'.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Build Model Architecture
    model = video_models.r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # Load existing weights if present to fine-tune (Transfer Learning)
    model_path = 'safespace_bullying_model.pth'
    if os.path.exists(model_path):
        print(f"Loading existing weights from {model_path} for fine-tuning...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No existing weights found. Training from scratch...")
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Save the newly trained model
    print("Training complete. Saving new weights...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}. Your live app will now use this smarter model!")

if __name__ == "__main__":
    train()
