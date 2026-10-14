# A heavy, complex model
import torch
import torch.nn as nn

class DeepTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10) # 10 digits
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# A tiny, lightweight model
class LightweightStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 32), # Completely flattens the image
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)
