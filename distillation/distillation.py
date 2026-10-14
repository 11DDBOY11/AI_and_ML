import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a complex Teacher Model
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        return self.net(x)

# 2. Define a smaller Student Model
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2) # Just a single linear layer
    def forward(self, x):
        return self.fc(x)

# 3. Custom Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    T: Temperature scaling factor to soften probabilities.
    alpha: Weight balancing hard labels and soft teacher predictions.
    """
    # Hard Loss: Standard error relative to the real ground-truth labels
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    
    # Soft Loss: Compares student and teacher outputs softened by Temperature (T)
    soft_student = nn.functional.log_softmax(student_logits / T, dim=-1)
    soft_teacher = nn.functional.softmax(teacher_logits / T, dim=-1)
    soft_loss = nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss

# --- Dummy Training Setup ---
# Generate 10 random samples with 5 features, and 10 random binary labels
x_train = torch.randn(10, 5)
y_train = torch.randint(0, 2, (10,))

# Initialize models
teacher = TeacherModel() # Assume this is pre-trained
student = StudentModel()
optimizer = optim.Adam(student.parameters(), lr=0.01)

# Training Loop Step
student.train()
optimizer.zero_grad()

# Forward passes
student_logits = student(x_train)
with torch.no_grad(): # Keep teacher frozen
    teacher_logits = teacher(x_train)

# Calculate dynamic loss and update student weights
loss = distillation_loss(student_logits, teacher_logits, y_train, T=2.0, alpha=0.5)
loss.backward()
optimizer.step()

print(f"Distillation Step Complete. Loss: {loss.item():.4f}")
