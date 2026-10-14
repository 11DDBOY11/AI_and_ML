import torch
import torch.nn as nn
import torch.optim as optim

from data import train_loader
from teach_stud import DeepTeacher, LightweightStudent


def train_teacher(teacher, train_loader, epochs=1):
    teacher.train()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = teacher(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Teacher epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")


def train_distillation(teacher, student, train_loader, epochs=3, T=3.0, alpha=0.3):
    teacher.eval() # Teacher is frozen, we do not calculate gradients for it
    student.train()
    
    optimizer = optim.Adam(student.parameters(), lr=0.003)
    
    # Loss components
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward passes
            student_logits = student(data)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # Loss Component 1: Truth labels (Hard loss)
            loss_hard = criterion_hard(student_logits, targets)
            
            # Loss Component 2: Teacher distributions (Soft loss with Temperature scaling)
            soft_student = nn.functional.log_softmax(student_logits / T, dim=-1)
            soft_teacher = nn.functional.softmax(teacher_logits / T, dim=-1)
            loss_soft = criterion_soft(soft_student, soft_teacher) * (T ** 2)
            
            # Combined Loss
            loss = (alpha * loss_hard) + ((1.0 - alpha) * loss_soft)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Distillation Loss: {running_loss/len(train_loader):.4f}")

# Example invocation (Assuming teacher is already initialized and pre-trained)
teacher_model = DeepTeacher() 
student_model = LightweightStudent()

train_teacher(teacher_model, train_loader, epochs=1)
train_distillation(teacher_model, student_model, train_loader, epochs=2)
