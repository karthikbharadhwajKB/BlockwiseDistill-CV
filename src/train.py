import os
import torch 
from torch import nn
import wandb
from datetime import datetime
from models import get_teacher_model, StudentModel
from data import get_data_loaders
from block_distill import blockwise_distillation

def save_model(model, save_path, epoch_info=None):
    """Saves the model with metadata."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'time_stamp': datetime.now().isoformat(), 
    }

    if epoch_info:
        save_dict.update(epoch_info)

    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0 
    with torch.no_grad():
        for images, labels in dataloader: 
            images, labels = images.to(device), labels.to(device)
            # handle different model outputs
            outputs = model(images)
            if isinstance(outputs, tuple):
                out = outputs[0] # Student model returns logits and intermediate features
            else:
                out = outputs # Teacher model returns only logits
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def train_teacher_on_mnist(teacher, train_dataloader, device, num_epochs=5):
    """Fine-tunes the teacher model on MNIST."""
    model_dir = "saved_models"
    # check if model is already fine-tuned
    save_path = os.path.join(model_dir, "teacher_model_finetuned.pth")
    if os.path.exists(save_path):
        print(f"Teacher model already fine-tuned. Loading from {save_path}")
        return torch.load(save_path, map_location=device)

    teacher.train()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Fine-tuning teacher model on MNIST...")

    for epoch in range(num_epochs):
        running_loss = 0.0 
        correct, total = 0, 0
        for images, labels in train_dataloader: 
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            _ , preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    print("Teacher model fine-tuning complete.")
    # save the fine-tuned teacher model
    save_model(teacher, save_path, {
        "final_accuracy": accuracy,
        "num_epochs": num_epochs,
        "dataset": "MNIST",
    })
    return teacher


# Training Configuration
NUM_EPOCHS = 5

def main(): 
    try:
        wandb.init(project="blockwise-distillation")
        use_wandb = True
        print("WandB initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        use_wandb = False
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # loading data
    train_loader, test_loader = get_data_loaders(batch_size=128, num_workers=6)
    print(f"Loaded {len(train_loader)} training batches and {len(test_loader)} test batches.")
    # Load the teacher model
    teacher = get_teacher_model(pretrained=True).to(device)
    # Load the student model
    student = StudentModel().to(device)
    print("Models initialized successfully.")

    # Fine-tune the teacher model on MNIST
    ft_teacher = train_teacher_on_mnist(teacher, train_loader, device, num_epochs=NUM_EPOCHS)

    # Evaluate teacher baseline
    teacher_accuracy = evaluate(ft_teacher, test_loader, device)
    print(f"Teacher baseline accuracy: {teacher_accuracy:.2f}%")

    # Perform block-wise distillation
    blockwise_distillation(ft_teacher, student, train_loader, device, num_epochs=NUM_EPOCHS, use_wandb=use_wandb)

    # Evaluate the student model after distillation
    student_accuracy = evaluate(student, test_loader, device)
    print(f"Student model accuracy after distillation: {student_accuracy:.2f}%")

    if use_wandb:
        wandb.log({
            "Teacher_Accuracy": teacher_accuracy,
            "Student_Accuracy": student_accuracy,
            "Accuracy_Retention": student_accuracy / teacher_accuracy * 100
        })

    # Save the student model
    model_dir = "saved_models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"student_model_{timestamp}.pth")
    save_model(student, save_path, {
        "final_accuracy": student_accuracy,
        "teacher_accuracy": teacher_accuracy,
        "num_epochs": NUM_EPOCHS,
        "dataset": "MNIST",
    })

    print(f"Student model saved to: {save_path}")

if __name__ == "__main__":
    main()
    wandb.finish()