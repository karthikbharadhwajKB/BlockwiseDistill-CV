import torch 
import wandb

from models import get_teacher_model, StudentModel
from data import get_data_loaders
from block_distill import blockwise_distillation

def evaluate(student, dataloader, device):
    student.eval()
    correct, total = 0, 0 
    with torch.no_grad():
        for images, labels in dataloader: 
            images, labels = images.to(device), labels.to(device)
            out, _, _ = student(images)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def main(): 
    wandb.init(project="blockwise-distillation")
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading data
    train_loader, test_loader = get_data_loaders(batch_size=64, num_workers=6)
    # Load the teacher model
    teacher = get_teacher_model(pretrained=True).to(device)
    # Load the student model
    student = StudentModel().to(device)

    # Perform block-wise distillation
    blockwise_distillation(teacher, student, train_loader, device, num_epochs=10)

    # Evaluate the student model after distillation
    accuracy = evaluate(student, test_loader, device)
    print(f"Student model accuracy after distillation: {accuracy:.2f}%")
    wandb.log({"Test_Accuracy": accuracy})

if __name__ == "__main__":
    main()
    wandb.finish()