from tqdm.auto import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
import wandb 

def blockwise_distillation(teacher, student, dataloader, device, num_epochs=2): 
    teacher.eval()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss() 

    print("Starting blockwise distillation...")

    # Stage-1: Block-1
    # Freeze all parameters except for the first block of the student model
    print("===== Stage-1: Training Block-1 =====")
    for param in student.block1.parameters():
        param.requires_grad = True
    for param in student.block2.parameters():
        param.requires_grad = False
    for param in student.fc.parameters():
        param.requires_grad = False

    # Initialize the optimizer for the first block
    optimizer = optim.Adam(student.block1.parameters(), lr=1e-3)

    # Training loop for the first block
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Stage-1, Epoch {epoch + 1}/{num_epochs}", unit="batch")
        # Iterate over the dataloader
        for images, _ in progress_bar: 
            images = images.to(device)
            # Forward pass through the teacher and student models
            with torch.no_grad(): 
                # Forward pass through the teacher model
                teacher_x1 = teacher.conv1(images)
                teacher_x1 = teacher.bn1(teacher_x1)
                teacher_x1 = teacher.relu(teacher_x1)
                teacher_x1 = teacher.maxpool(teacher_x1)
            # Forward pass through the student model
            _, student_x1, _ = student(images)
            # Compute the loss between the student and teacher outputs
            # Using MSE loss for block-wise distillation
            loss = mse_loss(student_x1, teacher_x1)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(dataloader)
        wandb.log({"Stage1_Block1_Loss": avg_loss, "Epoch": epoch + 1})

    # Stage-2: Block-2
    # Freeze all parameters except for the second block of the student model
    print("===== Stage-2: Training Block-2 =====")
    for param in student.block1.parameters():
        param.requires_grad = False
    for param in student.block2.parameters():
        param.requires_grad = True 
    for param in student.fc.parameters():
        param.requires_grad = False
    # Initialize the optimizer for the second block
    optimizer = optim.Adam(student.block2.parameters(), lr=1e-3)

    # Training loop for the second block
    for epoch in range(num_epochs): 
        running_loss = 0.0 
        progress_bar = tqdm(dataloader, desc=f"Stage-2, Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for images, _ in progress_bar: 
            images = images.to(device)
            with torch.no_grad():
                # Forward pass through the teacher model
                teacher_x1 = teacher.conv1(images)
                teacher_x1 = teacher.bn1(teacher_x1)
                teacher_x1 = teacher.relu(teacher_x1)
                teacher_x1 = teacher.maxpool(teacher_x1)
                teacher_x2 = teacher.layer1(teacher_x1)
            # Forward pass through the student model
            _, _, student_x2 = student(images)
            # Compute the loss between the student and teacher outputs
            loss = mse_loss(student_x2, teacher_x2)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        avg_loss = running_loss / len(dataloader)
        wandb.log({"Stage2_Block2_Loss": avg_loss, "Epoch": epoch + 1})

    # Stage-3: Fully Connected Layer (Classifier)
    # Freeze all parameters except for the fully connected layer of the student model
    print("===== Stage-3: Training Fully Connected Layer =====")
    for param in student.block1.parameters():
        param.requires_grad = False
    for param in student.block2.parameters():
        param.requires_grad = False
    for param in student.fc.parameters():
        param.requires_grad = True

    # Initialize the optimizer for the fully connected layer
    optimizer = optim.Adam(student.fc.parameters(), lr=1e-3)

    # Training loop for the fully connected layer
    for epoch in range(num_epochs):
        running_loss = 0.0 
        progress_bar = tqdm(dataloader, desc=f"Stage-3, Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for images, labels in progress_bar: 
            images, labels = images.to(device), labels.to(device)
            # Forward pass through the student model
            out, _, _ = student(images)
            loss = ce_loss(out, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        avg_loss = running_loss / len(dataloader)
        wandb.log({"Stage3_Classification_Loss": avg_loss, "Epoch": epoch + 1})