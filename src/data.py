import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=32, num_workers=4) -> tuple[DataLoader, DataLoader]:
    """
    Function to get data loaders for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing the training and testing data loaders.
    """
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Create data loaders for training and testing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )   

    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=64, num_workers=2)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")