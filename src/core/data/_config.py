from torchvision import transforms

DATA_DIR: str = "data"
PEDIATRIC_PNEUMONIA_TRANSFORMS = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])