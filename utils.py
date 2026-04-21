import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random


def equal_energy_regularizer(individual_outputs, x):
    x_size = x.size(-1) * x.size(-2) * x.size(-3)
    loss = 0.0
    avg_energy_x = torch.linalg.norm(x.view(x.size(0), -1), dim=1) ** 2 / len(individual_outputs)
    for y in individual_outputs:
        e = torch.linalg.norm(y.view(y.size(0), -1), dim=1) ** 2
        loss += torch.mean((e - avg_energy_x) ** 2)
    return loss / (len(individual_outputs) * x_size)


def orthogonality_regularizer(individual_outputs):
    B = individual_outputs[0].shape[0]
    K = len(individual_outputs)

    Y = torch.stack([o.flatten(1) for o in individual_outputs], dim=1)

    denom = Y.norm(dim=2, keepdim=True)
    Yhat = Y / denom

    G = torch.bmm(Yhat, Yhat.transpose(1, 2))
    I = torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)
    return (G - I).pow(2).mean()


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n, c, h, 1, w, 1)
    out = out.view(n, c, scale * h, scale * w)
    return out


def sr_model(img, scale, img_size):
    A = torch.nn.AdaptiveAvgPool2d((img_size // scale, img_size // scale))
    Ap = lambda z: MeanUpsample(z, scale)

    y = A(img)
    x_hat = Ap(y)
    return y, x_hat


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images: int = None):
        self.image_dir = image_dir
        self.transform = transform

        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpg")]
        if max_images is not None:
            self.image_paths = self.image_paths[: max(0, max_images)]
        # Preload all images into RAM as tensors
        self.images = []
        # Use multithreading to speed up image loading

        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert("RGB")  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            self.images = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Loading images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _maybe_subset(dataset, debug_flag: bool):
    if not debug_flag:
        return dataset
    return torch.utils.data.Subset(dataset, range(min(20, len(dataset))))


def get_dataloaders(args):
    """Create data loaders based on dataset configuration."""
    debug_mode = getattr(args, "debug", False)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((args.img_size[1], args.img_size[2]), antialias=True), transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x)]
    )

    if args.dataset == "mnist":
        trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "fashionmnist":
        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "STL10":
        trainset = datasets.STL10(root="./data", split="train", download=True, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.STL10(root="./data", split="test", download=True, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "cifar10":
        root = "./data"
        already = os.path.isdir(os.path.join(root, "cifar-10-batches-py"))
        trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "celeba":
        dataset = ImageDataset(r"/home/hdspdeep/Usuarios/carlosmo/Agentic/data/CelebA2/train", transform=transform, max_images=20 if debug_mode else 5000)
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataset_test = ImageDataset(r"/home/hdspdeep/Usuarios/carlosmo/Agentic/data/CelebA2/test", transform=transform, max_images=20 if debug_mode else 500)
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    elif args.dataset == "places":
        print('with dataset places')
        dataset = ImageDataset(r"/home/hdspdeep/Usuarios/carlosmo/Agentic/data/places/train", transform=transform, max_images=20 if debug_mode else 5000)
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataset_test = ImageDataset(r"/home/hdspdeep/Usuarios/carlosmo/Agentic/data/places/test", transform=transform, max_images=20 if debug_mode else 500)
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
        
    elif args.dataset == "div2k":
        dataset = ImageDataset(r"/home/hdsp5090/Documents/Users/Roman/data/DIV2K_patches/patches/   train", transform=transform, max_images=20 if debug_mode else None)
        trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataset_test = ImageDataset(r"/home/hdsp5090/Documents/Users/Roman/data/DIV2K_patches/patches/valid", transform=transform, max_images=20 if debug_mode else None)
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    return trainloader, testloader


def get_dataloaders_testing(args):
    """Create data loaders based on dataset configuration."""
    debug_mode = getattr(args, "debug", False)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((args.n, args.n), antialias=True), transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x)]
    )

    if args.dataset == "mnist":
        trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "fashionmnist":
        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "cifar10":
        root = "./data"
        already = os.path.isdir(os.path.join(root, "cifar-10-batches-py"))
        trainset = datasets.CIFAR10(root=root, train=True, download=not already, transform=transform)
        trainloader = DataLoader(_maybe_subset(trainset, debug_mode), batch_size=args.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=root, train=False, download=not already, transform=transform)
        testloader = DataLoader(_maybe_subset(testset, debug_mode), batch_size=args.batch_size, shuffle=True)

    elif args.dataset == "celeba":

        dataset_test = ImageDataset(r"/home/hdsp5090/Documents/Users/Roman/data/CelebA2\test", transform=transform, max_images=20 if debug_mode else None)
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "div2k":

        dataset_test = ImageDataset(r"/home/hdsp5090/Documents/Users/Roman/data/DIV2K_patches\patches\valid", transform=transform, max_images=20 if debug_mode else None)
        testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    return testloader


#### Utility functions for matrix generation


def generate_matrix(m, n):

    path = f"matrices/fixed_matrix_{m}_{n}.pt"
    if os.path.exists(path):
        fixed_matrix = torch.load(path)
    else:
        fixed_matrix = torch.round(torch.rand(m, n, device="cuda")) * 2 - 1
        torch.save(fixed_matrix, path)
    return fixed_matrix


def generate_orthogonal_rows_qr(A: torch.Tensor, mB: int, device="cuda") -> torch.Tensor:
    mA, n = A.shape
    path = f"matrices/orthogonal_rows_{mA}_{mB}_{n}.pt"
    if os.path.exists(path):
        return torch.load(path)
    else:
        if mB == 0:
            return torch.empty((0, n), device=device, dtype=A.dtype)
        if mB > (n - mA):
            print(f"No se pueden extraer {mB} vectores de un nullspace de dim {n-mA}")

        # QR completa sobre A^T para tener Q_full (n×n)
        Q_full, _ = torch.linalg.qr(A.T, mode="complete")  # (n, n)
        nullspace_basis = Q_full[:, mA:]  # (n, n-mA)

        # combinaciones ortonormales aleatorias dentro del nullspace
        P = torch.randn(nullspace_basis.shape[1], mB, device=device, dtype=A.dtype)
        U, _ = torch.linalg.qr(P)  # Q reducido: (n-mA, mB)
        # cada fila nueva
        new_rows = U.T.matmul(nullspace_basis.T)  # (mB, n)
        torch.save(new_rows, path)
        return new_rows
