import deepinv as dinv
import torch
from utils import set_seed
import matplotlib.pyplot as plt
from algorithms2acc import PGDStep
from tqdm import tqdm

def build_SPC_physics(rate: float, img_size: tuple[int], noise_lvl: float, ordering: str, device: torch.device):
    
    m = int(rate*img_size[1]*img_size[2])
    physics = dinv.physics.SinglePixelCamera(
        m=m,
        img_size=img_size,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_lvl),
        device=device,
        ordering=ordering,
    )
    
    return physics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(seed=0)

metric = dinv.metric.PSNR()

physics = build_SPC_physics(0.3, (3, 256, 256), 0.0, "cake_cutting", device)

x = dinv.utils.load_example("butterfly.png", device=device)
print(x.shape)

y = physics(x)
print(y.shape)

x0 = physics.A_adjoint(y)

# Reconstruction Algorithm

data_fidelity = dinv.optim.L2()
denoiser = dinv.models.DnCNN(device=device, pretrained="download_lipschitz")
prior = dinv.optim.PnP(denoiser=denoiser)
stepsize = 0.1 / physics.compute_norm(physics.A_adjoint(y), tol=1e-3).item()
_lambda = 0.01

pgd = PGDStep(data_fidelity=data_fidelity, prior=prior,
              stepsize=stepsize, lambd=_lambda)

max_iter = 500
PSNRs = []

x_k = x0
with torch.inference_mode():
    for k in tqdm(range(max_iter), colour='magenta', desc="PGD Step"):
        x_k = pgd.step(x_k, y, physics)
        PSNRs.append(metric(x_k, x).item())
# PLOT 

plt.figure()
plt.subplot(1, 5, 1)
plt.imshow(x.squeeze(0).permute(1,2,0))
plt.title('Original (x)')
plt.axis('off')
plt.subplot(1, 5, 2)
plt.imshow(y.squeeze().permute(1,2,0))
plt.title(f'Measurment (y) cr:0.3 \n {metric(y, x).item():.2f} dB')
plt.axis('off')
plt.subplot(1, 5, 3)
plt.imshow(x0.squeeze().permute(1,2,0))
plt.title(f'Reconstruction (x0) \n {metric(x0, x).item():.2f} dB')
plt.axis('off')
plt.subplot(1, 5, 4)
plt.imshow(x_k.squeeze().permute(1,2,0).detach().numpy())
plt.title(f'Reconstruction (xk) \n {metric(x_k, x).item():.2f} dB')
plt.axis('off')
plt.subplot(1, 5, 5)
plt.plot(PSNRs)
plt.title(f'PSNR per iter [dB]')
plt.gca().set_box_aspect(1) 
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()