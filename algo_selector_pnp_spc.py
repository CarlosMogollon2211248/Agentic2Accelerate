import argparse
import os

import deepinv as dinv
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from deepinv.physics.blur import gaussian_blur

# from algorithms import (
#     AlgoRLConfig,
#     AlgorithmController,
#     AlgorithmRLRollout,
#     AlgorithmSwitcherPnP,
# )

from algorithms2acc import (
    AlgoRLConfig,
    AcceleratorController,
    AlgorithmRLRollout,
    AcceleratorSwitcherPnP
)

torch.serialization.add_safe_globals([AlgoRLConfig])
from utils import ImageDataset, get_dataloaders, set_seed


def _save_controller_checkpoint(controller, optimizer, args, epoch: int = None):
    """Persist controller/optimizer state, optionally tagging with the epoch number."""
    ckpt = {
        "controller_state_dict": controller.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": controller.config,
        "args": vars(args),
        "epoch": epoch,
    }
    base_path = args.rl_checkpoint
    root, ext = os.path.splitext(base_path)
    if epoch is not None:
        ckpt_path = f"{root}_epoch{epoch:04d}{ext or '.pth'}"
    else:
        ckpt_path = base_path
    ckpt_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    tag = f" epoch {epoch}" if epoch is not None else ""
    print(f"Saved controller{tag} to {ckpt_path}")


def build_SPC_physics(rate: float, img_size: tuple[int], noise_lvl: float, ordering: str, device: torch.device):
    print('Trabajando con SPC')
    m = int(rate*img_size[1]*img_size[2])
    physics = dinv.physics.SinglePixelCamera(
        m=m,
        img_size=img_size,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_lvl),
        device=device,
        ordering=ordering,
    )
    
    return physics

def train_controller(
    switcher: AcceleratorSwitcherPnP,
    controller: AcceleratorController,
    trainloader,
    physics,
    args,
    device: torch.device,
):
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.rl_lr)

    # Para darle robustez ante el ruido
    max_sigma = 0.075  
    start_sigma = 1e-5  
    for ep in range(args.rl_episodes):
        epoch_max_sigma = start_sigma + (max_sigma - start_sigma) * (ep / args.rl_episodes)
        pbar = tqdm(trainloader, desc=f"RL epoch {ep+1}", total=min(len(trainloader), args.rl_max_samples), colour='magenta')
        for step_idx, x_batch in enumerate(pbar):
            if step_idx >= args.rl_max_samples:
                break
            current_sigma = np.random.uniform(0, epoch_max_sigma)
            physics.noise_model.sigma = torch.tensor(current_sigma, device=device)
            # current_sigma = torch.empty(1, device=device).uniform_(1e-5, 0.08)
            # current_sigma = 10**(-5 + torch.rand(1).item() * 4)
            # physics.noise_model.sigma = torch.tensor(current_sigma, device=device)
            x_batch = x_batch.to(device)
            y_batch = physics(x_batch)
            x0_batch = physics.A_adjoint(y_batch)
            rollout = AlgorithmRLRollout(device)
            switcher(
                x0_batch,
                y_batch,
                physics,
                mode="rl",
                controller=controller,
                rollout=rollout,
                gt=x_batch,
                deterministic=False,
                store_history=False,
            )
            stats = controller.update_policy(rollout, optimizer)
            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                policy=f"{stats['policy_loss']:.4f}",
                value=f"{stats['value_loss']:.4f}",
                entropy=f"{stats['entropy']:.4f}",
                sigma=f"{current_sigma:.6f}",
            )
        pbar.close()
        _save_controller_checkpoint(controller, optimizer, args, epoch=ep + 1)
    _save_controller_checkpoint(controller, optimizer, args)


def evaluate(switcher: AcceleratorSwitcherPnP, physics, args, device: torch.device):
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((args.img_size[1], args.img_size[2]), antialias=True),
            transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x),
        ]
    )
    set5_dataset = ImageDataset(args.eval_dir, transform=eval_transform)
    set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False)
    psnr_metric = dinv.metric.PSNR()

    results = []
    first_curve = None

    for idx_img, x_eval in enumerate(set5_loader):
        x_eval = x_eval.to(device)
        # physics.noise_model.sigma = torch.tensor(0.00, device=device)
        y = physics(x_eval)
        x0 = physics.A_adjoint(y)

        store_history = True
        history_stride = 1

        x_hat_pgd, x_hist_pgd = switcher(
            x0,
            y,
            physics,
            mode="fixed",
            fixed_params={"alpha": switcher.default_alpha, "lambd": switcher.default_lambda, "rho": switcher.default_rho},
            store_history=store_history,
            history_stride=history_stride,
        )
        x_hat_rl, x_hist_rl, actions_rl = switcher(
            x0,
            y,
            physics,
            mode="rl",
            controller=switcher.controller,
            rollout=None,
            gt=x_eval,
            deterministic=False,
            store_history=store_history,
            history_stride=history_stride,
            return_actions=True,
        )

        res_img = {
            "pgd": psnr_metric(x_hat_pgd, x_eval).mean().item(),
            "rl": psnr_metric(x_hat_rl, x_eval).mean().item(),
        }
        results.append(res_img)

        if store_history:
            x_eval_cpu = x_eval.cpu()
            psnr_pgd = [psnr_metric(x_k.to(x_eval_cpu.device), x_eval_cpu).mean().item() for x_k in x_hist_pgd]
            psnr_rl = [psnr_metric(x_k.to(x_eval_cpu.device), x_eval_cpu).mean().item() for x_k in x_hist_rl]
            if first_curve is None:
                first_curve = {"pgd": psnr_pgd, "rl": psnr_rl}

            steps = [a["step"] for a in actions_rl]
            algo_ids = [a["acc_idx"] for a in actions_rl]
            alphas = [a["alpha"] for a in actions_rl]
            lambdas = [a["lambda"] for a in actions_rl]
            rhos = [a["rho"] for a in actions_rl]

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            axes[0].plot(range(len(psnr_rl)), psnr_rl, label="PSNR (RL)")
            axes[0].plot(range(len(psnr_pgd)), psnr_pgd, label="PSNR (PGD)", linestyle="--")
            axes[0].set_ylabel("PSNR (dB)")
            axes[0].grid(True)
            axes[0].legend()

            axes[1].plot(steps, algo_ids, marker="o")
            axes[1].set_yticks(range(len(switcher.acc_names)))
            axes[1].set_yticklabels(switcher.acc_names)
            axes[1].set_ylabel("Selected iterator")
            axes[1].grid(True, axis="x")

            axes[2].plot(steps, alphas, label="alpha (step)")
            axes[2].plot(steps, lambdas, label="lambda (PnP reg)")
            axes[2].plot(steps, rhos, label="rho (momentum)")
            axes[2].set_yscale("log")
            axes[2].set_ylabel("Hyperparameters (log)")
            axes[2].set_xlabel("Iteration")
            axes[2].legend()
            axes[2].grid(True)

            fig.suptitle(f"Convergence and choices - Set5 image {idx_img+1}")
            os.makedirs("figures_resultsAcc", exist_ok=True)
            fig.savefig(f"figures_resultsAcc/rl_selection_spc_image{idx_img+1}pgd.png", bbox_inches="tight")
            plt.close(fig)

    print("PSNR per Set5 image:")
    for idx_img, res in enumerate(results):
        print(f"Image {idx_img+1}: PGD={res['pgd']:.2f} | RL={res['rl']:.2f}")

    if first_curve is not None:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(first_curve["pgd"])
        plt.plot(first_curve["rl"])
        plt.legend(["PGD baseline", "RL algo selector"])
        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.title("PSNR vs iteration (first Set5 image, deblurring)")
        plt.grid()
        os.makedirs("figures_resultsAcc", exist_ok=True)
        plt.savefig(f"figures_resultsAcc/rl_acc_selector_spc_cr{args.cr}_sigma{args.sigma}pgd.png")
        plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(seed=0)

    physics = build_SPC_physics(args.cr, args.img_size, args.sigma, args.ordering, device)

    trainloader, _ = get_dataloaders(args)
    x_init = next(iter(trainloader)).to(device)
    y_init = physics(x_init)

    data_fidelity = dinv.optim.L2()
    denoiser = dinv.models.DnCNN(device=device, pretrained="download_lipschitz")
    prior = dinv.optim.PnP(denoiser=denoiser)

    stepsize = 0.1 / physics.compute_norm(physics.A_adjoint(y_init), tol=1e-3).item()
    base_lambda = args.lambd
    cfg = AlgoRLConfig(
        # Adaptar límites al problema específico
        alpha_min=max(1e-5, stepsize * 0.1),
        alpha_max=stepsize * 5.0,
        lambda_min=base_lambda * 0.2,
        lambda_max=base_lambda * 5.0,
        rho_min=1e-3,
        rho_max=1.5,
        convergence_weight=args.rl_conv_coef,
        # Tus pesos personalizados
        w1=1.0, w2=1.0,
        iter_cost=args.rl_iter_cost, 
        terminal_weight=args.rl_terminal_weight,
        
        # Estabilidad RL
        discount=args.rl_discount,
        entropy_beta=args.rl_entropy_beta,
        value_coef=args.rl_value_coef,
        beta_clip=1e-4
        )
    controller = AcceleratorController(acc_names=list(AcceleratorSwitcherPnP.DEFAULT_ACC), config=cfg, device=device)
    switcher = AcceleratorSwitcherPnP(
        data_fidelity=data_fidelity,
        prior_pnp=prior,
        max_iter=args.max_iter,
        base_stepsize=stepsize,
        base_lambda=base_lambda,
        base_rho=args.base_rho,
        controller=controller,
        config=cfg,
    )

    if os.path.isfile(args.rl_checkpoint):
        ckpt = torch.load(args.rl_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = controller.load_state_dict(ckpt.get("controller_state_dict", {}), strict=False)
        if missing or unexpected:
            print(f"Loaded controller with missing={missing} unexpected={unexpected}")

    if args.rl_episodes > 0 and not args.eval_only:
        train_controller(switcher, controller, trainloader, physics, args, device)

    evaluate(switcher, physics, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL selection over deepinv optim iterators for Gaussian deblurring with PnP DnCNN.")
    parser.add_argument("--cr", type=float, default=0.3, help="Compression Rate")
    parser.add_argument("--img_size", type=tuple[int], default=(3, 128, 128), help="Image size C x H x W.")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise level added to the SPC.")
    parser.add_argument("--ordering", type=str, default="cake_cutting", help="The ordering of selecting the first m measurements, available options (sequency, cake_cutting, zig_zag, xy, old_sequency).")
    parser.add_argument("--dataset", type=str, default="places", help="Dataset (celeba, mnist, fashionmnist, cifar10, div2k).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale images.")
    parser.add_argument("--max_iter", type=int, default=500, help="Iterations per reconstruction.")
    parser.add_argument("--lambd", type=float, default=0.01, help="Base regularization weight for PnP denoiser.")
    parser.add_argument("--base_rho", type=float, default=0.9, help="Base relaxation for splitting methods.")
    parser.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur sigma (pixels).")
    parser.add_argument("--rl_episodes", type=int, default=10, help="Number of RL training epochs.")
    parser.add_argument("--rl_lr", type=float, default=1e-4, help="RL learning rate.")
    parser.add_argument("--rl_max_samples", type=int, default=5000, help="Max samples per RL epoch.")
    parser.add_argument("--rl_iter_cost", type=float, default=0.001, help="Per-iteration penalty in reward.")
    parser.add_argument("--rl_conv_coef", type=float, default=0.7, help="Weight on convergence penalty.")
    parser.add_argument("--rl_terminal_weight", type=float, default=1.0, help="Terminal PSNR weight.")
    parser.add_argument("--rl_discount", type=float, default=0.83, help="Discount factor.")
    parser.add_argument("--rl_entropy_beta", type=float, default=1e-3, help="Entropy regularization.")
    parser.add_argument("--rl_value_coef", type=float, default=0.1, help="Value loss weight.")
    parser.add_argument("--rl_checkpoint", type=str, default="results/rl_acc_selector_spc_exp2.pth", help="Controller checkpoint path.")
    parser.add_argument("--eval_only", action="store_true", help="Skip RL training.")
    parser.add_argument("--debug", action="store_true", help="Load only a small debug subset (20 images).")
    parser.add_argument("--eval_dir", type=str, default=r"/home/hdspdeep/Usuarios/carlosmo/Agentic/Set5_HR", help="Directory of evaluation images (e.g., Set5).")
    args = parser.parse_args()

    main(args)
