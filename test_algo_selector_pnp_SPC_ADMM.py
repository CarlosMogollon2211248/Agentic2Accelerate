import argparse
import json
import os
import numpy as np
import deepinv as dinv
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from dataclasses import asdict
import pandas as pd

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from algo_selector_pnp_spc import build_SPC_physics
from algorithms2acc import (
    AlgoRLConfig,
    AcceleratorController,
    AcceleratorSwitcherPnPADMM
)
from utils import ImageDataset, set_seed

torch.serialization.add_safe_globals([AlgoRLConfig])
from deepinv.physics.blur import gaussian_blur

def build_blur_physics(n: int, sigma_noise: float, blur_sigma: float, grayscale: bool, device: torch.device):
    channels = 1 if grayscale else 3
    blur_filter = gaussian_blur(sigma=blur_sigma).to(device)
    if blur_filter.shape[1] == 1 and channels > 1:
        blur_filter = blur_filter.repeat(1, channels, 1, 1)
    physics = dinv.physics.BlurFFT(
        img_size=(channels, n, n),
        filter=blur_filter,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    )
    return physics

def _load_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _resolve_config_from_checkpoint(ckpt, fallback_cfg: AlgoRLConfig) -> AlgoRLConfig:
    cfg = ckpt.get("config")
    if cfg is None:
        return fallback_cfg
    if isinstance(cfg, dict):
        return AlgoRLConfig(**cfg)
    if isinstance(cfg, AlgoRLConfig):
        return AlgoRLConfig(**asdict(cfg))
    return fallback_cfg


def load_controller_from_checkpoint(controller: AcceleratorController, ckpt):
    state_dict = ckpt.get("controller_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint is missing controller_state_dict.")
    missing, unexpected = controller.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing parameters when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected parameters when loading checkpoint: {unexpected}")
    controller.eval()


def _clamp_unit_range(t: torch.Tensor) -> torch.Tensor:
    return torch.clamp(t, 0.0, 1.0)


def _prepare_image_for_save(t: torch.Tensor) -> torch.Tensor:
    return _clamp_unit_range(t.detach().cpu())


def _save_triplet_image(measurement: torch.Tensor, gt: torch.Tensor, reconstruction: torch.Tensor, path: str):
    """Save [measurement | ground-truth | reconstruction] grid for visual inspection."""
    imgs = torch.stack(
        [
            _prepare_image_for_save(measurement),
            _prepare_image_for_save(gt),
            _prepare_image_for_save(reconstruction),
        ],
        dim=0,
    )
    save_image(imgs, path, nrow=3)

def save_comparison_grid(all_images, output_dir):
    import matplotlib.pyplot as plt

    num_imgs = len(all_images)
    if num_imgs == 0:
        return

    # 🔹 ahora: 4 filas (métodos), N columnas (imágenes)
    fig, axes = plt.subplots(4, num_imgs, figsize=(3 * num_imgs, 10))

    if num_imgs == 1:
        axes = axes.reshape(4, 1)

    row_titles = ["Ground Truth", "Measurement", "ADMM", "RL Agent"]

    for j, data in enumerate(all_images):

        images = [
            _clamp_unit_range(data['gt'][0]),
            _clamp_unit_range(data['y'][0]),
            _clamp_unit_range(data['admm'][0]),
            _clamp_unit_range(data['rl'][0]),
        ]

        for i in range(4):
            ax = axes[i, j]
            img = images[i].permute(1, 2, 0).cpu().numpy()

            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)

            ax.axis('off')

            # 🔹 títulos de columnas (imagenes)
            if i == 0:
                ax.set_title(f"Image {j+1}", fontsize=14)

            # 🔹 títulos de filas (métodos)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=14)

        # =========================
        # 🔹 Label PSNR/SSIM en GT (fila 0)
        # =========================
        if j == 0:
            axes[0, j].text(
                0.98, 0.02, "PSNR / SSIM",
                color="white",
                fontsize=14,
                ha='right', va='bottom',
                transform=axes[0, j].transAxes,
                bbox=dict(facecolor='black', alpha=0.5, pad=2)
            )

        # =========================
        # 🔹 Métricas ADMM
        # =========================
        txt_pgd = f"{data['admm_psnr']:.1f} / {data['admm_ssim']:.3f}"
        axes[2, j].text(
            0.98, 0.02, txt_pgd,
            color="white",
            fontsize=14,
            ha='right', va='bottom',
            transform=axes[2, j].transAxes,
            bbox=dict(facecolor='black', alpha=0.5, pad=2)
        )

        # =========================
        # 🔹 Métricas RL
        # =========================
        txt_rl = f"{data['rl_psnr']:.1f} / {data['rl_ssim']:.3f}"
        axes[3, j].text(
            0.98, 0.02, txt_rl,
            color="white",
            fontsize=14,
            ha='right', va='bottom',
            transform=axes[3, j].transAxes,
            bbox=dict(facecolor='black', alpha=0.5, pad=2)
        )

    # 🔹 sin espacios
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    save_path = os.path.join(output_dir, "visual_comparison_grid.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"\n[OK] Grid TRANSPOSED guardado en: {save_path}")

def plot_rl_behavior(actions_rl, acc_names, idx_img, output_dir):
    """
    Versión mejorada:
    - Mantiene tus 2 subplots (decisiones + hiperparámetros)
    - Arriba: raster plot + histograma lateral
    """

    steps = [a["step"] for a in actions_rl]
    algo_ids = [a["acc_idx"] for a in actions_rl]
    alphas = [a["alpha"] for a in actions_rl]
    lambdas = [a["lambda"] for a in actions_rl]
    rhos = [a["rho"] for a in actions_rl]

    num_acc = len(acc_names)

    # 🔥 Layout principal
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1], wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])       # raster
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax1)  # histograma
    ax2 = fig.add_subplot(gs[1, :])       # hiperparámetros (igual que antes)

    # =========================
    # 🔹 AX1: Raster plot (líneas delgadas)
    # =========================
    ax1.step(steps, algo_ids, where='post', color='navy', linewidth=0.6, marker='o', markersize=0.9)

    ax1.set_yticks(range(num_acc))
    display_names = ["None" if name.upper() == "PGD" else name.upper() for name in acc_names]
    ax1.set_yticklabels(display_names)

    ax1.set_ylabel("Selected Iterator")
    ax1.set_title(f"Iteration")
    ax1.set_xlim(min(steps), max(steps))
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    # =========================
    # 🔹 Histograma lateral
    # =========================
    counts = np.bincount(algo_ids, minlength=num_acc)

    ax_hist.barh(range(num_acc), counts, color="navy", alpha=0.6)
    ax_hist.set_xlabel("Action\nFrequency")
    ax_hist.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Quitar labels duplicados
    plt.setp(ax_hist.get_yticklabels(), visible=False)

    # =========================
    # 🔹 AX2: Hiperparámetros (igual que tu versión)
    # =========================
    ax2.plot(steps, alphas, label=r"$\alpha$ (Gradient Step)", color='red')
    ax2.plot(steps, lambdas, label=r"$\lambda$ (Reg weight)", color='green')
    ax2.plot(steps, rhos, label=r"$\rho$ (Momentum)", color='purple')

    ax2.set_yscale('log')
    ax2.set_ylabel("Value (Log Scale)")
    ax2.set_xlabel("Iteration")
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # =========================
    # Guardar
    # =========================
    plt.subplots_adjust(wspace=0.05, hspace=0.25)  # 🔥 evita warning tight_layout
    plt.savefig(os.path.join(output_dir, f"rl_behavior_img_{idx_img+1}.svg"), format="svg")
    plt.close()

def plot_convergence_rate(curves_res, idx_img, styles, output_dir):
    """Genera la gráfica de la tasa de convergencia (Residual)"""
    plt.figure(figsize=(8, 5))
    for name, res_curve in curves_res.items():
        upper_name = name.upper() if name != "RL Agent" else "RL Agent"
        display_label = "ADMM" if upper_name == "PGD" else upper_name
        s = styles.get(name.lower() if name != "RL Agent" else "RL Agent", {'marker': 'v', 'alpha': 0.7})
        plt.plot(res_curve, 
                 label=display_label, 
                 color=s.get('color'), 
                 marker=s.get('marker'), 
                 markevery=0.1 if s.get('marker') else 1,
                 linewidth=s.get('lw', 1.5), 
                 zorder=s.get('zorder', 1))
    
    plt.xscale('log')
    plt.yscale('log') # Crítico para ver la tasa de convergencia
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.title(f"Convergence Rate (Step Norm) - Image {idx_img+1}")
    plt.xlabel("Iteration (Log Scale)")
    plt.ylabel("||x_k - x_{k-1}|| / ||x_{k-1}|| (Log Scale)")
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_img_{idx_img+1}.svg"), format='svg')
    plt.close()

def plot_gt_convergence(curves_gt, idx_img, styles, output_dir):
    """Genera la gráfica de Error Relativo con respecto al Ground Truth"""
    plt.figure(figsize=(8, 5))
    for name, gt_curve in curves_gt.items():
        upper_name = name.upper() if name != "RL Agent" else "RL Agent"
        display_label = "ADMM" if upper_name == "PGD" else upper_name
        s = styles.get(name.upper() if name != "RL Agent" else "RL Agent", {'marker': 'v', 'alpha': 0.7})
        plt.plot(gt_curve, 
                 label=display_label, 
                 color=s.get('color'), 
                 marker=s.get('marker'), 
                 markevery=0.1 if s.get('marker') else 1,
                 linewidth=s.get('lw', 1.5), 
                 zorder=s.get('zorder', 1))
    
    plt.xscale('log')
    plt.yscale('log') # Escala logarítmica para ver la caída del error
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.title(f"Error Relative to GT - Image {idx_img+1}")
    plt.xlabel("Iteration (Log Scale)")
    plt.ylabel("||x_k - x_GT|| / ||x_GT|| (Log Scale)")
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_gt_img_{idx_img+1}.svg"), format='svg')
    plt.close()

def evaluate_set5(switcher: AcceleratorSwitcherPnPADMM, physics, loader: DataLoader, device: torch.device, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    psnr_metric = dinv.metric.PSNR()
    ssim_metric = dinv.metric.SSIM()
    
    grid_data = [] 

    final_report = {acc: {"psnr": [], "ssim": []} for acc in switcher.acc_names}
    final_report["RL Agent"] = {"psnr": [], "ssim": []}

    print(f"\n>>> Iniciando evaluación. Generando 4 figuras (PSNR, SSIM, StepNorm, ErrorGT) por cada una de las {len(loader)} imágenes...")

    all_records = []        # curvas completas
    threshold_records = []  # iteración a cierto PSNR
    PSNR_TARGET = 31.0

    for idx_img, x_eval in enumerate(loader):
        x_eval = x_eval.to(device)
        y = physics(x_eval)
        x0 = physics.A_adjoint(y)
        
        curves = {} 
        curves_res = {} # Tasa de convergencia (x_k - x_{k-1})
        curves_gt = {}  # Error con respecto al GT (x_k - x_GT)
        temp_visual = {'gt': x_eval.cpu(),
                       'y': y.cpu() 
                       }

        # 1. Ejecutar Baselines Fijos
        for acc_name in switcher.acc_names:
            x_hat, x_hist = switcher(x0, y, physics, mode="fixed", fixed_acc=acc_name, store_history=True)
            p_curve = [psnr_metric(xk.to(device), x_eval).item() for xk in x_hist]
            s_curve = [ssim_metric(xk.to(device), x_eval).item() for xk in x_hist]
            
            # Cálculo de Residual y Error GT
            res_curve = [torch.norm(x_hist[i] - x_hist[i-1]).item() / torch.norm(x_hist[i-1]).item() for i in range(1, len(x_hist))]
            gt_err_curve = [torch.norm(xk.to(x_eval.device) - x_eval).item() / torch.norm(x_eval).item() for xk in x_hist]
            
            curves[acc_name] = {"psnr": p_curve, "ssim": s_curve}
            curves_res[acc_name] = res_curve
            curves_gt[acc_name] = gt_err_curve
            
            # =========================
            # Guardar datos por iteración (baseline)
            # =========================
            for i in range(len(p_curve)):
                all_records.append({
                    "image": idx_img,
                    "method": acc_name,
                    "iter": i,
                    "psnr": p_curve[i],
                    "ssim": s_curve[i],
                    "residual": res_curve[i-1] if i > 0 else None,
                    "error_gt": gt_err_curve[i],
                    "alpha": None,
                    "lambda": None, 
                    "rho": None,
                    "acc_idx": None,
                })

            # Iteración donde alcanza PSNR target
            iter_target = next((i for i, v in enumerate(p_curve) if v >= PSNR_TARGET), None)

            threshold_records.append({
                "image": idx_img,
                "method": acc_name,
                "iter_to_target_psnr": iter_target
            })

            final_report[acc_name]["psnr"].append(p_curve[-1])
            final_report[acc_name]["ssim"].append(s_curve[-1])

            if acc_name.lower() == 'pgd':
                temp_visual['admm'] = x_hat.cpu()
                temp_visual['admm_psnr'] = p_curve[-1]
                temp_visual['admm_ssim'] = s_curve[-1]

        # 2. Ejecutar Agente RL
        print(f'Ejecutando el Agente RL')
        x_hat_rl, x_hist_rl, actions_rl = switcher(x0, y, physics, mode="rl", controller=switcher.controller, 
                                          gt=x_eval, store_history=True, return_actions=True)
        
        plot_rl_behavior(actions_rl, switcher.acc_names, idx_img, output_dir)

        p_curve_rl = [psnr_metric(xk.to(device), x_eval).item() for xk in x_hist_rl]
        s_curve_rl = [ssim_metric(xk.to(device), x_eval).item() for xk in x_hist_rl]
        
        # Residual y Error GT para RL
        res_curve_rl = [torch.norm(x_hist_rl[i] - x_hist_rl[i-1]).item() / torch.norm(x_hist_rl[i-1]).item() for i in range(1, len(x_hist_rl))]
        gt_err_rl = [torch.norm(xk.to(x_eval.device) - x_eval).item() / torch.norm(x_eval).item() for xk in x_hist_rl]

        # =========================
        # Guardar datos RL
        # =========================
        print("len actions:", len(actions_rl))
        print("len hist:", len(x_hist_rl))
        for i, a in enumerate(actions_rl):
            all_records.append({
                "image": idx_img,
                "method": "RL Agent",
                "iter": i,
                "psnr": p_curve_rl[i],
                "ssim": s_curve_rl[i],
                "residual": res_curve_rl[i-1] if i > 0 else None,
                "error_gt": gt_err_rl[i],
                "alpha": a["alpha"],
                "lambda": a["lambda"], 
                "rho": a["rho"],
                "acc_idx": a["acc_idx"],
            })

        iter_target_rl = next((i for i, v in enumerate(p_curve_rl) if v >= PSNR_TARGET), None)

        threshold_records.append({
            "image": idx_img,
            "method": "RL Agent",
            "iter_to_target_psnr": iter_target_rl
        })

        curves["RL Agent"] = {"psnr": p_curve_rl, "ssim": s_curve_rl}
        curves_res["RL Agent"] = res_curve_rl
        curves_gt["RL Agent"] = gt_err_rl
        
        final_report["RL Agent"]["psnr"].append(p_curve_rl[-1])
        final_report["RL Agent"]["ssim"].append(s_curve_rl[-1])

        temp_visual['rl'] = x_hat_rl.cpu()
        temp_visual['rl_psnr'] = p_curve_rl[-1]
        temp_visual['rl_ssim'] = s_curve_rl[-1]
        grid_data.append(temp_visual)

        # --- 3. Estilos y Guardado de Figuras ---
        styles = {
            'PGD': {'color': 'black', 'marker': 'o'},
            'FISTA': {'color': '#4B0082', 'marker': 's'},
            'Momentum': {'color': "#343BA0", 'marker': 'D'},
            'AA_m3': {'color': '#A29100', 'marker': '^'},
            'AA_m4': {'color': "#00F514", 'marker': 'v'},
            'AA_m5': {'color': "#F700FF", 'marker': 'p'},
            'RL Agent': {'color': 'red', 'marker': None}
        }

        def save_single_plot(metric_key, y_label, title_prefix, filename):
            plt.figure(figsize=(8, 5))
            for name, data in curves.items():
                upper_name = name.upper() if name != "RL Agent" else "RL Agent"
                display_label = "ADMM" if upper_name == "PGD" else upper_name
                s = styles.get(name.upper() if name != "RL Agent" else "RL Agent", {'marker': 'v', 'alpha': 0.7})
                plt.plot(data[metric_key], label=display_label, color=s.get('color'), marker=s.get('marker'), 
                         markevery=0.1 if s.get('marker') else 1, linewidth=s.get('lw', 1.5), zorder=s.get('zorder', 1))
            plt.xscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.xlabel("Iteration (Log Scale)")
            plt.ylabel(y_label)
            plt.title(f"{title_prefix} - Image {idx_img+1}")
            plt.legend(loc='lower right', shadow=True, fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), format='svg')
            plt.close()

        save_single_plot("psnr", "PSNR (dB)", "PSNR Convergence", f"psnr_img_{idx_img+1}.svg")
        save_single_plot("ssim", "SSIM Index", "SSIM Convergence", f"ssim_img_{idx_img+1}.svg")
        plot_convergence_rate(curves_res, idx_img, styles, output_dir)
        plot_gt_convergence(curves_gt, idx_img, styles, output_dir) # NUEVA GRÁFICA
        
        print(f"Figuras guardadas para imagen {idx_img+1}: [psnr, ssim, convergence, error_gt]")

    save_comparison_grid(grid_data, output_dir)

    print("\n" + "="*60)

    # =========================
    # Guardar CSVs
    # =========================

    df_curves = pd.DataFrame(all_records)
    df_summary = pd.DataFrame([
        {
            "method": name,
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_ssim": np.mean(metrics["ssim"])
        }
        for name, metrics in final_report.items()
    ])
    df_threshold = pd.DataFrame(threshold_records)

    df_curves.to_csv(os.path.join(output_dir, "curves_full.csv"), index=False)
    df_summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    df_threshold.to_csv(os.path.join(output_dir, "psnr_threshold.csv"), index=False)

    print("[OK] CSVs guardados")

    print(f"{'MÉTODO':<18} | {'PSNR PROM.':<15} | {'SSIM PROM.':<12}")
    print("-" * 60)
    for name, metrics in final_report.items():
        avg_p = np.mean(metrics["psnr"])
        avg_s = np.mean(metrics["ssim"])

        display_name = "ADMM" if name.upper() == "PGD" else name.upper()
        tag = " [WINNER]" if name == "RL Agent" else ""
        print(f"{display_name:<18} | {avg_p:>12.2f} dB | {avg_s:>12.4f} {tag}")
    print("="*60 + "\n")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(seed=0)

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((args.img_size[1], args.img_size[2]), antialias=True),
            transforms.Grayscale() if args.grayscale else transforms.Lambda(lambda x: x),
        ]
    )
    set5_dataset = ImageDataset(args.set5_dir, transform=eval_transform, max_images=args.max_eval_images)
    if len(set5_dataset) == 0:
        raise RuntimeError(f"No images found in {args.set5_dir}.")
    set5_loader = DataLoader(set5_dataset, batch_size=1, shuffle=False)

    physics = build_SPC_physics(args.cr, args.img_size, args.sigma, args.ordering, device)
    # physics = build_blur_physics(args.img_size[1], args.sigma, args.blur_sigma, args.grayscale, device) # n, sigma_noise, blur_sigma, grayscale 
    sample = set5_dataset[0].unsqueeze(0).to(device)
    # y_init = physics(sample)

    data_fidelity = dinv.optim.L2()
    denoiser = dinv.models.DnCNN(device=device, pretrained="download_lipschitz")
    prior = dinv.optim.PnP(denoiser=denoiser)

    # stepsize = 0.1 / physics.compute_norm(physics.A_adjoint(y_init), tol=1e-3).item()
    base_lambda = args.lambd
    cfg = AlgoRLConfig(
        # Adaptar límites al problema específico
        alpha_min=1e-3,
        alpha_max=5.0,
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
    ckpt = _load_checkpoint(args.rl_checkpoint, device)
    cfg = _resolve_config_from_checkpoint(ckpt, cfg)
    controller = AcceleratorController(acc_names=list(AcceleratorSwitcherPnPADMM.DEFAULT_ACC), config=cfg, device=device)
    load_controller_from_checkpoint(controller, ckpt)
    print(f"Loaded controller checkpoint: {args.rl_checkpoint} (epoch={ckpt.get('epoch')})")
    switcher = AcceleratorSwitcherPnPADMM(
        data_fidelity=data_fidelity,
        prior_pnp=prior,
        max_iter=args.max_iter,
        base_rho_admm=args.base_rho_admm,
        base_lambda=base_lambda,
        base_rho=args.base_rho,
        controller=controller,
        config=cfg,
    )

    evaluate_set5(switcher, physics, set5_loader, device, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pretrained RL algorithm selector on Set5 images for SPC.")
    parser.add_argument("--rl_checkpoint", type=str, default="/home/hdspdeep/Usuarios/carlosmo/Agentic/AgentReg/results/rl_acc_selector_spcADMM_epoch0005.pth", help="Path to controller checkpoint.")
    parser.add_argument("--set5_dir", type=str, default="/home/hdspdeep/Usuarios/carlosmo/Agentic/Set5_HR", help="Directory containing the Set5 images.")
    parser.add_argument("--cr", type=float, default=0.7, help="Compression Rate")
    parser.add_argument("--img_size", type=tuple[int], default=(3, 128, 128), help="Image size C x H x W.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Gaussian noise level.")
    parser.add_argument("--ordering", type=str, default="cake_cutting", help="The ordering of selecting the first m measurements, available options (sequency, cake_cutting, zig_zag, xy, old_sequency).")
    parser.add_argument("--blur_sigma", type=float, default=3.5, help="Gaussian blur sigma (pixels).")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale images.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Iterations per reconstruction.")
    parser.add_argument("--lambd", type=float, default=0.01, help="Base regularization weight for PnP denoiser.")
    parser.add_argument("--base_rho_admm", type=float, default=0.9, help="Base penalty ADMM")
    parser.add_argument("--base_rho", type=float, default=0.9, help="Base relaxation for splitting methods.")
    parser.add_argument("--rl_iter_cost", type=float, default=0.0001, help="Per-iteration penalty in reward.")
    parser.add_argument("--rl_conv_coef", type=float, default=0.7, help="Weight on convergence penalty.")
    parser.add_argument("--rl_terminal_weight", type=float, default=1.0, help="Terminal PSNR weight.")
    parser.add_argument("--rl_discount", type=float, default=0.96, help="Discount factor.")
    parser.add_argument("--rl_entropy_beta", type=float, default=1e-3, help="Entropy regularization.")
    parser.add_argument("--rl_value_coef", type=float, default=0.1, help="Value loss weight.")
    parser.add_argument("--output_dir", type=str, default="/home/hdspdeep/Usuarios/carlosmo/Agentic/AgentReg/figures_resultsAccADMM/set5_eval_SPC_sigma_0.02_cr0.7insvg", help="Directory to save plots and logs.")
    parser.add_argument("--max_eval_images", type=int, default=5, help="Maximum number of Set5 images to evaluate.")
    args = parser.parse_args()

    main(args)
