import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import deepinv as dinv
from tqdm import tqdm

class PGDStep(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, stepsize, lambd):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.stepsize = stepsize
        self.lambd = lambd

    def step(self, x_k, y, physics, **kwargs):
        """Algorithm PGD Step.
        
        :param torch.Tensor x_k: reconstruction at k
        :param torch.Tensor y: measurements.
        :param dinv.physics.Physics physics: measurement operator.
        :return: torch.Tensor: reconstructed image.
        """
        u = x_k - self.stepsize * self.data_fidelity.grad(
            x_k, y, physics
        )  # Gradient step
        x_k = self.prior.prox(
            u, sigma_denoiser=self.lambd * self.stepsize
        )  # Proximal step with denoiser

        return x_k

class ADMMStep(dinv.models.Reconstructor):
    def __init__(self, data_fidelity, prior, rho_admm, lambd):
        super().__init__()
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.rho_admm = rho_admm
        self.lambd = lambd

    def step(self, z_k, x_k, u_k, y, physics):

        z_next = self.prior.prox(x_k+u_k, sigma_denoiser=self.lambd)

        x_next = self.data_fidelity.prox(z_next-u_k, y, physics, gamma= 1.0/self.rho_admm)

        u_next = u_k + x_next - z_next

        return z_next, x_next, u_next

class AccStep():
    """
    Class to apply different acceleration schemes
    about a base operator (e.g. PGDStep.step).
    """

    def __init__(self, rho=0.9, max_m=5):
        self.rho = rho
        self.max_m = max_m  # Memoria máxima para no saturar RAM
        self.reset()

    def reset(self):
        # To FISTA
        self.t = 1.0

        # Internal States
        self.x_prev = None      # x_{k} (entrada al operador)
        self.z_prev = None      # z_{k} (salida acelerada)

        # To Anderson
        self.x_hist = []        # g(x_k)'s history
        self.f_hist = []      # r_k's history

    def step(self, x_next, action):
        """
        x_next: resultado del operador g(x_prev)
        action: esquema de aceleración
        """
        # History
        if self.x_prev is not None:
            # Residual r_k = g(x_k) - x_k
            res = (x_next - self.x_prev).detach()

            self.x_hist.append(x_next.detach())
            self.f_hist.append(res)

            # Keep the history of len m
            if len(self.x_hist) > self.max_m:
                self.x_hist.pop(0)
                self.f_hist.pop(0)

        # Select the accelerator
        if action == 0:
            z_next = self._no_accel(x_next)

        elif action == 1:
            z_next = self._fista_classic(x_next)

        elif action == 2:
            z_next = self._fixed_momentum(x_next)

        elif action in [3, 4, 5]:
            m = action   # 4->3, 5->4, 6->5
            z_next = self._anderson_math(x_next, m)

        else:
            raise ValueError("Unknown action")

        # Update x & z
        self.x_prev = z_next.detach()
        self.z_prev = z_next.detach()

        return z_next

    # --------------------------
    # Actions
    # --------------------------

    def _no_accel(self, x_next):
        return x_next

    def _fista_classic(self, x_next):
        if self.z_prev is None:
            return x_next

        t_new = 0.5 * (1.0 + (1.0 + 4.0 * self.t**2) ** 0.5)
        beta = (self.t - 1.0) / t_new

        # z_{k+1} = x_{k+1} + beta * (x_{k+1} - x_k)
        z_next = x_next + beta * (x_next - self.x_prev)

        self.t = t_new
        return z_next

    def _fixed_momentum(self, x_next):
        if self.x_prev is None:
            return x_next

        z_next = x_next + self.rho * (x_next - self.x_prev)
        return z_next

    def _anderson_math(self, x_next, m):
        """
        Compute Anderson Acceleration with memory (m).
        """
        k = len(self.f_hist)

        # We need at least 2 samples.
        if k < 2:
            return x_next

        # Use the last m elements
        actual_m = min(k, m)
        f_samples = self.f_hist[-actual_m:]
        x_samples = self.x_hist[-actual_m:]

        # Build matrices
        R = torch.stack([ri.reshape(-1) for ri in f_samples], dim=1)
        X = torch.stack([xi.reshape(-1) for xi in x_samples], dim=1)

        device = x_next.device
        n_cols = R.shape[1]

        # System KKT: min ||R @ alpha||^2 s.t. sum(alpha) = 1
        G = R.t() @ R
        G += torch.eye(n_cols, device=device) * 1e-6  # Reg to stability

        KKT = torch.zeros((n_cols + 1, n_cols + 1), device=device)
        KKT[:n_cols, :n_cols] = G
        KKT[:n_cols, n_cols] = 1.0
        KKT[n_cols, :n_cols] = 1.0

        rhs = torch.zeros(n_cols + 1, device=device)
        rhs[n_cols] = 1.0

        try:
            sol = torch.linalg.solve(KKT, rhs)
            alpha = sol[:n_cols]
            # linear combination of the operator's outputs
            z_next = (X @ alpha).reshape_as(x_next)
        except RuntimeError:
            # If solver fails, return without accelerating 
            z_next = x_next

        return z_next

# --- COMPONENTES RL (IGUALES AL MAIN) ---

@dataclass
class AlgoRLConfig:
    alpha_min: float # En el caso del ADMM este se usa para definir rho_admm min y rho_admm max
    alpha_max: float
    lambda_min: float
    lambda_max: float
    rho_min: float
    rho_max: float
    iter_cost: float
    convergence_weight: float
    terminal_weight: float
    discount: float
    entropy_beta: float
    value_coef: float
    reward_type: str = "custom"
    beta_clip: float = 1e-4
    w1: float = 1.0
    w2: float = 1.0

class AlgorithmRLRollout:
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.observations: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.actions: List[Dict[str, torch.Tensor]] = []

    def add(self, obs: torch.Tensor, info: Dict[str, torch.Tensor], reward: torch.Tensor, done: bool):
        self.observations.append(obs.detach().cpu())
        self.log_probs.append(info["logprob"])
        self.values.append(info["value"])
        self.entropies.append(info["entropy"])
        self.rewards.append(torch.as_tensor(reward, device=self.device))
        self.dones.append(done)
        self.actions.append(
            {
                "acc": info["acc"].detach(),
                "alpha": info["alpha"].detach(),
                "lambd": info["lambd"].detach(),
                "rho": info["rho"].detach(),
            }
        )
    def compute_returns(self, discount: float):

        if len(self.rewards) == 0:
            return torch.tensor([], device=self.device)
        R = torch.zeros(1, device=self.device)
        returns: List[torch.Tensor] = []
        for r, done in reversed(list(zip(self.rewards, self.dones))):
            if done:
                R = torch.zeros_like(R)
            R = r + discount * R
            returns.append(R)
        returns = list(reversed(returns))
        return torch.stack([r.squeeze() for r in returns])

# ==========================================
# POLICY
# ==========================================

class AlgorithmPolicy(nn.Module):
    def __init__(self, obs_dim: int, num_acc: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head_algo = nn.Linear(hidden_dim, num_acc)
        self.head_alpha = nn.Linear(hidden_dim, 2)
        self.head_lambda = nn.Linear(hidden_dim, 2)
        self.head_rho = nn.Linear(hidden_dim, 2)
        self.head_value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        feat = self.encoder(obs)
        return {
            "acc_logits": self.head_algo(feat),
            "alpha_params": self.head_alpha(feat),
            "lambda_params": self.head_lambda(feat),
            "rho_params": self.head_rho(feat),
            "value": self.head_value(feat).squeeze(-1),
        }

# ==========================================
# 3. CONTROLADOR INTEGRADO CON TU RED
# ==========================================

class AcceleratorController(nn.Module):
    OBS_DIM = 10 

    def __init__(self, 
                 acc_names: List[str], 
                 obs_dim: Optional[int] = None,
                 hidden_dim: int = 128,
                 config: Optional[AlgoRLConfig] = None, 
                 device: Optional[torch.device] = None):
        super().__init__()
        self.acc_names = acc_names
        self.obs_dim = obs_dim or self.OBS_DIM
        self.config = config or AlgoRLConfig()
        self.policy = AlgorithmPolicy(self.obs_dim, len(acc_names), hidden_dim=hidden_dim)
        if device is not None:
            self.to(device)

    def select_action(self, obs: torch.Tensor, deterministic: bool=False):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
        out = self.policy(obs_in)
        
        acc_logits = out["acc_logits"].squeeze(0)
        acc_dist = Categorical(logits=acc_logits)
        acc_sample = acc_dist.probs.argmax(dim=-1) if deterministic else acc_dist.sample()

        alpha_raw = F.softplus(out["alpha_params"].squeeze(0)) + self.config.beta_clip
        alpha_dist = Beta(alpha_raw[..., 0], alpha_raw[..., 1])
        alpha_unit = alpha_dist.mean if deterministic else alpha_dist.rsample()
        alpha = self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * alpha_unit

        lambda_raw = F.softplus(out["lambda_params"].squeeze(0)) + self.config.beta_clip
        lambda_dist = Beta(lambda_raw[..., 0], lambda_raw[..., 1])
        lambda_unit = lambda_dist.mean if deterministic else lambda_dist.rsample()
        lambd = self.config.lambda_min + (self.config.lambda_max - self.config.lambda_min) * lambda_unit

        rho_raw = F.softplus(out["rho_params"].squeeze(0)) + self.config.beta_clip
        rho_dist = Beta(rho_raw[..., 0], rho_raw[..., 1])
        rho_unit = rho_dist.mean if deterministic else rho_dist.rsample()
        rho = self.config.rho_min + (self.config.rho_max - self.config.rho_min) * rho_unit

        logprob = acc_dist.log_prob(acc_sample) + alpha_dist.log_prob(alpha_unit) + lambda_dist.log_prob(lambda_unit) + rho_dist.log_prob(rho_unit)
        entropy = acc_dist.entropy() + alpha_dist.entropy() + lambda_dist.entropy() + rho_dist.entropy()

        return {
            "acc": acc_sample, 
            "alpha": alpha, 
            "lambd": lambd, 
            "rho": rho,
            "logprob": logprob,
            "entropy": entropy,
            "value": out["value"].squeeze(0)
        }

    def update_policy(self, rollout: AlgorithmRLRollout, optimizer: torch.optim.Optimizer):
        returns = rollout.compute_returns(self.config.discount)
        if returns.numel() == 0: return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        log_probs = torch.stack(rollout.log_probs)
        values = torch.stack(rollout.values)
        advantages = returns.detach() - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = torch.stack(rollout.entropies).mean()
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_beta * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item(), 
                "policy_loss": policy_loss.item(), 
                "value_loss": value_loss.item(), 
                "entropy": entropy.item(),
                }

# --- SWITCHER ADAPTADO ---

class AcceleratorSwitcherPnP(dinv.models.Reconstructor):
    # Nombres que aparecerán en tus gráficas del Main
    DEFAULT_ACC = ("PGD", "FISTA", "Momentum", "AA_m3", "AA_m4", "AA_m5")

    def __init__(
        self,
        data_fidelity,
        prior_pnp,
        max_iter: int,
        base_stepsize: float,
        base_lambda: float,
        base_rho: float = 0.9,
        controller: Optional[AcceleratorController] = None,
        config: Optional[AlgoRLConfig] = None,
        acc_names: Optional[List[str]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        # Asignación de parámetros principales
        self.data_fidelity = data_fidelity
        self.prior_pnp = prior_pnp
        self.max_iter = max_iter
        self.eps = eps
        
        self.controller = controller
        self.config = config or AlgoRLConfig()  

        self.acc_names = acc_names or list(self.DEFAULT_ACC)
        
        # Guardar valores base para el observador
        self.default_alpha = base_stepsize
        self.default_lambda = base_lambda
        self.default_rho = base_rho
        
        # Motores base de optimización y aceleración
        self.pgd = PGDStep(data_fidelity, prior_pnp, base_stepsize, base_lambda)
        self.acc_engine = AccStep(rho=base_rho, max_m=5)
        self.psnr_metric = dinv.metric.PSNR()

    def _build_obs(
        self,
        x_k: torch.Tensor,
        x_prev: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        step: int,
        last_acc: Optional[int],
        alpha_prev: float,
        lambda_prev: float,
        rho_prev: float,
        gt: Optional[torch.Tensor],
    ):
        # Cálculo de métricas para la observación
        f_val = self.data_fidelity(x_k, y, physics).mean()
        
        grad = self.data_fidelity.grad(x_k, y, physics)
        grad_norm = torch.norm(grad)
        
        # 2. Residuo relativo (Estabilidad)
        residual = torch.norm(x_k - x_prev) / (torch.norm(x_prev) + self.eps)
        
        # 3. Ratios normalizados
        iter_ratio = torch.tensor(step / max(self.max_iter, 1), device=x_k.device)
        
        num_acc = len(self.acc_names)
        acc_ratio = torch.tensor(
            0.0 if last_acc is None else last_acc / max(num_acc - 1, 1), 
            device=x_k.device
        )
        
        if gt is not None:
            psnr_val = self.psnr_metric(x_k, gt).mean().detach()
        else:
            psnr_val = torch.tensor(0.0, device=x_k.device)

        aa_ratio = len(self.acc_engine.f_hist) / 5.0

        # 6. Construcción del vector de observación 10D
        obs_vec = torch.tensor([
            f_val.item(), 
            psnr_val.item(), 
            grad_norm.item(), 
            residual.item(),
            iter_ratio.item(), 
            acc_ratio.item(),
            float(alpha_prev), 
            float(lambda_prev), 
            float(rho_prev), 
            aa_ratio
        ], device=x_k.device, dtype=torch.float32)
        
        obs_vec = torch.nan_to_num(obs_vec, nan=0.0, posinf=1e3, neginf=-1e3)
        
        return obs_vec, f_val.detach(), psnr_val.detach(), residual.detach()

    def _compute_reward(
        self, 
        psnr_curr: torch.Tensor, 
        psnr_next: torch.Tensor, 
        f_curr: torch.Tensor, 
        f_next: torch.Tensor, 
        residual: torch.Tensor, 
        done: bool
    ):
        # Inicialización con el costo por iteración (C)
        reward = -self.config.iter_cost
        
        reward += self.config.w1 * (f_curr - f_next) 
        reward += self.config.w2 * (psnr_next - psnr_curr)
        
        # Penalización por residuo o inestabilidad
        reward -= self.config.convergence_weight * residual
        
        if done:
            # Premio extra por calidad final
            reward += self.config.terminal_weight * psnr_next
            
        return reward.detach()

    def forward(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        mode: str = "rl",
        controller: Optional[AcceleratorController] = None,
        rollout: Optional[AlgorithmRLRollout] = None,
        gt: Optional[torch.Tensor] = None,
        deterministic: bool = False,

        store_history: bool = False,
        history_stride: int = 1,
        return_actions: bool = False,
        fixed_params: dict = None,
        fixed_acc: Optional[str] = None,
    ):
        # Inicialización de la reconstrucción
        z_k = x0.clone()
        x_prev = x0.clone()
        self.acc_engine.reset()

        # Listas para historial
        x_hist = []
        actions_taken = []

        ctrl = controller or self.controller
        # Seguimiento de parámetros para el observador
        alpha_prev = self.default_alpha
        lambd_prev = self.default_lambda
        rho_prev = self.default_rho
        last_acc = None

        for it in tqdm(range(self.max_iter), desc="Algorithm", colour='green'):
            
            # Guardar historial si se solicita
            if store_history and it % history_stride == 0:
                x_hist.append(z_k.detach().cpu() if z_k.is_cuda else z_k.detach())

            # Obtener observación del estado actual
            obs, f_curr, psnr_curr, res_curr = self._build_obs(
                z_k, x_prev, y, physics, it, last_acc, alpha_prev, lambd_prev, rho_prev, gt
            )
            
            # Selección de acción
            if mode == "rl" and ctrl is not None:
                action = ctrl.select_action(obs, deterministic)
                
                acc_idx = int(action["acc"].item())
                alpha_curr = action["alpha"]
                lambd_curr = action["lambd"]
                rho_curr = action["rho"]
                if return_actions:
                    # Guardamos la info para las gráficas
                    actions_taken.append({
                        "step": it,
                        "acc_idx": acc_idx,
                        "alpha": alpha_curr.item(),
                        "lambda": lambd_curr.item(),
                        "rho": rho_curr.item()
                    })
            elif mode == "fixed":
                # Asegúrate de pasar el device de z_k
                acc_name = (fixed_acc or "pgd")
                acc_idx = self.acc_names.index(acc_name) if acc_name in self.acc_names else 0
                if it == 0: 
                    print(f'acelerador: {acc_idx}')
                    print(f'acc name: {acc_name}')
                alpha_curr = torch.tensor(fixed_params["alpha"] if fixed_params else alpha_prev, device=z_k.device)
                lambd_curr = torch.tensor(fixed_params["lambd"] if fixed_params else lambd_prev, device=z_k.device)
                rho_curr = torch.tensor(fixed_params["rho"] if fixed_params else rho_prev, device=z_k.device)
            else:
                # Caso por defecto (PGD puro sin aceleración)
                acc_idx = 0
                alpha_curr = torch.tensor(alpha_prev, device=z_k.device)
                lambd_curr = torch.tensor(lambd_prev, device=z_k.device)
                rho_curr = torch.tensor(rho_prev, device=z_k.device)

            
            self.pgd.stepsize = alpha_curr.item()
            self.pgd.lambd = lambd_curr.item()

            # Paso de optimización base (PGD)
            with torch.inference_mode():
                x_next = self.pgd.step(z_k, y, physics)
            
            # Aplicación de la aceleración seleccionada
            self.acc_engine.rho = rho_curr.item()
            with torch.inference_mode():
                z_next = self.acc_engine.step(x_next, acc_idx)

            # Cálculo de recompensa y almacenamiento de experiencia
            if mode == "rl" and gt is not None:
                # Observamos el resultado para el cálculo del delta
                _, f_next, psnr_next, res_next = self._build_obs(
                    z_next, 
                    z_k, 
                    y, 
                    physics, 
                    it + 1, 
                    acc_idx, 
                    alpha_curr.detach().item(), 
                    lambd_curr.detach().item(), 
                    rho_curr.detach().item(), 
                    gt
                ) # AQUI PODRIA SER LA VARIABLE X_NEXT SI ES EL CASO EN EL QUE SE QUIERE CALCULAR ANTES DE ACELERAR
                
                done = (it == self.max_iter - 1)
                res_reward = torch.norm(z_next-gt)/(torch.norm(gt) + self.eps)
                reward = self._compute_reward(psnr_curr, psnr_next, f_curr, f_next, res_reward, done)
                
                if rollout is not None and mode == "rl":
                    rollout.add(
                        obs, 
                        {"acc": torch.tensor(acc_idx), 
                         "alpha": alpha_curr, 
                         "lambd": lambd_curr, 
                         "rho": rho_curr, 
                         "logprob": action["logprob"], 
                         "entropy": action["entropy"], 
                         "value": action["value"]}, 
                        reward, 
                        done
                    )

            # Actualización de estados para la iteración k+1
            x_prev = z_k
            z_k = z_next
            
            alpha_curr.detach().item(), 
            lambd_curr.detach().item(), 
            rho_curr.detach().item()
            last_acc = acc_idx
            
        # --- RETORNO DINÁMICO ---
        out = [z_k]
        if store_history:
            out.append(x_hist)
        if return_actions:
            out.append(actions_taken)
            
        return out[0] if len(out) == 1 else tuple(out)
    
class AcceleratorSwitcherPnPADMM(dinv.models.Reconstructor):
    # Nombres que aparecerán en tus gráficas del Main
    DEFAULT_ACC = ("PGD", "FISTA", "Momentum", "AA_m3", "AA_m4", "AA_m5")

    def __init__(
        self,
        data_fidelity,
        prior_pnp,
        max_iter: int,
        base_rho_admm: float,
        base_lambda: float,
        base_rho: float = 0.9,
        controller: Optional[AcceleratorController] = None,
        config: Optional[AlgoRLConfig] = None,
        acc_names: Optional[List[str]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        # Asignación de parámetros principales
        self.data_fidelity = data_fidelity
        self.prior_pnp = prior_pnp
        self.max_iter = max_iter
        self.eps = eps
        
        self.controller = controller
        self.config = config or AlgoRLConfig()  

        self.acc_names = acc_names or list(self.DEFAULT_ACC)
        
        # Guardar valores base para el observador
        self.default_rho_admm = base_rho_admm
        self.default_lambda = base_lambda
        self.default_rho = base_rho
        
        # Motores base de optimización y aceleración
        self.admm = ADMMStep(data_fidelity=data_fidelity, prior=prior_pnp, rho_admm=base_rho_admm, lambd=base_lambda)
        self.acc_engine = AccStep(rho=base_rho, max_m=5)
        self.psnr_metric = dinv.metric.PSNR()

    def _build_obs(
        self,
        x_k: torch.Tensor,
        x_prev: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        step: int,
        last_acc: Optional[int],
        rho_admm_prev: float,
        lambda_prev: float,
        rho_prev: float,
        gt: Optional[torch.Tensor],
    ):
        # Cálculo de métricas para la observación
        f_val = self.data_fidelity(x_k, y, physics).mean()
        
        grad = self.data_fidelity.grad(x_k, y, physics)
        grad_norm = torch.norm(grad)
        
        # 2. Residuo relativo (Estabilidad)
        residual = torch.norm(x_k - x_prev) / (torch.norm(x_prev) + self.eps)
        
        # 3. Ratios normalizados
        iter_ratio = torch.tensor(step / max(self.max_iter, 1), device=x_k.device)
        
        num_acc = len(self.acc_names)
        acc_ratio = torch.tensor(
            0.0 if last_acc is None else last_acc / max(num_acc - 1, 1), 
            device=x_k.device
        )
        
        if gt is not None:
            psnr_val = self.psnr_metric(x_k, gt).mean().detach()
        else:
            psnr_val = torch.tensor(0.0, device=x_k.device)

        aa_ratio = len(self.acc_engine.f_hist) / 5.0

        # 6. Construcción del vector de observación 10D
        obs_vec = torch.tensor([
            f_val.item(), 
            psnr_val.item(), 
            grad_norm.item(), 
            residual.item(),
            iter_ratio.item(), 
            acc_ratio.item(),
            float(rho_admm_prev), 
            float(lambda_prev), 
            float(rho_prev), 
            aa_ratio
        ], device=x_k.device, dtype=torch.float32)
        
        obs_vec = torch.nan_to_num(obs_vec, nan=0.0, posinf=1e3, neginf=-1e3)
        
        return obs_vec, f_val.detach(), psnr_val.detach(), residual.detach()

    def _compute_reward(
        self, 
        psnr_curr: torch.Tensor, 
        psnr_next: torch.Tensor, 
        f_curr: torch.Tensor, 
        f_next: torch.Tensor, 
        residual: torch.Tensor, 
        done: bool
    ):
        # Inicialización con el costo por iteración (C)
        reward = -self.config.iter_cost
        
        reward += self.config.w1 * (f_curr - f_next) 
        reward += self.config.w2 * (psnr_next - psnr_curr)
        
        # Penalización por residuo o inestabilidad
        reward -= self.config.convergence_weight * residual
        
        if done:
            # Premio extra por calidad final
            reward += self.config.terminal_weight * psnr_next
            
        return reward.detach()

    def forward(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        mode: str = "rl",
        controller: Optional[AcceleratorController] = None,
        rollout: Optional[AlgorithmRLRollout] = None,
        gt: Optional[torch.Tensor] = None,
        deterministic: bool = False,

        store_history: bool = False,
        history_stride: int = 1,
        return_actions: bool = False,
        fixed_params: dict = None,
        fixed_acc: Optional[str] = None,
    ):
        # Inicialización de la reconstrucción
        z_k = x0.clone()
        x_k = x0.clone()
        x_prev = x0.clone()
        u_k = torch.zeros_like(x0)
        self.acc_engine.reset()

        # Listas para historial
        x_hist = []
        actions_taken = []

        ctrl = controller or self.controller
        # Seguimiento de parámetros para el observador
        rho_admm_prev = self.default_rho_admm
        lambd_prev = self.default_lambda
        rho_prev = self.default_rho
        last_acc = None

        for it in tqdm(range(self.max_iter), desc="Algorithm", colour='green'):
            
            # Guardar historial si se solicita
            if store_history and it % history_stride == 0:
                x_hist.append(z_k.detach().cpu() if z_k.is_cuda else z_k.detach())

            # Obtener observación del estado actual
            obs, f_curr, psnr_curr, res_curr = self._build_obs(
                x_k, x_prev, y, physics, it, last_acc, rho_admm_prev, lambd_prev, rho_prev, gt
            )
            
            # Selección de acción
            if mode == "rl" and ctrl is not None:
                action = ctrl.select_action(obs, deterministic)
                
                acc_idx = int(action["acc"].item())
                rho_admm_curr = action["alpha"]
                lambd_curr = action["lambd"]
                rho_curr = action["rho"]
                if return_actions:
                    # Guardamos la info para las gráficas
                    actions_taken.append({
                        "step": it,
                        "acc_idx": acc_idx,
                        "alpha": rho_admm_curr.item(),
                        "lambda": lambd_curr.item(),
                        "rho": rho_curr.item()
                    })
            elif mode == "fixed":
                # Asegúrate de pasar el device de z_k
                acc_name = (fixed_acc or "pgd")
                acc_idx = self.acc_names.index(acc_name) if acc_name in self.acc_names else 0
                if it == 0: 
                    print(f'acelerador: {acc_idx}')
                    print(f'acc name: {acc_name}')
                rho_admm_curr = torch.tensor(fixed_params["alpha"] if fixed_params else rho_admm_prev, device=z_k.device)
                lambd_curr = torch.tensor(fixed_params["lambd"] if fixed_params else lambd_prev, device=z_k.device)
                rho_curr = torch.tensor(fixed_params["rho"] if fixed_params else rho_prev, device=z_k.device)
            else:
                # Caso por defecto (PGD puro sin aceleración)
                acc_idx = 0
                rho_admm_curr = torch.tensor(rho_admm_prev, device=z_k.device)
                lambd_curr = torch.tensor(lambd_prev, device=z_k.device)
                rho_curr = torch.tensor(rho_prev, device=z_k.device)

            
            self.admm.rho_admm = rho_admm_curr.item()
            self.admm.lambd = lambd_curr.item()

            # Paso de optimización base (PGD)
            with torch.inference_mode():
                z_next, x_next, u_next = self.admm.step(z_k=z_k, x_k=x_k, u_k=u_k, y=y, physics=physics)
            
            # Aplicación de la aceleración seleccionada
            self.acc_engine.rho = rho_curr.item()
            with torch.inference_mode():
                x_next = self.acc_engine.step(x_next, acc_idx)

            # Cálculo de recompensa y almacenamiento de experiencia
            if mode == "rl" and gt is not None:
                # Observamos el resultado para el cálculo del delta
                _, f_next, psnr_next, res_next = self._build_obs(
                    x_next, 
                    x_k, 
                    y, 
                    physics, 
                    it + 1, 
                    acc_idx, 
                    rho_admm_curr.detach().item(), 
                    lambd_curr.detach().item(), 
                    rho_curr.detach().item(), 
                    gt
                ) # AQUI PODRIA SER LA VARIABLE X_NEXT SI ES EL CASO EN EL QUE SE QUIERE CALCULAR ANTES DE ACELERAR
                
                done = (it == self.max_iter - 1)
                res_reward = torch.norm(z_next-gt)/(torch.norm(gt) + self.eps)
                reward = self._compute_reward(psnr_curr, psnr_next, f_curr, f_next, res_reward, done)
                
                if rollout is not None and mode == "rl":
                    rollout.add(
                        obs, 
                        {"acc": torch.tensor(acc_idx), 
                         "alpha": rho_admm_curr, 
                         "lambd": lambd_curr, 
                         "rho": rho_curr, 
                         "logprob": action["logprob"], 
                         "entropy": action["entropy"], 
                         "value": action["value"]}, 
                        reward, 
                        done
                    )

            # Actualización de estados para la iteración k+1
            x_prev = x_k.clone()
            x_k = x_next
            z_k = z_next
            u_k = u_next
            
            rho_admm_curr.detach().item(), 
            lambd_curr.detach().item(), 
            rho_curr.detach().item()
            last_acc = acc_idx
            
        # --- RETORNO DINÁMICO ---
        out = [x_k]
        if store_history:
            out.append(x_hist)
        if return_actions:
            out.append(actions_taken)
            
        return out[0] if len(out) == 1 else tuple(out)