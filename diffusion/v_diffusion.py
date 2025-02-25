from ddpm_1d import GaussianDiffusion1D

import torch
import torch.nn.functional as F


class GaussianDiffusion1DDefault(GaussianDiffusion1D):
    def __init__(self, model, seq_length, objective, betas,  time_scale=1, gamma = 0):
        super(GaussianDiffusion1DDefault, self).__init__(model=model, seq_length=seq_length, betas=betas, objective = objective, time_scale=time_scale)
        self.gamma = gamma

    def distill_loss(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = self.q_sample(x, t+1) 
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.model(z.float(), t.float() + 1)
            rec = self.predict_start_from_v(z, t+1, v)

            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)

            v_1 = self.model(z_1.float(), t.float()).double()
            x_2 = self.predict_start_from_v(z_1, t, v_1) 

            eps_2 = (z - alpha_s * x_2) / sigma_s
            v_2 = alpha_s * eps_2 - sigma_s * x_2
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
        v = student_diffusion.model(z.float(), t.float() * self.time_scale, **extra_args)
        my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        return F.mse_loss(w * v.float(), w * v_2.float())