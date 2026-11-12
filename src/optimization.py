import numpy as np
import torch
from tqdm import tqdm


def FISTA(init_volume, step_num, step_size, grad_solver, regu_solver):
    s = init_volume.clone()
    q_1 = 1
    x_1 = init_volume
    pbar = tqdm(range(step_num))
    for i in pbar:
        grad, loss = grad_solver(s)
        z = s - (step_size) * grad
        pbar.set_description(f"Reconstruction MSE: {loss.detach()}")
        x_2 = regu_solver(z)
        q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1**2))
        s = x_2 + ((q_1 - 1) / q_2) * (x_2 - x_1)
        x_1 = x_2 
        q_1 = q_2
    return x_1