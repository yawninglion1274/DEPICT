import numpy as np
import torch
from tqdm import tqdm
from utils.img_utils import get_device
from functools import partial
device = get_device()

def make_regulizer(tv_param, value_range_param, sparse_param, ROI=[None, None, None, None, None, None], step_size=0):
    value_range_regu = partial(projector_threshold,ROI=ROI, a = value_range_param[0], b=value_range_param[1])
    sparse_regu = partial(soft_threshold, thres = sparse_param * step_size) if sparse_param != None else indentical_map
    if  tv_param[0] == None:
        return lambda x: sparse_regu(value_range_regu(x))
    else:
        # def fista_regu(z):
        #     # print(d.device)
        #     # print(z.device)
        #     for i in (range(step_num)):
        #         del d
        #         # d = g_2.cpu() + ((q_1 - 1) / q_2) * (g_2.cpu() - g_1.cpu())
        #         # d = d.cuda()
        #     return sparse_regu(x).to('cuda:0')
        def fista_regu(z):
            tau = tv_param[0]
            step_num = tv_param[1]
            g_1 = operator_grad(z)
            d = g_1.clone()
            q_1 = 1
            gamma = 1 / (12 * tau)
            for i in (range(step_num)):
                g_2 = projector_grad(d + gamma * operator_grad(value_range_regu(z - tau * operator_div(d))))
                x = (value_range_regu(z - tau * operator_div(g_2)))
                q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1 ** 2))
                del d
                d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
                d = d
                q_1 = q_2
                g_1 = g_2
            return sparse_regu(x)
        return fista_regu

# def make_regulizer(tv_param, value_range_param, sparse_param):
#     if  tv_param[0] == None:
#         return lambda x: sparse_regu(value_range_regu(x))
#     else:
#         def fista_regu(z):
#             for i in (range(step_num)):
#             return x
#         return fista_regu


def indentical_map(x, *args):
    return x

def FGP_tv(z, tau, step_num = 1000, pixel_size_z = 1):

    g_1 = operator_grad(z, pixel_size_z)
    d = g_1.clone()
    q_1 = 1
    gamma = 1 / (12 * tau)
    for i in (range(step_num)):
        g_2 = projector_grad(d + gamma * operator_grad(projector_threshold(z - tau * operator_div(d, pixel_size_z)), pixel_size_z))
        x = projector_threshold(z - tau * operator_div(g_2, pixel_size_z))
        q_2 = 1 / 2 * (1 + np.sqrt(1 + 4 * q_1**2))
        d = g_2 + ((q_1 - 1) / q_2) * (g_2 - g_1)
        if((g_2 - g_1).abs().mean() < 1e-6):break
    return x#.to('cuda:0')

def soft_threshold(z, thres = 1e-3):
    print('thres', thres)
    return torch.sign(z)*torch.max(torch.abs(z) - thres,torch.zeros_like(z))

def projector_grad(g):
    '''
    g: shape(Nx, Ny, Nz, 3), isotropic case
    '''
    norm = torch.linalg.norm(g, dim = -1)
    norm[norm < 1] = 1
    norm = norm.reshape(g.shape[:-1] + (1,))
    g /= norm
    return g

def projector_threshold(x, ROI, a = 0, b=100, ):
    '''
    a < x < b
    '''
    x[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]]=torch.clamp(x[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]], min = a, max=b) #+ x.imag[ROI[0]:ROI[1], ROI[2]:ROI[3], ROI[4]:ROI[5]]
    return x

def operator_grad(x, pixel_size_z=1):
    '''
    x: shape (Nx, Ny, Nz)
    '''
    g = torch.zeros(x.shape + (3,), dtype = torch.float32, device=device)
    g[:-1, :, :, 0] = x[1:, :, :] - x[:-1, :, :] / pixel_size_z
    g[:, :-1, :, 1] = x[:, 1:, :] - x[:, :-1, :]
    g[:, :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1]
    return g

def operator_div(g, pixel_size_z=1):
    x = torch.zeros(g.shape[:-1], dtype = torch.float32, device=device)
    tmp = x.clone()
    tmp[1:-1, :, :] = (g[1:-1, :, :, 0] - g[:-2, :, :, 0])
    tmp[0, :, :] = g[0, :, :, 0]
    tmp[-1, :, :] = - g[-2, :, :, 0] 
    x += tmp / pixel_size_z
    tmp[:, 1:-1, :] = g[:, 1:-1, :, 1] - g[:, :-2, :, 1]
    tmp[:, 0, :] = g[:, 0, :, 1]
    tmp[:, -1, :] = - g[:, -2, :, 1]
    x += tmp
    tmp[:, :, 1:-1] = g[:, :, 1:-1, 2] - g[:, :, :-2, 2]
    tmp[:, :, 0] = g[:, :, 0, 2]
    tmp[:, :, -1] = - g[:, :, -2, 2]
    x += tmp
    return - x

def forward_delta_x(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[:-1, :, :] = value[1:, :, :] - value[:-1, :, :]
    return grad

def backward_delta_x(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[-1, :, :] =  - value[-2, :, :]
    grad[0, :, :] = value[0, :, :]
    grad[1:-1, :, :] = (value[1:-1, :, :] - value[:-2, :, :])
    return grad


def forward_delta_y(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[:, :-1, :] = value[:, 1:, :] - value[:, :-1, :]
    return grad

def backward_delta_y(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[:, -1, :] =  - value[:, -2, :]
    grad[:, 0, :] = value[:, 0, :]
    grad[:, 1:-1, :] = (value[:, 1:-1, :] - value[:, :-2, :])
    return grad

def forward_delta_z(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[:, :, :-1] = value[:, :, 1:] - value[:, :, :-1]
    return grad

def backward_delta_z(value):
    '''
    value: shape, (X, Y, Z)
    '''
    grad = torch.zeros(value.shape, dtype=torch.float32, device=device)
    grad[:, :, -1] =  - value[:, :, -2]
    grad[:, :, 0] = value[:, :, 0]
    grad[:, :, 1:-1] = (value[:, :, 1:-1] - value[:, :, :-2])
    return grad


def proj_pq(pq, alpha):
    return pq / torch.clamp(pq.abs() / alpha, min=1)

def proj_u(u, f, tau, beta):
    return (u + tau * beta * f) / (1 + tau * beta)

def grad_u_3d(u):
    p = torch.zeros(u.shape + (3,), dtype=torch.float32, device=device)
    p[..., 0] = forward_delta_x(u)
    p[..., 1] = forward_delta_y(u)
    p[..., 2] = forward_delta_z(u)
    return p

def sym_grad_p_3d(p):
    v = torch.zeros(p.shape[:-1] + (6,), dtype=torch.float32, device=device)
    v[..., 0] = forward_delta_x(p[..., 0])#xx
    v[..., 1] = forward_delta_y(p[..., 1])#yy
    v[..., 2] = forward_delta_y(p[..., 2])#zz
    v[..., 3] = (forward_delta_x(p[..., 1]) + forward_delta_y(p[..., 0])) / 2 #xy
    v[..., 4] = (forward_delta_x(p[..., 2]) + forward_delta_z(p[..., 0])) / 2 #xz
    v[..., 5] = (forward_delta_y(p[..., 2]) + forward_delta_z(p[..., 1])) / 2 #yz
    return v

def div_p_3d(p):
    u = backward_delta_x(p[..., 0]) + backward_delta_y(p[..., 1]) + backward_delta_z(p[..., 2])

    return u

def div_q_2d(q):
    w = torch.zeros(q.shape[:-1] + (3,), dtype=torch.float32, device=device)
    w[..., 0] = backward_delta_x(q[..., 0]) + backward_delta_y(q[..., 3]) + backward_delta_z(q[..., 4])
    w[..., 1] = backward_delta_x(q[..., 3]) + backward_delta_y(q[..., 1]) + backward_delta_z(q[..., 5])
    w[..., 2] = backward_delta_x(q[..., 4]) + backward_delta_y(q[..., 5]) + backward_delta_z(q[..., 2])
    return w

def tgv_pd_3d(f, alpha_1 = 0.008, max_iter=3000, beta=0.16, tau=1/24, sigma=1/4, alpha_0 = 0.008):
    '''
    3d, tgv, primal dual algorithm, alpha 0, control the max of 1-second grad, alpha 1, control the max of 2-second grad
    alpha-1 take the lead role, alpha-1 big, act as blur, alpha-1 small, act as TV, with staircase artifact
    
    '''
    u_k = f.clone()
    u_k_bar = f.clone()
    w_k = torch.zeros(f.shape + (3,), dtype=torch.float32, device=device)
    w_k_1 = torch.zeros(f.shape + (3,), dtype=torch.float32, device=device)
    w_k_bar = torch.zeros(f.shape + (3,), dtype=torch.float32, device=device)
    p_k = torch.zeros(f.shape + (3,), dtype=torch.float32, device=device)
    q_k = torch.zeros(f.shape + (6,), dtype=torch.float32, device=device)
    for i in tqdm(range(max_iter)):
    # for i in range(max_iter):
        p_k_1 = proj_pq(p_k + sigma * (grad_u_3d(u_k_bar) - w_k_bar), alpha_1)
        q_k_1 = proj_pq(q_k + sigma * sym_grad_p_3d(w_k_bar), alpha_0)
        u_k_1 = projector_threshold(proj_u(u_k + tau * div_p_3d(p_k_1), f, tau, beta))
        u_k_bar = 2 * u_k_1 - u_k
        w_k_1 = w_k + tau * (p_k + div_q_2d(q_k_1))
        w_k_bar = 2 * w_k_1 - w_k
        p_k = p_k_1
        q_k = q_k_1
        # if((u_k - u_k_1).abs().mean() < 1e-6):break
        u_k = u_k_1
    return u_k

def init_proximal_prior(prior_type, *args, **kwargs):
    if prior_type == 'tv':
        def prior(f):
            return FGP_tv(f, *args, **kwargs)
    elif prior_type == 'tgv':
        def prior(f):
            return tgv_pd_3d(f, *args, **kwargs)
    elif prior_type == 'GP':
        def prior(f):
            return projector_threshold(f, *args, **kwargs)
        
    return prior


    



