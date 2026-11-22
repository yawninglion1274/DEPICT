import torch


def angular_spectrum_kernel(domain_size, spec_pixel_size, pixel_size, km):
    kx = (torch.linspace((-domain_size[1] // 2+1), (domain_size[1] // 2 ), domain_size[1]) - 1) * spec_pixel_size
    ky = (torch.linspace((-domain_size[2] // 2+1), (domain_size[2] // 2 ), domain_size[2]) - 1) * spec_pixel_size
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    kernel = torch.exp(1j * Kz * pixel_size)
    ol_correction = km / Kz
    ol_correction[-K2 + (km * 1.2/1.33)**2<0]=0.
    return (torch.fft.fftshift(kernel)), (torch.fft.fftshift(ol_correction))
# This angular-spectrum kernel propagates the full field U = a * exp(1j * km * z),
# whereas the conventional nonparaxial BPM kernel propagates the envelope a.
# With km = k0 * nm and kz = sqrt(km**2 - kx**2 - ky**2):
#   H_envelope = exp(1j * (kz - km) * dz)
#   H_field    = exp(1j * km * dz) * H_envelope = exp(1j * kz * dz)
# Thus, both kernels describe the same homogeneous-background propagation;
# the full-field kernel includes the reference-carrier phase.

def bpm_kernel(domain_size, spec_pixel_size, pixel_size, km):
    kx = torch.linspace((-domain_size[1] // 2), (domain_size[1] // 2 ), domain_size[1]) * spec_pixel_size
    ky = torch.linspace((-domain_size[2] // 2), (domain_size[2] // 2 ), domain_size[2]) * spec_pixel_size
    [Ky, Kx] = torch.meshgrid(ky, kx, indexing='ij')
    K2 = Kx**2 + Ky**2
    Kz = torch.sqrt(-K2 + km**2 + 0j)
    kernel = torch.exp(-1j * K2 * pixel_size[0]/ (km + Kz))
    return torch.fft.fftshift(kernel)


def compute_bpm_grad(init_delta_ri, u_in, u_out, batch_size, cos_factor, dz, domain_size, k0, p_kernel, device):
    ol_factor = k0 * dz / cos_factor.unsqueeze(-1)
    p_kernel = p_kernel.unsqueeze(0)
    bp_kernel = p_kernel.conj()
    grad = torch.zeros_like(init_delta_ri)
    s = torch.zeros((batch_size,)+init_delta_ri.shape, dtype=torch.cfloat, device=device)
    delta_ri = init_delta_ri
    loss=0
    for i in range(u_in.shape[0]//batch_size):

        start_idx = i * batch_size
        end_idx = (i + 1)*batch_size if (i + 1) * batch_size < u_in.shape[0] else None
        sub_u_in = u_in[start_idx:end_idx, ...]
        sub_u_out = u_out[start_idx:end_idx, ...]
        u = sub_u_in
        if(torch.any(torch.isnan(ol_factor[start_idx:end_idx, ...]))):
            continue
        for m in range(init_delta_ri.shape[0]):
            u = torch.fft.ifft2(torch.fft.fft2(u) * p_kernel)#广播机制
            s[:, m, ...] = u.conj()
            u = u * torch.exp(1j * ol_factor[start_idx:end_idx, ...] * delta_ri[m, ...].unsqueeze(0))
        r = u - sub_u_out

        loss +=r.abs().mean()
        for m in reversed(range(init_delta_ri.shape[0])):
            r = r * torch.exp(-1j * ol_factor[start_idx:end_idx, ...] * delta_ri[m, ...].unsqueeze(0))
            s[:, m, ...] = -1j * ol_factor[start_idx:end_idx, ...] * s[:, m, ...] * r
            r = torch.fft.ifft2(torch.fft.fft2(r) * bp_kernel)
        grad = grad + s.sum(axis=0)
    grad = grad / u_in.shape[0]
    loss = loss/(u_in.shape[0]//batch_size)
    print('loss', loss)
    return grad, loss
