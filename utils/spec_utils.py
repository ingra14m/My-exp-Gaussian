import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.quaternion_utils import init_predefined_omega


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts
    
class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi

        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)

    def forward(self, omega_o, a, la, mu):
        Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True)) # N, num_theta, num_phi, 1

        la = F.softplus(la - 1)
        mu = F.softplus(mu - 1)
        exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
        out = a * Smooth * torch.exp(exp_input)

        return out
    
class ASGRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(ASGRender, self).__init__()

        self.num_theta = 4
        self.num_phi = 8
        self.ch_normal_dot_viewdir = 1
        self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2 + self.ch_normal_dot_viewdir
        self.viewpe = viewpe
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, 'cuda')

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out
    
    def safe_normalize(self, x, eps=1e-8):
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

    def forward(self, pts, viewdirs, features, normal):
        asg_params = features.view(-1, self.num_theta, self.num_phi, 4)  # [N, 8, 16, 4]
        a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        reflect_dir = self.safe_normalize(self.reflect(-viewdirs, normal))

        color_feature = self.ree_function(reflect_dir, a, la, mu)
        # color_feature = color_feature.view(color_feature.size(0), -1, 3)
        color_feature = color_feature.view(color_feature.size(0), -1)  # [N, 256]

        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)  # [N, 1]
        indata = [color_feature, normal_dot_viewdir]
        if self.view_pe > -1:
            indata += [viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        # rgb = torch.sum(color_feature, dim=1)
        # rgb = torch.sigmoid(rgb)

        return rgb


class SpecularNetwork(nn.Module):
    def __init__(self, D=4, W=128, input_ch=3, output_ch=59, view_multires=4, multires=4):
        super(SpecularNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.view_multires = view_multires
        self.skips = [D // 2]
        
        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8
        # self.asg_hidden = self.num_theta * self.num_phi * 5
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # self.embed_view_fn, view_input_ch = get_embedder(view_multires, 3)
        # self.embed_fn, xyz_input_ch = get_embedder(multires, self.asg_feature)
        # self.input_ch = xyz_input_ch
        
        # self.linear = nn.ModuleList(
        #     [nn.Linear(self.input_ch, W)] + [
        #         nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
        #         for i in range(D - 1)]
        # )

        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)
        
        self.render_module = ASGRender(self.asg_hidden, 2, 2, 128)

    def forward(self, x, view, normal):
        # v_emb = self.embed_view_fn(view)
        # x_emb = self.embed_fn(x)
        # h = torch.cat([x_emb, v_emb], dim=-1)
        # h = x
        # for i, l in enumerate(self.linear):
        #     h = self.linear[i](h)
        #     h = F.relu(h)
        #     if i in self.skips:
        #         h = torch.cat([x_emb, h], -1)
        
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal)

        return spec
