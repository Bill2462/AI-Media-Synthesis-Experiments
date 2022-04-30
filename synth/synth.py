import os
import torch
import clip
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer
from taming.models.vqgan import VQModel
from PIL import ImageFile, Image

def load_vqgan(checkpoint_filepath, config_filepath, device,
               model_type="conditional"):

    conf = OmegaConf.load(config_filepath)

    if model_type == "conditional":
        model = Net2NetTransformer(**conf.model.params)
        model.init_from_ckpt(checkpoint_filepath)
        model.eval().requires_grad_(False)
        model = model.first_stage_model.to(device)

    elif model_type == "unconditional":
        model = VQModel(**conf.model.params)
        model.init_from_ckpt(checkpoint_filepath)
        model.eval().requires_grad_(False).to(device)

    else:
        raise ValueError("Invalid model type, only 'conditional' and 'unconditional' are supported")

    return model

def load_clip(model_filepath, device):
    model, preprocessor = clip.load(model_filepath, device=device, jit=False)
    model = model.eval().requires_grad_(False)

    return model, preprocessor

def random_z_init(width, height, model, device):
    f = 2**(model.decoder.num_resolutions - 1)
    n_toks = model.quantize.n_e
    e_dim = model.quantize.e_dim

    input_x = width // f
    input_y = height // f

    indexes = torch.randint(n_toks, [input_x * input_y], device=device)
    one_hot = F.one_hot(indexes, n_toks).float()
    z = one_hot @ model.quantize.embedding.weight

    return z.view([-1, input_y, input_x, e_dim]).permute(0, 3, 1, 2)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

def vector_quantize(z, codebook):
    d = z.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * z @ codebook.T
    indices = d.argmin(-1)
    z_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(z_q, z)

def generate(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return model.decode(z_q).add(1).div(2).clamp(0, 1)

def run(opt, z, objective, model_vqgan, model_clip, niter, display_progress=True,
        save_interval=100, save_path="img", image_extension=".png", on_save_cb=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if display_progress:
        pbar = tqdm(range(niter))
    else:
        pbar = range(niter)

    for i in pbar:
        opt.zero_grad()
        img = generate(z, model_vqgan)
        loss = objective(img)

        loss.backward(retain_graph=False)

        opt.step()

        if display_progress:
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        if i % save_interval == 0 or i == niter - 1:
            img_pil = TF.to_pil_image(img[0])
            image_filepath = os.path.join(save_path, f"{i}{image_extension}")
            img_pil.save(image_filepath)

            if on_save_cb:
                on_save_cb(image_filepath)

    return image_filepath
