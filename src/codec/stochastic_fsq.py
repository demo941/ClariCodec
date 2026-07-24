"""
Stochastic Finite Scalar Quantization (S-FSQ)
Adapted for RL-based Audio Codec optimization.
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn import Module
from torch import tensor, int32
from torch.amp import autocast
from einops import rearrange, pack, unpack
import torch.nn.functional as F

# --- Helper Functions ---

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# --- Rounding & Sampling Logic ---

def round_ste(z):
    """ Standard deterministic rounding with straight-through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ Floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

def gumbel_softmax(
    logits,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
):
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )
    scaled_logits = logits / tau
    gumbels = scaled_logits + gumbels
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        selected_log_prob = F.log_softmax(scaled_logits, dim=dim).gather(dim, index).squeeze(dim)
    else:
        # Reparametrization trick.
        ret = y_soft
        selected_log_prob = None
    return ret, selected_log_prob

class FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
        # --- New Parameters for Stochasticity ---
        temperature: float = 1.0, 
        stochastic: bool = False  
    ):
        super().__init__()

        # Compatibility checks
        assert not (any([l == 2 for l in levels]) and not preserve_symmetry), 'turn on `preserve_symmetry` for levels == 2'

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections
        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

        # --- Stochastic Configuration ---
        self.temperature = temperature
        self.stochastic = stochastic
        self.saved_log_probs = None # To store log probs for RL update
        self.steps = 0

    def set_temperature(self, t):
        self.temperature = t
        
    def set_stochastic(self, active: bool):
        self.stochastic = active
        
    def _discretize(self, z): #for preserve_symetric=False
        """
        Input: z (continuous, roughly in integer range e.g., -2.3, 1.1)
        Output: z_q (discrete integer values e.g., -2.0, 1.0)
        """
        if not self.stochastic:
            self.saved_log_probs = None
            return floor_ste(z + 0.5)

        else:
            #grid
            max_level = self._levels.max()
            if self.preserve_symmetry:
                grid = torch.arange(0, max_level, device=z.device).float()
            else:
                max_grid = (max_level - 1) // 2
                min_grid = - (max_level // 2)
                grid = torch.arange(min_grid, max_grid+1, device=z.device).float()
            
            z_expanded = z.unsqueeze(-1) # [B, ..., D, 1]
            grid_expanded = grid.view(*([1] * z.ndim), -1) # [1, ..., 1, G]
            dists = (z_expanded - grid_expanded).pow(2)
            logits = -dists * 5

            if self.preserve_symmetry:
                upper_bound = (self._levels - 1).view(1, 1, -1, 1) # [1, 1, D, 1]
                lower_bound = torch.zeros_like(upper_bound)
            else:
                upper_bound = ((self._levels - 1) // 2).view(1, 1, -1, 1)
                lower_bound = (-(self._levels // 2)).view(1, 1, -1, 1)
            
            mask = (grid_expanded >= lower_bound) & (grid_expanded <= upper_bound)
            logits = logits.masked_fill(~mask, -1e9)

            y_one_hot, self.saved_log_probs = gumbel_softmax(
                logits, 
                tau=self.temperature, 
                hard=True, 
                dim=-1
            )
        
            z_q = (y_one_hot * grid_expanded).sum(dim=-1)
            
            return z + (z_q - z).detach()
            # return z_q

    # --- Bounds ---

    def bound(self, z, eps = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        bounded_z = (z + shift).tanh() * half_l - offset
        # Discretize (Round or Stochastic Sample)
        quantized_z = self._discretize(bounded_z)
        half_width = self._levels // 2
        return quantized_z / half_width

    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """

        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        
        # Transform to integer-like space where floor() creates valid levels
        bracket = levels_minus_1 * (z.tanh() + 1) / 2.
        
        # Discretize (Floor or Stochastic Sample)
        # Note: _discretize handles the stochastic logic. 
        # If deterministic, we need floor logic for symmetry method.
        # This is handled inside _discretize checking self.preserve_symmetry
        quantized_bracket = self._discretize(bracket)
        
        return scale * quantized_bracket - 1.

    # --- Quantize ---

    def quantize(self, z):
        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        # Perform Bounding + Discretization (Stochastic or Deterministic)
        bounded_z = bound_fn(z)

        # Standard noise dropout (orthogonal to stochastic quantization)
        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    # --- Helpers for Conversion ---

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def forward(self, z):
        """
        Returns: 
          out: Quantized embedding
          indices: Indices
        
        *Access `self.saved_log_probs` after forward pass if using RL.*
        """
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            # --- Quantization Happens Here (Stochastic or Deterministic) ---
            codes = self.quantize(z)

            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.to(orig_dtype)

        out = self.project_out(codes)

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        return out, indices


class IFSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
        # --- New Parameters for Stochasticity ---
        temperature: float = 1.0, 
        stochastic: bool = False  
    ):
        super().__init__()

        # Compatibility checks
        assert not (any([l == 2 for l in levels]) and not preserve_symmetry), 'turn on `preserve_symmetry` for levels == 2'

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections
        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

        # --- Stochastic Configuration ---
        self.temperature = temperature
        self.stochastic = stochastic
        self.saved_log_probs = None # To store log probs for RL update
        self.steps = 0

    def set_temperature(self, t):
        self.temperature = t
        
    def set_stochastic(self, active: bool):
        self.stochastic = active
        
    def _discretize(self, z): #for preserve_symetric=False
        """
        Input: z (continuous, roughly in integer range e.g., -2.3, 1.1)
        Output: z_q (discrete integer values e.g., -2.0, 1.0)
        """
        if not self.stochastic:
            self.saved_log_probs = None
            return floor_ste(z + 0.5)

        else:
            #grid
            max_level = self._levels.max()
            if self.preserve_symmetry:
                grid = torch.arange(0, max_level, device=z.device).float()
            else:
                max_grid = (max_level - 1) // 2
                min_grid = - (max_level // 2)
                grid = torch.arange(min_grid, max_grid+1, device=z.device).float()
            
            z_expanded = z.unsqueeze(-1) # [B, ..., D, 1]
            grid_expanded = grid.view(*([1] * z.ndim), -1) # [1, ..., 1, G]
            dists = (z_expanded - grid_expanded).pow(2)
            logits = -dists * 5

            if self.preserve_symmetry:
                upper_bound = (self._levels - 1).view(1, 1, -1, 1) # [1, 1, D, 1]
                lower_bound = torch.zeros_like(upper_bound)
            else:
                upper_bound = ((self._levels - 1) // 2).view(1, 1, -1, 1)
                lower_bound = (-(self._levels // 2)).view(1, 1, -1, 1)
            
            mask = (grid_expanded >= lower_bound) & (grid_expanded <= upper_bound)
            logits = logits.masked_fill(~mask, -1e9)

            y_one_hot, self.saved_log_probs = gumbel_softmax(
                logits, 
                tau=self.temperature, 
                hard=True, 
                dim=-1
            )
        
            z_q = (y_one_hot * grid_expanded).sum(dim=-1)
            
            return z + (z_q - z).detach()
            # return z_q

    # --- Bounds ---

    def bound(self, z, eps = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        # shift = (offset / half_l).atanh()
        # shift = 1.25 * (offset / half_l).atanh()
        y0 = offset / half_l
        shift = (1.0 / 1.6) * torch.log((1 + y0) / (1 - y0))
        # bounded_z = (z + shift).tanh() * half_l - offset
        active_z = 2.0 * torch.sigmoid(1.6 * (z + shift)) - 1.0
        bounded_z = active_z * half_l - offset
        # Discretize (Round or Stochastic Sample)
        quantized_z = self._discretize(bounded_z)
        half_width = self._levels // 2
        return quantized_z / half_width

    def symmetry_preserving_bound(self, z):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """

        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        
        # Transform to integer-like space where floor() creates valid levels
        bracket = levels_minus_1 * (z.tanh() + 1) / 2.
        
        # Discretize (Floor or Stochastic Sample)
        # Note: _discretize handles the stochastic logic. 
        # If deterministic, we need floor logic for symmetry method.
        # This is handled inside _discretize checking self.preserve_symmetry
        quantized_bracket = self._discretize(bracket)
        
        return scale * quantized_bracket - 1.

    # --- Quantize ---

    def quantize(self, z):
        shape, device, noise_dropout, preserve_symmetry = z.shape[0], z.device, self.noise_dropout, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        # Perform Bounding + Discretization (Stochastic or Deterministic)
        bounded_z = bound_fn(z)

        # Standard noise dropout (orthogonal to stochastic quantization)
        if not self.training or noise_dropout == 0.:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    # --- Helpers for Conversion ---

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def forward(self, z):
        """
        Returns: 
          out: Quantized embedding
          indices: Indices
        
        *Access `self.saved_log_probs` after forward pass if using RL.*
        """
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            # --- Quantization Happens Here (Stochastic or Deterministic) ---
            codes = self.quantize(z)

            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.to(orig_dtype)

        out = self.project_out(codes)

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        return out, indices


class RFSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_quantizers: int = 2,
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
        temperature: float = 1.0,
        stochastic: bool = False,
    ):
        super().__init__()
        assert num_quantizers >= 1, 'num_quantizers must be at least 1'

        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([
            IFSQ(
                levels=levels,
                dim=dim,
                channel_first=channel_first,
                projection_has_bias=projection_has_bias,
                return_indices=return_indices,
                force_quantization_f32=force_quantization_f32,
                preserve_symmetry=preserve_symmetry,
                noise_dropout=noise_dropout,
                temperature=temperature,
                stochastic=stochastic,
            )
            for _ in range(num_quantizers)
        ])
        self.saved_log_probs = None

    @property
    def codebook_size(self):
        return self.layers[0].codebook_size

    def set_temperature(self, temperature: float):
        for layer in self.layers:
            layer.set_temperature(temperature)

    def set_stochastic(self, active: bool):
        for layer in self.layers:
            layer.set_stochastic(active)

    def forward(self, x):
        quantized_out = torch.zeros_like(x)
        residual = x
        all_indices = []
        all_log_probs = []

        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if exists(indices):
                all_indices.append(indices)

            if exists(layer.saved_log_probs):
                all_log_probs.append(layer.saved_log_probs)

        indices = None
        if all_indices:
            indices = torch.stack(all_indices, dim=1)

        self.saved_log_probs = all_log_probs if all_log_probs else None
        return quantized_out, indices

# --- Example Usage for your RL Loop ---
if __name__ == "__main__":
    # 1. Setup
    fsq = FSQ(levels=[8, 5, 5, 5], dim=256, stochastic=True) # Start in stochastic mode
    fsq.set_temperature(1.0) # High temp for exploration

    # 2. Forward Pass (Action)
    dummy_input = torch.randn(1, 100, 256) # [Batch, Time, Dim]
    quantized_output, indices = fsq(dummy_input)

    # 3. RL Update Calculation
    # Assume you got a Reward from the ASR (e.g., Reward = -WER)
    reward = -0.2 # Example WER of 20%
    
    # Retrieve Log Probs: shape [Batch, Time, NumCodebooks, Dim]
    log_probs = fsq.saved_log_probs 
    
    # Calculate Loss (REINFORCE)
    # loss = - (Reward * Sum_of_log_probs)
    # usually we sum over time and dimensions to get log_prob of the *sequence*
    rl_loss = - (reward * log_probs.sum())
    
    print(f"Quantized shape: {quantized_output.shape}")
    print(f"RL Loss: {rl_loss.item()}")
    
    # 4. Annealing (Later in training)
    fsq.set_temperature(0.5) 
    
    # 5. Switch to Inference (Deterministic)
    fsq.set_stochastic(False)
    out_det, _ = fsq(dummy_input)
    print("Switched to deterministic mode.")
