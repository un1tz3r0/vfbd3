"""
Shared self-attention ("reference-only", KV injection) for UNet models (SDXL): the technique
behind controlnetvideo's `refxl` mode, extended to several reference images at once.

At every denoising step the reference images, noised to the current level, go through the
UNet in one batched pass that only records each self-attention layer's input. In the real
pass each self-attention layer then attends over its own tokens *and* every reference's.
A reference's strength is added to the attention scores of its tokens as log(strength):
0 switches it off, 1 is plain shared attention, above 1 favours it over the frame itself.
Optionally (AdaIN) each ResNet block's per-channel mean and spread is matched to the first
reference's, which holds colour and contrast.

With `record=True` it also returns where each output location drew from in the last
reference: the attention-weighted mean key position (a dense correspondence field, as in
DIFT) and how concentrated that attention was. It is read from the coarsest attention layers
on the last step, where the field is cheapest and most semantic.
"""

import math

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.resnet import ResnetBlock2D


class _State:
	def __init__(self):
		self.mode = None        # None, "write" (reference pass) or "read" (real pass)
		self.log_scales = []    # per reference
		self.adain = 0.0
		self.record = False
		self.bank, self.stats, self.i, self.j = [], [], 0, 0
		self.min_tokens = None
		self.fields = []        # recorded (mean position, spread) per coarse layer


class _SharedAttention:
	def __init__(self, state):
		self.state, self.base = state, AttnProcessor2_0()

	def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
		state = self.state
		if state.mode == "write":
			state.bank.append(hidden_states.detach())
			n = hidden_states.shape[1]
			state.min_tokens = n if state.min_tokens is None else min(state.min_tokens, n)
		if state.mode != "read":
			return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

		batch, n, _ = hidden_states.shape
		refs = state.bank[state.i].chunk(len(state.log_scales))
		state.i += 1
		context = torch.cat([hidden_states, *refs], dim=1)
		bias = torch.cat([hidden_states.new_zeros(n)] + [
			hidden_states.new_full((r.shape[1],), s) for r, s in zip(refs, state.log_scales)])
		mask = bias.view(1, 1, -1).expand(batch, 1, -1)
		if state.record and n == state.min_tokens:
			self._record(attn, hidden_states[-1:], refs[-1][-1:])
		return self.base(attn, hidden_states, context, mask, **kwargs)

	def _record(self, attn, query_states, ref):
		''' how the (conditional) frame's tokens would attend over the last reference's tokens
		alone, whatever that reference's strength: a correspondence, not what was injected. '''
		n, m, heads = query_states.shape[1], ref.shape[1], attn.heads
		q = attn.to_q(query_states).view(1, n, heads, -1).transpose(1, 2)
		k = attn.to_k(ref).view(1, m, heads, -1).transpose(1, 2)
		self.state.fields.append(torch.softmax((q @ k.transpose(-1, -2)).float() * attn.scale, dim=-1).mean(dim=1)[0])


class Reference:
	''' installs itself on `unet`; wrap a pipeline call in `active(...)`. '''

	def __init__(self, unet):
		self.unet = unet
		self.state = _State()
		unet.set_attn_processor({
			name: _SharedAttention(self.state) if name.endswith("attn1.processor") else AttnProcessor2_0()
			for name in unet.attn_processors})
		for module in unet.modules():
			if isinstance(module, ResnetBlock2D):
				module.register_forward_hook(self._adain)

	def _adain(self, module, inputs, output):
		state = self.state
		if state.mode is None or state.adain <= 0:
			return output
		if state.mode == "write":
			# statistics of the first reference only
			first = output[:output.shape[0] // len(state.log_scales)].float()
			var, mean = torch.var_mean(first, dim=(2, 3), keepdim=True)
			state.stats.append((mean, var))
			return output
		r_mean, r_var = state.stats[state.j]
		state.j += 1
		x = output.float()
		var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
		matched = (x - mean) * ((r_var + 1e-5) / (var + 1e-5)).sqrt() + r_mean
		return (x + (matched - x) * state.adain).to(output.dtype)

	def active(self, scheduler, references, noise, adain=0.0, record=False, steps=1):
		''' context manager. `references`: [(latents, strength), ...]. While inside, every UNet
		call is preceded by a reference pass with those latents noised by `noise` to the current
		timestep. With `record`, the correspondence to the last reference is gathered on the
		final UNet call; read it from `.correspondence` afterwards. '''
		unet, state = self.unet, self.state
		original = unet.forward
		patched = "forward" in vars(unet)   # accelerate's offload hooks patch forward per instance
		latents = torch.cat([r for r, _ in references])
		calls = [0]
		outer = self

		def forward(sample, timestep, *args, **kwargs):
			t = timestep.reshape(-1)[:1] if torch.is_tensor(timestep) else torch.tensor([timestep])
			ref = scheduler.scale_model_input(scheduler.add_noise(latents, noise.expand_as(latents), t.to(latents.device)), timestep)
			batch = sample.shape[0]
			ref = ref.to(sample.dtype).repeat_interleave(batch, dim=0)
			plain = {k: (v.repeat(len(references), *[1] * (v.dim() - 1)) if torch.is_tensor(v) and v.shape[:1] == (batch,) else v)
				for k, v in kwargs.items() if k not in ("down_block_additional_residuals", "mid_block_additional_residual")}
			if "added_cond_kwargs" in plain:
				plain["added_cond_kwargs"] = {k: v.repeat(len(references), *[1] * (v.dim() - 1)) for k, v in plain["added_cond_kwargs"].items()}
			hidden = [args[0].repeat(len(references), 1, 1)] if args else []
			state.bank.clear(), state.stats.clear()
			state.i = state.j = 0
			calls[0] += 1
			state.record = record and calls[0] >= steps
			try:
				state.mode = "write"
				original(ref, timestep, *hidden, **plain)
				state.mode = "read"
				return original(sample, timestep, *args, **kwargs)
			finally:
				state.mode = None

		class Active:
			def __enter__(self):
				state.log_scales = [math.log(s) if s > 0 else -math.inf for _, s in references]
				state.adain, state.fields = adain, []
				outer.correspondence = None
				unet.forward = forward

			def __exit__(self, *exc):
				if patched:
					unet.forward = original
				else:
					del unet.forward
				state.bank.clear(), state.stats.clear()
				if state.fields:
					outer.correspondence = outer._field(torch.stack(state.fields).mean(0), latents.shape[-2:])
				state.fields, state.record = [], False

		return Active()

	@staticmethod
	def _field(weights, latent_hw):
		''' (queries, keys) attention -> (h, w, 2) mean key position and (h, w) spread, in
		latent-pixel units, on the coarse token grid. '''
		n = weights.shape[0]
		lh, lw = latent_hw
		f = math.sqrt(lh * lw / n)
		h, w = round(lh / f), round(lw / f)
		ys, xs = torch.meshgrid(torch.arange(h, device=weights.device), torch.arange(w, device=weights.device), indexing="ij")
		pos = torch.stack([xs, ys], -1).reshape(-1, 2).float()
		mean = weights @ pos
		spread = (weights * ((pos[None] - mean[:, None]) ** 2).sum(-1)).sum(-1).sqrt()
		scale = lw / w
		return ((mean.reshape(h, w, 2) + 0.5) * scale).cpu().numpy(), (spread.reshape(h, w) * scale).cpu().numpy()
