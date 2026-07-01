import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


@contextmanager
def svr_base_projector(model):
    """Temporarily force every GatedSVDLinear in ``model`` to its base weight.

    Used to produce the KD *teacher* outputs (= the base/previous model) during
    SVR Stage-2 training: with the gates forced to 0, the projector emits exactly
    W_base, so the model reproduces the base MEUSLI (base projector + frozen base
    LoRA). Restores normal (gated) behaviour on exit.
    """
    gated = [m for m in model.modules() if isinstance(m, GatedSVDLinear)]
    for m in gated:
        m.teacher_mode = True
    try:
        yield
    finally:
        for m in gated:
            m.teacher_mode = False


def svr_kd_loss(student_logits, teacher_logits, labels, temperature=1.0):
    """KL(teacher || student) over the supervised (label != -100) positions.

    Mirrors the cross-entropy masking/shift used by HF causal LMs so the KD term
    is computed on exactly the transcription tokens. Returns a scalar tensor
    (0.0 if no supervised positions are present).
    """
    # shift to align logit[i] with label[i+1], like HF CausalLM loss
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    lbl = labels[:, 1:]
    mask = lbl != -100
    if mask.sum() == 0:
        return student_logits.new_zeros(())
    s = s[mask]
    t = t[mask]
    log_p_student = F.log_softmax(s.float() / temperature, dim=-1)
    p_teacher = F.softmax(t.float() / temperature, dim=-1)
    kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
    return kd * (temperature * temperature)


class GatedSVDLinear(nn.Module):
    """Linear layer for Singular Value-based Rehearsal (SVR).

    Implements the Stage-2 mechanism of "Efficient Rehearsal for Continual
    Learning in ASR via Singular Value Tuning" (Vander Eeckt & Van hamme, 2026)
    for a single ``nn.Linear`` layer.

    Given the layer's weight before adaptation ``W_base`` (= W_{t-1}) and the
    weight after Stage-1 fine-tuning ``W_ft`` (= W~_t), the weight update
    ``dW = W_ft - W_base`` is decomposed via SVD: ``dW = U diag(s) Vh``.
    The effective weight used at run time is

        W = W_base + U diag(sigmoid(alpha) * s) Vh

    Only ``alpha`` (one scalar gate per singular value) is trainable; everything
    else is stored as a frozen buffer. ``alpha`` is initialised to a negative
    value so that ``sigmoid(alpha) ~= 0`` and therefore ``W ~= W_base`` at the
    start of Stage-2 training, preserving performance on previous tasks
    (paper, Sec. III-B).

    The layer can be constructed empty (correct buffer shapes inferred from
    ``in_features``/``out_features``) so that an SVR checkpoint can be reloaded
    with ``load_state_dict``; use :meth:`load_delta` to populate it from a pair
    of weight matrices.
    """

    def __init__(self, in_features, out_features, bias=True, alpha_init=-10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_init = alpha_init
        # when True, forward uses W_base (gates off) -> KD teacher / base model
        self.teacher_mode = False
        # rank of a full (thin) SVD of an (out_features x in_features) matrix
        k = min(in_features, out_features)
        self.rank = k

        # Frozen components (no gradients): base weight + thin SVD of the update.
        self.register_buffer("weight_base", torch.zeros(out_features, in_features))
        self.register_buffer("U", torch.zeros(out_features, k))
        self.register_buffer("s", torch.zeros(k))
        self.register_buffer("Vh", torch.zeros(k, in_features))

        # The only trainable parameter: one gate per singular value.
        self.alpha = nn.Parameter(torch.full((k,), float(alpha_init)))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def load_delta(self, w_base, w_ft, b_base=None, b_ft=None):
        """Populate the layer from base / fine-tuned weights (and biases).

        ``w_base`` and ``w_ft`` are ``(out_features, in_features)`` tensors.
        The bias (a non-linear-layer parameter in the paper's terminology) is
        averaged between the two models and frozen, following Eq. (7).
        """
        w_base = w_base.to(torch.float32)
        w_ft = w_ft.to(torch.float32)
        delta = w_ft - w_base
        U, s, Vh = torch.linalg.svd(delta, full_matrices=False)

        self.weight_base.copy_(w_base)
        self.U.copy_(U)
        self.s.copy_(s)
        self.Vh.copy_(Vh)
        # reset gates to (near) zero so the layer starts at W_base
        self.alpha.data.fill_(float(self.alpha_init))

        if self.bias is not None and b_base is not None:
            if b_ft is not None:
                self.bias.copy_(((b_base.float() + b_ft.float()) / 2.0))
            else:
                self.bias.copy_(b_base.float())

    def effective_weight(self):
        if self.teacher_mode:
            # gates off -> base/previous model weight (KD teacher)
            return self.weight_base
        gated_s = torch.sigmoid(self.alpha) * self.s
        # (out, k) * (k,) -> (out, k), then @ (k, in) -> (out, in)
        delta_w = (self.U * gated_s) @ self.Vh
        return self.weight_base + delta_w

    def forward(self, x):
        weight = self.effective_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def export_linear(self):
        """Return a plain ``nn.Linear`` whose weight folds in the learned gates.

        Useful for inference/evaluation: the result is mathematically identical
        but drops the SVD buffers, so it can be loaded by the standard
        ``EncoderProjectorConcat`` (``encoder_projector: linear``) path.
        """
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.copy_(self.effective_weight())
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderProjectorConcatSVR(nn.Module):
    """SVR variant of :class:`EncoderProjectorConcat`.

    Architecturally identical to the MEUSLI linear projector, but the two
    linear layers are :class:`GatedSVDLinear` modules. After Stage-1
    fine-tuning, build this projector from the base (W_{t-1}) and fine-tuned
    (W~_t) projector checkpoints with :meth:`build_from_state_dicts`; then train
    only the gating vectors ``alpha`` with rehearsal data (Stage 2).
    """

    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        alpha_init = float(config.get("svr_alpha_init", -10.0))
        self.linear1 = GatedSVDLinear(self.encoder_dim * self.k, 2048, alpha_init=alpha_init)
        self.relu = nn.ReLU()
        self.linear2 = GatedSVDLinear(2048, config.llm_dim, alpha_init=alpha_init)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    @torch.no_grad()
    def build_from_state_dicts(self, base_sd, ft_sd, prefix="encoder_projector."):
        """Initialise the gated layers from base and fine-tuned projector ckpts.

        ``base_sd`` / ``ft_sd`` are state dicts (as saved by SLAM-LLM, i.e. keyed
        ``encoder_projector.linear1.weight`` ...). ``prefix`` is stripped when
        looking up keys; pass ``""`` for bare ``linear1.weight`` keys.
        """
        def get(sd, name):
            for key in (prefix + name, name):
                if key in sd:
                    return sd[key]
            raise KeyError(f"could not find '{name}' (prefix '{prefix}') in state dict")

        self.linear1.load_delta(
            get(base_sd, "linear1.weight"), get(ft_sd, "linear1.weight"),
            get(base_sd, "linear1.bias"), get(ft_sd, "linear1.bias"),
        )
        self.linear2.load_delta(
            get(base_sd, "linear2.weight"), get(ft_sd, "linear2.weight"),
            get(base_sd, "linear2.bias"), get(ft_sd, "linear2.bias"),
        )
        return self

    @torch.no_grad()
    def export_dense_state_dict(self, prefix="encoder_projector."):
        """Fold the learned gates into a plain-linear projector state dict.

        The returned dict matches :class:`EncoderProjectorConcat`, so it can be
        evaluated/served through the standard ``encoder_projector: linear`` path.
        """
        l1 = self.linear1.export_linear()
        l2 = self.linear2.export_linear()
        return {
            prefix + "linear1.weight": l1.weight.detach().cpu(),
            prefix + "linear1.bias": l1.bias.detach().cpu(),
            prefix + "linear2.weight": l2.weight.detach().cpu(),
            prefix + "linear2.bias": l2.bias.detach().cpu(),
        }


class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj