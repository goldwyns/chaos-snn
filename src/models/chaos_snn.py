import torch
import torch.nn as nn
import torch.nn.functional as F


class ChaosModulator(nn.Module):
    """
    Chaos modulation layer:
    x: (batch, C, T) -> u: (batch, C, T)

    z_{t+1} = (1-gamma) z_t + gamma * sigma(alpha z_t (1-z_t) + beta x_t)
    u_t = lambda * x_t + (1-lambda) * (2 z_t - 1)
    """

    def __init__(self, channels, alpha=3.5, beta=0.5, gamma=0.5, lam=0.5):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam

    def forward(self, x):
        """
        x: (batch, C, T)
        returns u: (batch, C, T)
        """
        b, c, t = x.shape
        assert c == self.channels, "channel mismatch in ChaosModulator"

        device = x.device
        # initialize z_0 in (0,1)
        z = torch.rand(b, c, device=device) * 0.5 + 0.25  # (b, C)
        u_out = []

        for n in range(t):
            x_t = x[:, :, n]  # (b, C)
            # sigma: bounded nonlinearity
            sigma_in = self.alpha * z * (1 - z) + self.beta * x_t
            z_next = (1 - self.gamma) * z + self.gamma * 0.5 * (1.0 + torch.tanh(sigma_in))
            z_next = torch.clamp(z_next, 0.0, 1.0)
            z_tilde = 2 * z_next - 1.0
            u_t = self.lam * x_t + (1.0 - self.lam) * z_tilde
            u_out.append(u_t.unsqueeze(-1))
            z = z_next

        u = torch.cat(u_out, dim=-1)  # (b, C, T)
        return u


class ThresholdRateEncoder(nn.Module):
    """
    Simple threshold-rate spike encoder with fixed threshold.
    u: (batch, C, T) -> spikes: (batch, C, T) in {0,1}
    """

    def __init__(self, threshold=0.0, leak=0.9, target_rate=None):
        super().__init__()
        self.threshold = threshold
        self.leak = leak
        self.target_rate = target_rate  # placeholder if we later add homeostasis

    def forward(self, u):
        """
        u: (b, C, T)
        returns spikes: (b, C, T)
        """
        b, c, t = u.shape
        device = u.device

        v = torch.zeros(b, c, device=device)
        spikes = []

        for n in range(t):
            u_t = u[:, :, n]
            v = self.leak * v + u_t
            s_t = (v > self.threshold).float()
            # soft reset
            v = v - s_t
            spikes.append(s_t.unsqueeze(-1))

        s = torch.cat(spikes, dim=-1)
        return s


class SurrogateHeaviside(torch.autograd.Function):
    """
    Heaviside step with triangular surrogate gradient.
    """

    @staticmethod
    def forward(ctx, input, v_th=1.0, width=0.5):
        ctx.save_for_backward(input)
        ctx.v_th = v_th
        ctx.width = width
        out = (input > v_th).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        v_th = ctx.v_th
        width = ctx.width

        x = (input - v_th) / width
        mask = (x.abs() <= 1.0).float()
        grad_input = grad_output * (1.0 - x.abs()) * mask
        return grad_input, None, None


surrogate_heaviside = SurrogateHeaviside.apply


class RecurrentLIFReservoir(nn.Module):
    """
    One-layer recurrent LIF reservoir with chaos-controlled recurrent matrix.
    Input spikes: (b, C_in, T)
    Output spikes: (b, N, T)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        tau_mem=20e-3,
        tau_syn=10e-3,
        dt=1e-3,
        v_th=1.0,
        alpha_rec=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.v_th = v_th

        # time constants -> discrete decays
        self.alpha_v = torch.exp(torch.tensor(-dt / tau_mem))
        self.alpha_s = torch.exp(torch.tensor(-dt / tau_syn))

        # weights
        self.w_in = nn.Linear(input_size, hidden_size, bias=False)
        self.w_rec_raw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.orthogonal_(self.w_rec_raw)

        # scale for chaos regime
        self.alpha_rec = nn.Parameter(torch.tensor(alpha_rec), requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, s_in):
        """
        s_in: (b, C_in, T)
        returns:
          spikes: (b, hidden, T)
          v_traj: (b, hidden, T) membrane potentials
        """
        b, c, t = s_in.shape
        device = s_in.device

        v = torch.zeros(b, self.hidden_size, device=device)
        I = torch.zeros(b, self.hidden_size, device=device)
        w_rec = self.alpha_rec * self.w_rec_raw  # (hidden, hidden)

        spikes = []
        v_traj = []

        for n in range(t):
            s_t = s_in[:, :, n]  # (b, C)
            I = self.alpha_s * I + self.w_in(s_t) + F.linear(spikes[-1].squeeze(-1), w_rec) if spikes else self.w_in(s_t)
            v = self.alpha_v * v + I + self.bias
            s_out = surrogate_heaviside(v, self.v_th, 0.5)
            v = v - s_out * self.v_th
            spikes.append(s_out.unsqueeze(-1))
            v_traj.append(v.unsqueeze(-1))

        spikes = torch.cat(spikes, dim=-1)   # (b, hidden, T)
        v_traj = torch.cat(v_traj, dim=-1)   # (b, hidden, T)
        return spikes, v_traj


class ChaosSNNSeizureDetector(nn.Module):
    """
    Full pipeline (without filtering):
      x -> chaos modulator -> spike encoder -> reservoir -> readout -> probability
    """

    def __init__(
        self,
        n_channels,
        n_hidden=128,
        alpha=3.5,
        beta=0.5,
        gamma=0.5,
        lam=0.5,
        v_th=1.0,
        alpha_rec=1.0,
        dt=1e-3,          
        tau_mem=20e-3,
        tau_syn=10e-3,
    ):
        super().__init__()
        self.n_channels = n_channels

        self.chaos = ChaosModulator(
            channels=n_channels,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lam=lam,
        )

        self.encoder = ThresholdRateEncoder(
            threshold=0.0,
            leak=0.9,
        )

        self.reservoir = RecurrentLIFReservoir(
            input_size=n_channels,
            hidden_size=n_hidden,
            v_th=v_th,
            alpha_rec=alpha_rec,
            dt=dt,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
        )


        # readout on spike counts
        self.readout = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """
        x: (b, C, T) preprocessed EEG window
        returns:
          p: (b,) seizure probability
          extra: dict of internal states
        """
        # chaos modulation
        u = self.chaos(x)             # (b, C, T)
        # spike encoding
        s_in = self.encoder(u)        # (b, C, T)
        # reservoir
        s_res, v_res = self.reservoir(s_in)  # (b, H, T)

        # temporal pooling: spike counts
        r = s_res.mean(dim=-1)         # (b, H)
        logits = self.readout(r).squeeze(-1)  # (b,)
        p = torch.sigmoid(logits)

        extra = {
            "u": u,
            "s_in": s_in,
            "s_res": s_res,
            "v_res": v_res,
            "r": r,
            "logits": logits,
        }
        return p, extra
