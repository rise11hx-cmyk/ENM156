"""
The spiking networks. Like denoiser.nets they map input frames
(..., T, n_in) to output frames (..., T, n_out) and never look ahead, but
they remember the past in the state of their neurons instead of looking at
a window of past frames: `context` is 1 and the state is carried along.

forward() runs a whole clip from rest; step() runs one frame given the
state, for live use. Both give the same output (tests/test_snn_stream.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets import ACTIVATIONS, FrameModel
from .neurons import LIF, Leaky


class SpikingModel(FrameModel):
    """What every spiking network offers. After forward():
      spike_rate  mean spikes per neuron per time step, with gradient, so a
                  loss can ask for fewer spikes
      stats       {"frames": frames run, "spikes": [spikes of each layer]},
                  summed over the batch, for the cost estimate (snn.stats)"""
    context = 1
    spike_rate = None
    stats = None

    def init_state(self, batch_shape, device):
        """The state of a network at rest, for inputs of (*batch_shape, n_in)."""
        raise NotImplementedError

    def step(self, x, state):
        """One frame x (..., n_in) -> (output frame (..., n_out), new state)."""
        raise NotImplementedError

    def count(self, spikes, n_frames):
        """Sets spike_rate and stats from each layer's spikes of a forward pass."""
        counts = torch.stack([s.sum() for s in spikes])
        self.spike_rate = counts.sum() / (n_frames * self.steps_per_frame * sum(self.hidden))
        self.stats = {"frames": n_frames, "spikes": counts.detach().tolist()}


class LIFNet(SpikingModel):
    """Layers of LIF neurons, each fed by the spikes of the one before
    (and, if `recurrent`, by its own spikes of the step before), then a
    leaky readout layer that doesn't spike: its membrane potentials are the
    output. The first layer gets the input frame as current. Every frame
    is `steps_per_frame` time steps with the same input; the output is read
    after the last one.

    `input_mask` (first-layer weights that exist: 1, pruned away: 0) lets
    input connections be removed; see prune.py."""

    def __init__(self, n_in, n_out, hidden=(256, 256, 256), recurrent=False, decay=0.9,
                 readout_decay=0.5, learn_decay=True, threshold=1.0, slope=25.0,
                 steps_per_frame=1, out_act=None):
        super().__init__()
        self.n_in, self.n_out, self.hidden = n_in, n_out, tuple(hidden)
        self.steps_per_frame = steps_per_frame
        sizes = (n_in, *hidden)
        self.inputs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.recurrent = nn.ModuleList(nn.Linear(h, h, bias=False) for h in hidden) if recurrent else None
        self.neurons = nn.ModuleList(LIF(h, decay, threshold, slope, learn_decay) for h in hidden)
        self.out = nn.Linear(hidden[-1], n_out, bias=False)
        self.readout = Leaky(n_out, readout_decay, learn_decay)
        self.bias = nn.Parameter(torch.zeros(n_out))  # of the output, added after the readout
        self.out_act = ACTIVATIONS[out_act]()
        self.register_buffer("input_mask", torch.ones(hidden[0], n_in))

    def load_state_dict(self, state, *args, **kwargs):
        # checkpoints from before input_mask: nothing pruned
        return super().load_state_dict({"input_mask": self.input_mask, **state}, *args, **kwargs)

    def first(self, x):
        """The input frame as current into the first layer, without the pruned weights."""
        return F.linear(x, self.inputs[0].weight * self.input_mask, self.inputs[0].bias)

    def init_state(self, batch_shape, device):
        zeros = lambda n: torch.zeros(*batch_shape, n, device=device)
        return {"v": [zeros(h) for h in self.hidden],  # membrane potentials
                "s": [zeros(h) for h in self.hidden],  # spikes of the last step
                "u": zeros(self.n_out)}                # readout potentials

    def step(self, x, state):
        v, s, u = list(state["v"]), list(state["s"]), state["u"]
        first = self.first(x)  # the same in every time step of the frame
        for _ in range(self.steps_per_frame):
            below = None
            for i, lif in enumerate(self.neurons):
                current = first if i == 0 else self.inputs[i](below)
                if self.recurrent is not None:
                    current = current + self.recurrent[i](s[i])
                s[i], v[i] = lif.step(current, v[i])
                below = s[i]
            y, u = self.readout.step(self.out(below), u)
        return self.out_act(y + self.bias), {"v": v, "s": s, "u": u}

    def forward_valid(self, x):
        """A layer at a time over the whole clip, which is the same as a
        frame at a time with step(), but much faster."""
        lead_shape, (n_frames, n_in) = x.shape[:-2], x.shape[-2:]
        steps = self.steps_per_frame
        currents = self.first(x.reshape(-1, n_frames, n_in)).repeat_interleave(steps, dim=1)
        spikes = []
        for i, lif in enumerate(self.neurons):
            if i:
                currents = self.inputs[i](spikes[-1])
            spikes.append(lif.run(currents, self.recurrent[i] if self.recurrent is not None else None))
        self.count(spikes, x[..., 0].numel())
        y = self.readout.run(self.out(spikes[-1]))[:, steps - 1::steps]  # after each frame's last step
        return self.out_act(y + self.bias).reshape(*lead_shape, n_frames, self.n_out)
