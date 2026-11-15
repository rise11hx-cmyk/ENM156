"""
Checks the spiking network like tests/test_stream.py checks the others:
live (streamed) denoising gives the same audio as the whole recording at
once, the output never depends on later input, and running a clip frame by
frame with step() equals forward(). Also that a checkpoint rebuilds the same
network. Untrained model and random audio, so no dataset is needed.

  .venv/bin/python tests/test_snn_stream.py
"""
import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.model import load_checkpoint, make_model, save_checkpoint  # noqa: E402
from denoiser.pipeline import enhance  # noqa: E402
from test_stream import streamed  # noqa: E402

STFT = {"n_fft": 256, "hop": 64, "long_factor": 0, "causal": True}
CASES = [  # [snn] settings changed from config.toml, and the output
    ({}, "complex_mask"),
    ({"recurrent": True}, "complex_mask"),
    ({"steps_per_frame": 3}, "complex_mask"),
    ({"recurrent": True, "steps_per_frame": 2, "hidden": [48]}, "mask"),
    ({"learn_decay": False, "decay": 0.0}, "magnitude"),
    ({"hidden": [48, 48, 48], "skip": True, "recurrent": True, "steps_per_frame": 2}, "complex_mask"),
    ({"hidden": [48, 48, 48], "skip": True, "norm": True, "init_gain": 3.0}, "complex_mask"),
]
SMALL = {"hidden": [64, 48], "threshold": 0.1}  # low threshold: an untrained network should spike


def main():
    torch.manual_seed(0)
    wav = 0.1 * torch.randn(16000)  # 1 s
    failed = False
    for snn, output in CASES:
        cfg = merge(copy.deepcopy(load_config()), {"stft": STFT, "model": {"type": "snn", "output": output},
                                                   "snn": {**SMALL, **snn}})
        model = make_model(cfg).eval()
        with torch.no_grad():
            offline = enhance(model, wav, cfg)
            rate = model.spike_rate.item()
            live_err = (streamed(model, cfg, wav) - offline).abs().max().item()

            # frame by frame with step() == forward(), for a batch of two
            x = torch.randn(2, 50, model.n_in)
            state, frames = model.init_state((2,), x.device), []
            for t in range(x.shape[1]):
                y, state = model.step(x[:, t], state)
                frames.append(y)
            step_err = (torch.stack(frames, 1) - model(x)).abs().max().item()

            # zeroing the input from `cut` on must not change the output before
            # cut - (n_fft - hop), as in test_stream.py
            cut = 8000
            zeroed = wav.clone()
            zeroed[cut:] = 0
            diff = enhance(model, zeroed, cfg) - offline
            causal_err = diff[:cut - (STFT["n_fft"] - STFT["hop"])].abs().max().item()

        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(model, cfg, Path(tmp) / "model.pt")
            loaded, loaded_cfg = load_checkpoint(Path(tmp) / "model.pt", load_config(), torch.device("cpu"))
            with torch.no_grad():
                load_err = (enhance(loaded.eval(), wav, loaded_cfg) - offline).abs().max().item()

        ok = max(live_err, step_err, load_err) < 1e-4 and causal_err < 1e-5 and 0 < rate < 1
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {cfg['snn']} {output}: stream vs offline {live_err:.1e}, "
              f"step vs forward {step_err:.1e}, looks ahead {causal_err:.1e}, "
              f"reloaded {load_err:.1e}, spike rate {rate:.2f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
