"""
Removes input connections of a trained spiking model: the first layer is the
only one that multiplies (snn.stats), so it is where most of the estimated
energy goes. The pruned weights stay away during fine-tuning (LIFNet.input_mask).

A weight's importance is its size times how much its input varies (standard
deviation over each 28th validation clip). --by picks what is removed:
  weights   the least important single weights
  inputs    whole inputs, the ones with the least important weights overall

  .venv/bin/python prune.py runs/snn_full/model.pt runs/pruned/start.pt --keep 0.25
  .venv/bin/python train_pesq.py --set pesq.init_from=runs/pruned/start.pt --set train.epochs=5 \
      --set train.output_dir=runs/pruned     # fine-tune it
"""
import argparse
from pathlib import Path

import torch

from denoiser.config import add_config_args, get_device, load_config
from denoiser.data import load_pairs, split_files
from denoiser.model import load_checkpoint, save_checkpoint
from denoiser.pipeline import features
from denoiser.snn.nets import LIFNet


def input_std(cfg, device):
    """Standard deviation of each model input over the validation clips."""
    clips = load_pairs(split_files(cfg, "val"), 0, cfg["data"]["sample_rate"], every=28)
    frames = torch.cat([features(torch.from_numpy(noisy).to(device), cfg)[1] for noisy, _ in clips])
    return frames.std(dim=0)


def prune_mask(importance, keep, by):
    """1 for the `keep` share of weights (or inputs) with the highest importance."""
    scores = importance if by == "weights" else importance.pow(2).sum(dim=0)
    n_keep = max(1, round(keep * scores.numel()))
    kept = torch.zeros_like(scores).flatten()
    kept[scores.flatten().topk(n_keep).indices] = 1
    return kept.reshape(scores.shape).expand_as(importance).clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Checkpoint of a spiking model.")
    ap.add_argument("out", help="Where to save the pruned checkpoint.")
    ap.add_argument("--keep", type=float, required=True, help="Share of the input weights to keep, e.g. 0.25.")
    ap.add_argument("--by", choices=("weights", "inputs"), default="weights")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    device = get_device(cfg)
    model, cfg = load_checkpoint(args.model, cfg, device)
    if not isinstance(model, LIFNet):
        raise SystemExit("prune.py only handles model.type = \"snn\"")

    with torch.no_grad():
        importance = model.inputs[0].weight.abs() * input_std(cfg, device) * model.input_mask
        model.input_mask.copy_(prune_mask(importance, args.keep, args.by))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, cfg, args.out)
    mask = model.input_mask
    print(f"kept {int(mask.sum())} of {mask.numel()} input weights ({mask.mean():.1%}), "
          f"{int(mask.any(dim=0).sum())} of {mask.shape[1]} inputs still used; saved {args.out}")


if __name__ == "__main__":
    main()
