"""AHA put together: the vision part plus the hippocampal short-term memory.

This is the shared code that train.py and evaluate.py both use.

One evaluation step in the paper (Section 2.3) goes like this:
  1. Study: show a small set of drawings once. The short-term memory (STM)
     learns them. The vision part (VC) stays frozen.
  2. Recall: show new drawings. For each one, AHA should bring back the
     matching study drawing.
  3. Reset: wipe the STM, weights and optimizer state, before the next set.

Paper hyperparameters (Table 3) are the defaults in AHAConfig.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model import modules


@dataclass
class AHAConfig:
    image_size: int = 52
    vc_k: int = 4               # active filters per position at inference

    dg_units: int = 225         # PS(DG) and PC(CA3) size
    dg_k: int = 10              # active DG units per drawing
    dg_gamma: float = 0.95      # inhibition decay per drawing
    dg_knockout: float = 0.25   # share of DG input weights set to zero

    ca3_rule: str = "pinverse"
    ca3_gain: float = 2.7
    ca3_update: int = 20        # neurons updated per Hopfield step
    ca3_iters: int = 70         # Hopfield steps

    pr_hidden: int = 1000
    pr_lr: float = 0.01
    pr_l2: float = 2.5e-5
    pr_cue_gain: float = 10.0   # gamma in Appendix A.2

    pm_hidden: int = 100
    pm_lr: float = 0.01
    pm_l2: float = 4e-4

    study_steps: int = 60       # optimizer steps on the one study batch


def load_vc(checkpoint, interest_filter=True, device="cpu"):
    """Builds the inference VC and loads pretrained filters from a file."""
    vc = modules.EC(interest_filter=interest_filter)
    state = torch.load(checkpoint, map_location="cpu")
    vc.load_pretrained(state["state_dict"])
    return vc.to(device).eval()


class AHA():
    """The full model: a frozen VC(EC) feeding a resettable hippocampus.

    Data flow (paper Figure 4):

        study:  image -> VC -> DG -> CA3 stores DG pattern
                         VC -> PR learns to output the DG pattern
                               CA1 learns DG pattern -> image
        recall: image -> VC -> PR -> cue -> CA3 settles -> CA1 -> image

    study() and recall() return every internal state, so you can score
    classification at each stage the way the paper does.
    """

    def __init__(self, vc, config=None, device=torch.device("cpu")):
        self.cfg = config or AHAConfig()
        self.device = torch.device(device)
        self.vc = vc.to(self.device).eval()
        for p in self.vc.parameters():
            p.requires_grad_(False)

        # Size the STM from the VC instead of hard-coding it. The old train.py
        # said 27225 while the VC put out 20449.
        self.in_dim = self.vc.output_dim(self.cfg.image_size)

    def reset(self, seed=0):
        """Builds a fresh STM. Every run starts from new random weights."""
        cfg = self.cfg
        torch.manual_seed(seed)
        self.generator = torch.Generator().manual_seed(seed)

        self.dg = modules.DG(self.in_dim, cfg.dg_units, cfg.dg_k,
                             cfg.dg_gamma, cfg.dg_knockout).to(self.device)
        self.ca3 = modules.CA3(cfg.dg_units, cfg.ca3_gain, cfg.ca3_update, cfg.ca3_iters)
        self.pr = modules.ECToCA3(self.in_dim, cfg.dg_units, cfg.pr_hidden).to(self.device)
        self.pm = modules.CA1(cfg.dg_units, cfg.image_size**2, cfg.pm_hidden,
                              cfg.image_size).to(self.device)

        # Adam's weight_decay adds an L2 penalty to the gradient, which is the
        # paper's L2 regularisation.
        self.pr_opt = torch.optim.Adam(self.pr.parameters(), lr=cfg.pr_lr, weight_decay=cfg.pr_l2)
        self.pm_opt = torch.optim.Adam(self.pm.parameters(), lr=cfg.pm_lr, weight_decay=cfg.pm_l2)

    @torch.no_grad()
    def encode(self, images):
        return self.vc(images.to(self.device), k=self.cfg.vc_k)

    def study(self, images):
        """Memorises a batch of drawings in one exposure."""
        images = images.to(self.device)
        vc = self.encode(images)

        with torch.no_grad():
            dg = self.dg(vc)
            stored = modules.condition_memorise(dg)
        self.ca3.train(stored, self.cfg.ca3_rule)

        # PR learns to map each vision code to its DG pattern. The DG pattern
        # is an internally made label, so no outside labels are used.
        for _ in range(self.cfg.study_steps):
            loss = F.binary_cross_entropy_with_logits(self.pr.logits(vc), dg)
            self.pr_opt.zero_grad()
            loss.backward()
            self.pr_opt.step()

        # CA1 learns to turn each stored CA3 pattern back into its drawing.
        for _ in range(self.cfg.study_steps):
            loss = F.mse_loss(self.pm(stored), images)
            self.pm_opt.zero_grad()
            loss.backward()
            self.pm_opt.step()

        with torch.no_grad():
            return {"pixels": images, "vc": vc, "dg": dg, "pr": self.pr(vc),
                    "pc": stored, "pm": self.pm(stored)}

    @torch.no_grad()
    def recall(self, images):
        """Brings back a memory for each drawing, without learning."""
        images = images.to(self.device)
        vc = self.encode(images)
        pr = self.pr(vc)
        cue = modules.condition_recall(pr, self.cfg.dg_k, self.cfg.pr_cue_gain)
        pc = self.ca3.update(cue, generator=self.generator)
        return {"pixels": images, "vc": vc, "pr": pr, "pc": pc, "pm": self.pm(pc)}
