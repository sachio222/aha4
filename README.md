# AHA: an Artificial Hippocampal Algorithm in PyTorch

AHA is a model of the hippocampus that learns a new drawing from one look, with no labels. It comes from Kowadlo, Ahmed and Rawlinson (2019), who pair a slow-learning vision network with a fast-learning short-term memory built from four parts named after hippocampal regions. This repo is my PyTorch implementation of it, tested on the paper's one-shot benchmark with Omniglot, a set of handwritten characters from 50 alphabets.

![Signals passing through AHA on one-shot run 1](docs/aha_run01.png)

*One-shot run 1. Rows from top: the 20 drawings AHA studies, the sparse pattern stored for each, new drawings of the same characters by a different person, what pattern retrieval brings back, the Hopfield memory it settles on, and the image it recalls. 16 of 20 columns recall the right character.*

## Results

The paper names each part as Function(Brain region). The regions are the entorhinal cortex (EC), the main link between the neocortex and the hippocampus, the dentate gyrus (DG), and two fields of the hippocampus proper, CA3 and CA1. VC(EC) is the vision network. The four hippocampal parts are PS(DG) for pattern separation, PC(CA3) for pattern completion, PR(EC-CA3) for pattern retrieval and PM(CA1) for pattern mapping. PC(CA3) is a Hopfield network, a memory that settles a noisy input onto the closest pattern it has stored. A table further down maps each part to its code.

The benchmark is Lake et al.'s one-shot classification test (2015). There are 20 runs. In each run the model studies 20 handwritten characters from one alphabet, then has to match 20 new drawings of those characters, by a different person, back to the study set. Chance is 5%. Like the paper, I score each stage of the model by matching on minimum mean squared error (MSE) and average over the runs. Every run starts from a freshly initialised short-term memory, and the whole set repeats over 10 seeds.

| Stage scored | What it measures | This repo | Paper |
|---|---|---|---|
| Raw pixels | Floor for comparison | 47.5% | not reported |
| VC(EC) alone | Vision network only, the paper's baseline | 56.0% | 71.6% |
| PR(EC-CA3) | AHA, the paper's headline number | **61.3%** ± 0.9 | **86.4%** |
| PC(CA3) | AHA after the Hopfield memory settles | 62.5% ± 0.8 | "no better than" the baseline |
| PM(CA1) | AHA's recalled image | 61.5% ± 0.9 | not reported |

My result is lower than the paper's. AHA scores 61.3% here against 86.4% reported, and the vision network alone scores 56.0% against 71.6%. The vision network has no hippocampal parts in it, so about 16 of the 25 missing points are gone before AHA's memory gets involved. I have not found the cause. AHA adds about 5 points over vision alone, where the paper reports about 15. One more difference: here the Hopfield stage, PC(CA3), scores slightly above pattern retrieval, while the paper found it no better than vision alone.

Numbers come from `results/oneshot.json`: 10 seeds on an Apple M-series GPU through MPS, about 3 minutes in all. A one-seed run on the CPU takes 87 seconds and gives the same accuracy as the matching MPS seed at every stage but PM, where one of 400 classifications differs (`results/cpu_seed0.json`).

### Ablations

Each row uses the same 10 seeds and the same pretrained vision network, except the pretraining-seed row.

| Change | VC(EC) | PR(EC-CA3) | PC(CA3) |
|---|---|---|---|
| Default settings | 56.0% | 61.3% | 62.5% |
| Vision network pretrained with seeds 1 and 2 (mean) | 56.6% | 62.8% | 63.8% |
| No interest filter | 56.5% | 50.0% | 25.2% |
| No centring of drawings | 33.8% | 40.2% | 40.3% |
| Hebbian learning in CA3 instead of pseudo-inverse | 56.0% | 61.3% | 5.1% |

Turning off the interest filter barely changes vision alone but drops the Hopfield stage to 25%. With Hebbian learning, CA3 falls to chance: every stored pattern shares most of its -1 units with every other, and Hebbian weights can't keep patterns that similar apart. Each row has a JSON file in `results/`.

## How to run

Tested on Python 3.12. The scripts pick a device for you (`--device auto`: CUDA, then Apple MPS, then CPU), or you can pass `--device cpu`, `mps` or `cuda`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 32 tests, a few seconds
```

Omniglot and the 20 one-shot runs download into `data/` on first use.

**Evaluate.** A pretrained vision network ships in `checkpoints/vc_pretrained.pth.tar` (51 KB, filters only), so you can skip pretraining.

```bash
python evaluate.py --seeds 10                      # writes results/oneshot.json
python evaluate.py --no-interest-filter           # ablation; also --ca3-rule hebbian
```

**Pretrain.** Trains the vision network on the 30 background alphabets: 2000 batches of 128, about 2 to 3 minutes on MPS. It overwrites the shipped checkpoint and saves `figures/vc_filters.png`.

```bash
python pretrain.py --seed 0
```

For the no-centring ablation, pretrain with `--no-centre --name vc_nocentre`, then evaluate with `--no-centre --checkpoint checkpoints/vc_nocentre.pth.tar`.

**Train.** In AHA the hippocampal parts train inside every test: they study a set once, get tested, and get wiped. `train.py` runs one of those cycles on one run, prints each stage's accuracy and saves the picture at the top of this page.

```bash
python train.py --run 1    # writes figures/aha_run01.png
```

## How the code maps to the paper

| Code (`model/modules.py`) | Paper part | What it does | Size |
|---|---|---|---|
| `ECPretrain`, `EC` | VC(EC), Vision Component | Sparse convolutional autoencoder. Learns stroke-like filters on other alphabets, then stays frozen. This is the long-term memory. | 121 filters, 10x10. Output 121 x 13 x 13 = 20449 |
| `InterestFilter` | Part of VC(EC) | Keeps the vision code only near the strongest edges, so blank page stops dominating | Up to 40 points per drawing |
| `DG` | PS(DG), Pattern Separation | Fixed random layer that gives each drawing its own sparse pattern, even near-duplicates | 225 units, 10 active |
| `CA3` | PC(CA3), Pattern Completion | Hopfield network that stores the DG patterns in one step and settles a noisy cue onto the closest one | 225 units |
| `ECToCA3` | PR(EC-CA3), Pattern Retrieval | Learns to map a vision code to its DG pattern, so a new drawing of a studied character finds its memory | 20449 to 1000 to 225 |
| `CA1` | PM(CA1), Pattern Mapping | Turns a CA3 pattern back into an image | 225 to 100 to 2704 (52 x 52) |

`model/aha.py` wires them together. During study, DG makes a pattern for each drawing, CA3 stores it, PR learns vision code to pattern, and CA1 learns pattern to image. During recall, PR turns a new drawing into a cue, CA3 settles, and CA1 draws what it remembers.

![Filters learned by the vision network](docs/vc_filters.png)

*The 121 filters the vision network learns on the background alphabets. Most are pen strokes: edges, curves and junctions.*

## What differs from the paper

- **Centring.** Each drawing is shifted so its centre of ink sits mid-frame. The paper doesn't mention this, but the authors' later PyTorch code ([Cerenaut/pt-aha](https://github.com/Cerenaut/pt-aha)) does it, and it moves every score a lot (see ablations).
- **Interest filter order.** Appendix B.3 leaves the order of masking and blurring open. I mask the vision code at the interest points, then blur, then max-pool. Kernel sizes come from Table 3. The authors' later PyTorch code has this filter but leaves it switched off.
- **Pretraining loss.** The paper doesn't name one. I use binary cross-entropy on the reconstruction.
- **PR hidden layer.** 1000 units, from Table 3. The paper's Figure 13 says 800.
- **Padding.** A 10-wide filter can't be padded evenly. At stride 5 I pad 4 on each side, which maps 52 to 11 and back to 52. At stride 1 I pad 4 before and 5 after to keep 52 x 52.
- **DG inhibition.** The paper's formula for the refractory period reads two ways. I fully inhibit a unit when it fires and let the inhibition decay by 0.95 with each new drawing.
- **Hopfield cue offset.** Appendix A.2 shifts the cue so at least k units are positive. I shift only when fewer than k are, and centre zero between the kth and (k+1)th values.
- **CA1 training input.** CA1 learns from the stored CA3 patterns during study. The paper doesn't say whether it trains on those or on recalled ones.
- **States compared for matching.** For PR(EC-CA3) I compare its output on the study drawings, after learning, with its output on the test drawings. For PC(CA3) I compare the stored pattern with the state the network settles into.
- **Scope.** Only one-shot classification without noise or occlusion. The paper's instance-classification task, its noise and occlusion curves, and its FastNN baseline are not here.

## Credit

AHA is the work of Gideon Kowadlo, Abdelrahman Ahmed and David Rawlinson: [AHA! an 'Artificial Hippocampal Algorithm' for Episodic Machine Learning](https://arxiv.org/abs/1909.10340) (arXiv 1909.10340, 2019). Omniglot and the one-shot runs come from Brenden Lake and colleagues ([brendenlake/omniglot](https://github.com/brendenlake/omniglot)).

Thanks to Gideon Kowadlo and David Rawlinson for guiding me through some of the harder parts of the model, and to @ptrblck on the PyTorch forums for walking me through some of the more confusing parts of PyTorch.

## License

MIT, see [LICENSE](LICENSE).
