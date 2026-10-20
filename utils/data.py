"""Omniglot data: the background set for pretraining and the one-shot runs.

Omniglot (Lake et al. 2015) has 1623 handwritten characters from 50
alphabets, each drawn by 20 people. The 30 "background" alphabets are for
learning general features. The 20 "evaluation" alphabets hold the
one-shot test.

The background set comes from torchvision. The 20 official one-shot runs
come from the dataset authors' repository, the same place torchvision
downloads from. Everything lands in data/, which git ignores.
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Omniglot

RUNS_URL = ("https://raw.githubusercontent.com/brendenlake/omniglot/master/"
            "python/one-shot-classification/all_runs.zip")
RUNS_SHA256 = "437c1eeab0af8484112e1b4cc0f3d541ec727f7c47b9b202da738af9dfd6c325"
N_RUNS = 20


def invert(x):
    """Omniglot is black ink on white. Flip it so ink is 1 and paper is 0."""
    return 1.0 - x


def centre_of_mass(x):
    """Shifts a (1, H, W) drawing so its centre of ink sits mid-frame.

    The paper doesn't mention this step, but the authors' PyTorch code
    (github.com/Cerenaut/pt-aha) centres every drawing this way. Two people
    rarely draw a character in the same spot on the page, and pixel-wise
    matching punishes that hard. Shifts are sub-pixel, with bilinear
    interpolation and blank fill.
    """
    _, h, w = x.shape
    mass = x.sum()
    if mass <= 0:
        return x
    ys = torch.arange(h, dtype=x.dtype)
    xs = torch.arange(w, dtype=x.dtype)
    cy = (x[0].sum(dim=1) * ys).sum() / mass
    cx = (x[0].sum(dim=0) * xs).sum() / mass
    dy, dx = (h - 1) / 2 - cy, (w - 1) / 2 - cx

    # affine_grid works in [-1, 1] coordinates, where one pixel is 2/size.
    theta = torch.tensor([[1.0, 0.0, -2 * dx / w], [0.0, 1.0, -2 * dy / h]], dtype=x.dtype)
    grid = F.affine_grid(theta[None], [1, 1, h, w], align_corners=False)
    return F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0]


def omniglot_transform(image_size=52, centre=True):
    """105x105 drawings -> image_size x image_size tensors, ink = 1.

    The paper resizes by 0.5, from 105 to 52. Set centre=False to skip the
    centring step for an ablation.
    """
    steps = [
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(invert),
    ]
    if centre:
        steps.append(transforms.Lambda(centre_of_mass))
    return transforms.Compose(steps)


def background_set(root="data", image_size=52, centre=True):
    """The 30 background alphabets, for pretraining the vision part."""
    return Omniglot(root, background=True, download=True,
                    transform=omniglot_transform(image_size, centre))


def parse_class_labels(lines):
    """Reads a run's class_labels.txt into 0-based study indices.

    Each line looks like "run01/test/item01.png run01/training/class08.png",
    meaning test item 1 shows the character of study drawing 8.
    """
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        test, train = line.split()
        item = int(Path(test).stem.replace("item", ""))
        cls = int(Path(train).stem.replace("class", ""))
        pairs.append((item, cls - 1))
    return [cls for _, cls in sorted(pairs)]


def download_runs(root="data"):
    """Downloads and unpacks the 20 runs once. Returns the folder."""
    folder = Path(root) / "omniglot-runs"
    if (folder / "run20" / "class_labels.txt").is_file():
        return folder

    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "all_runs.zip"
    if not archive.is_file():
        print("Downloading {}".format(RUNS_URL))
        urllib.request.urlretrieve(RUNS_URL, archive)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != RUNS_SHA256:
        raise RuntimeError("all_runs.zip checksum mismatch: {}".format(digest))

    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            # Refuse anything that would unpack outside the folder.
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError("Unsafe path in archive: {}".format(name))
        z.extractall(folder)
    return folder


def load_runs(root="data", image_size=52, centre=True):
    """Loads the 20 official runs as tensors.

    Each run is a dict with:
        study:  (20, 1, H, W) study drawings, class01 ... class20
        recall: (20, 1, H, W) test drawings, item01 ... item20
        labels: (20,) index of the study drawing each test drawing matches
    """
    folder = download_runs(root)
    tsfm = omniglot_transform(image_size, centre)

    def load(path):
        return tsfm(Image.open(path).convert("L"))

    runs = []
    for r in range(1, N_RUNS + 1):
        run_dir = folder / "run{:02d}".format(r)
        study = torch.stack([load(run_dir / "training" / "class{:02d}.png".format(i))
                             for i in range(1, 21)])
        recall = torch.stack([load(run_dir / "test" / "item{:02d}.png".format(i))
                              for i in range(1, 21)])
        labels = parse_class_labels((run_dir / "class_labels.txt").read_text().splitlines())
        runs.append({"study": study, "recall": recall, "labels": torch.tensor(labels)})
    return runs
