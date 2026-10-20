"""The brain-region parts of AHA, the Artificial Hippocampal Algorithm.

AHA comes from "AHA! an 'Artificial Hippocampal Algorithm' for Episodic
Machine Learning" by Kowadlo, Ahmed and Rawlinson (2019, arXiv 1909.10340).
It follows Complementary Learning Systems (CLS) theory: a slow learner, the
neocortex, picks up general features over many examples, and a fast learner,
the hippocampus, memorises single experiences in one shot.

The paper names each part twice, as Function(Biology). I kept the
biological names for the classes and put the paper's functional name in each
docstring:

    ECPretrain, EC  ->  VC(EC)      Vision Component, the long-term memory (LTM)
    DG              ->  PS(DG)      Pattern Separation
    CA3             ->  PC(CA3)     Pattern Completion
    ECToCA3         ->  PR(EC-CA3)  Pattern Retrieval
    CA1             ->  PM(CA1)     Pattern Mapping

EC is the entorhinal cortex, the main link between neocortex and hippocampus.
DG is the dentate gyrus. CA3 and CA1 are fields of the hippocampus proper
(CA stands for cornu Ammonis). Everything except the vision part makes up
AHA's short-term memory (STM), which model/aha.py wires together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_top_k(x, k=10, mask_type="pass_through", topk_dim=0, scatter_dim=0):
    """Keeps the top k values of x along one dimension and zeroes the rest.

    Returns a tensor the same size as x. Pick the dimension carefully: for a
    batch you almost always want the units of each sample (dim 1), not the
    batch itself (dim 0).

        Example:
            >>> get_top_k(torch.tensor([1., 2., 3.]), k=1)
            tensor([0., 0., 3.])

    Args:
        x: (tensor) input.
        k: (int) how many values to keep.
        mask_type: (string) "pass_through" keeps the values, "binary" gives
            1 for the top k and 0 elsewhere, "hopfield" gives 1 for the top k
            and -1 elsewhere.
        topk_dim: (int) the dimension to rank over.
        scatter_dim: (int) keep this the same as topk_dim.
    """

    vals, idx = torch.topk(x, k, dim=topk_dim)

    if mask_type == "pass_through":
        return torch.zeros_like(x).scatter(scatter_dim, idx, vals)

    # Build the mask from the winning indices, not from the values. The old
    # version thresholded values, so zeros and negative winners came out wrong.
    mask = torch.zeros_like(x).scatter(scatter_dim, idx, 1.0)

    if mask_type == "binary":
        return mask
    if mask_type == "hopfield":
        return 2.0 * mask - 1.0

    raise ValueError(
        'Valid options: "pass_through", "hopfield" (-1, 1), or "binary" (0, 1)')


def gaussian_kernel(size, std):
    """A normalised 2D Gaussian, size x size, used by the interest filter."""
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax**2) / (2 * std**2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def condition_memorise(x):
    """Turns a 0/1 DG pattern into -1/+1 for the Hopfield net.

    Paper, Appendix A.1: X' = 2 sgn(X) - 1. Hopfield nets store bipolar
    patterns best.
    """
    return 2.0 * (x > 0).float() - 1.0


def condition_recall(y, k=10, gain=10.0):
    """Turns a PR(EC-CA3) output into a cue the Hopfield net can settle from.

    Paper, Appendix A.2. Each sample is scaled to sum to `gain`, mapped to the
    Hopfield range [-1, 1], then shifted by an offset (theta in the paper) so
    that at least k units are positive. Anything negative acts as
    inhibition, and the cue is mostly background, so without the shift the
    -1s would swamp the few units that should switch on.

    Args:
        y: (tensor) PR outputs in [0, 1], shape (batch, units).
        k: (int) active units in each stored pattern.
        gain: (float) gamma in the paper, 10.
    """
    z = gain * y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    cue = 2.0 * z - 1.0

    # Find the kth and (k+1)th largest values. If fewer than k units are
    # positive, slide the whole sample up so zero falls between them.
    top = torch.topk(cue, k + 1, dim=1).values
    kth, next_one = top[:, k - 1:k], top[:, k:k + 1]
    theta = torch.where(kth <= 0, -(kth + next_one) / 2, torch.zeros_like(kth))
    return cue + theta


class InterestFilter(nn.Module):
    """Suppresses the empty background in the vision code.

    Paper, Section 2.4.5 and Appendix B.3. Omniglot drawings are mostly blank
    page, and blank page still makes the convolution fire. This filter loosely
    copies the retina's centre-surround cells with a Difference of Gaussians
    (DoG) kernel, finds the strongest edges, and only lets the vision code
    through at those spots. A Gaussian blur then gives some tolerance to
    where exactly a feature sits.
    """

    def __init__(self, dog_size=7, dog_std=0.82, dog_k=1.6, nms_size=5, k=20,
                 smooth_size=15, smooth_std=2.375, resolution=1e-9):
        super(InterestFilter, self).__init__()
        self.nms_size, self.k, self.resolution = nms_size, k, resolution

        dog = gaussian_kernel(dog_size, dog_std) - gaussian_kernel(dog_size, dog_std * dog_k)
        self.register_buffer("dog", dog[None, None])
        # The blur is a Gaussian, which splits into a row pass and a column
        # pass. Same result, and far faster than a 15x15 kernel on the CPU.
        g = torch.exp(-(torch.arange(smooth_size) - (smooth_size - 1) / 2) ** 2 / (2 * smooth_std**2))
        self.register_buffer("smooth", (g / g.sum()).float())

    def mask(self, images):
        """Returns a (batch, 1, H, W) mask with up to 2k interest points."""

        # Peak finding compares pixels for exact equality, which float32
        # can't do reliably: batched CPU convolution leaves noise near 1e-8
        # on blank page and breaks ties between neighbouring pixels, and
        # each device does it differently. So the mask is always built in
        # float64 on the CPU, rounded to a fine step to clear the last
        # noise. It's one 52x52 channel per drawing, so it costs little,
        # and every device gets the same points.
        # Move first, then cast. From MPS, .to("cpu", torch.float64) in one
        # call came back as all zeros.
        device = images.device
        x = images.detach().cpu().double()
        dog = self.dog.detach().cpu().double()
        response = F.conv2d(x, dog, padding=dog.shape[-1] // 2)
        response = torch.round(response / self.resolution) * self.resolution

        mask = torch.zeros_like(response)
        # DoG+ catches light-to-dark edges and DoG- catches dark-to-light.
        for r in (response, -response):
            # Non-maxima suppression: keep a pixel only if it beats its
            # neighbours, so one stroke edge gives one point, not a smear.
            peaks = F.max_pool2d(r, self.nms_size, 1, self.nms_size // 2)
            r = r * (r == peaks) * (r > 0)

            flat = r.flatten(1)
            top = get_top_k(flat, self.k, "binary", 1, 1) * (flat > 0)
            mask = mask + top.view_as(r)

        return mask.clamp(max=1.0).float().to(device)

    def forward(self, images, features):
        features = features * self.mask(images)
        b, c, h, w = features.shape
        x = features.reshape(b * c, 1, h, w)
        pad = self.smooth.numel() // 2
        x = F.conv2d(x, self.smooth.view(1, 1, -1, 1), padding=(pad, 0))
        x = F.conv2d(x, self.smooth.view(1, 1, 1, -1), padding=(0, pad))
        return x.reshape(b, c, h, w)


class ECPretrain(nn.Module):
    """VC(EC), the Vision Component, set up for pretraining.

    Paper, Section 2.4.5 and Appendix B. A single-layer sparse convolutional
    autoencoder learns 121 filters on the Omniglot background set. After
    training they look like strokes: straight and curved edges, sometimes
    with junctions. Those strokes are the reusable "primitives" the
    hippocampus combines into a memory of a new character.

    It is a winner-take-all autoencoder (Makhzani and Frey 2015):
      * At each position, only the k strongest filters stay on (k=1 here).
      * During training, each filter also keeps its single strongest
        response in the batch, so every filter learns something each step.
        The paper calls this lifetime sparsity.
      * The decoder reuses the encoder's weights (tied weights).

    Pretraining runs at stride 5 and the full model runs the same filters at
    stride 1 (paper Table 3). With 10x10 filters and padding 4, a 52x52 image
    becomes an 11x11 code and decodes back to exactly 52x52.
    """

    def __init__(self, D_in=1, D_out=121, KERNEL_SIZE=10, STRIDE=5, PADDING=4):
        """
        Args:
            D_in: (int) input channels, 1 for black and white.
            D_out: (int) filter count, 121 as per paper.
            KERNEL_SIZE: (int) filter width, 10 as per paper.
            STRIDE: (int) 5 for pretraining, as per paper.
            PADDING: (int) 4, so 52 -> 11 -> 52 round trips exactly.
        """
        super(ECPretrain, self).__init__()
        self.encoder = nn.Conv2d(D_in, D_out, kernel_size=KERNEL_SIZE,
                                 stride=STRIDE, padding=PADDING)
        # Tied weights: the decoder only owns its bias.
        self.decoder_bias = nn.Parameter(torch.zeros(D_in))
        nn.init.xavier_uniform_(self.encoder.weight)

    def encode(self, x, k=1):
        """Returns the sparse code, shape (batch, 121, 11, 11) for 52x52."""
        z = self.encoder(x)

        # Top k over the filters (dim 1) at every position of every sample.
        code = get_top_k(z, k, topk_dim=1, scatter_dim=1)

        if self.training:
            # Lifetime sparsity: each filter's strongest response anywhere in
            # the batch survives too.
            per_filter = z.transpose(0, 1).reshape(z.shape[1], -1)
            best = per_filter.argmax(dim=1, keepdim=True)
            keep = torch.zeros_like(per_filter).scatter(1, best, 1.0)
            keep = keep.reshape(z.shape[1], z.shape[0], *z.shape[2:]).transpose(0, 1)
            code = torch.where(keep.bool(), z, code)

        return code

    def logits(self, x, k=1):
        """Pre-sigmoid reconstruction, for a numerically stable loss."""
        code = self.encode(x, k)
        return F.conv_transpose2d(code, self.encoder.weight, self.decoder_bias,
                                  stride=self.encoder.stride,
                                  padding=self.encoder.padding)

    def forward(self, x, k=1):
        return torch.sigmoid(self.logits(x, k))


class EC(nn.Module):
    """VC(EC), the Vision Component, as used inside AHA.

    Same filters as ECPretrain, now at stride 1 with 4 winners per position
    (paper Table 3). The output goes through the interest filter, then 4x4
    max-pooling, then gets flattened: 121 x 13 x 13 = 20449 numbers per
    drawing. This is the long-term memory's view of the drawing, and the
    input to every hippocampal part.

    An even filter width can't be padded evenly, so I pad 4 on the top and
    left and 5 on the bottom and right to keep the map at 52x52, matching
    the paper's 52x52x121 figure.
    """

    def __init__(self, D_in=1, D_out=121, KERNEL_SIZE=10, STRIDE=1, pool=4,
                 interest_filter=True):
        super(EC, self).__init__()
        self.encoder = nn.Conv2d(D_in, D_out, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.pad = ((KERNEL_SIZE - 1) // 2, KERNEL_SIZE - 1 - (KERNEL_SIZE - 1) // 2)
        self.pool = pool
        self.interest = InterestFilter() if interest_filter else None

    def load_pretrained(self, state_dict):
        """Copies the filters from an ECPretrain state dict."""
        self.encoder.load_state_dict({"weight": state_dict["encoder.weight"],
                                      "bias": state_dict["encoder.bias"]})

    def feature_map(self, x, k=4):
        """Sparse filter responses before pooling, (batch, 121, 52, 52)."""
        before, after = self.pad
        z = self.encoder(F.pad(x, (before, after, before, after)))
        z = get_top_k(z, k, topk_dim=1, scatter_dim=1)
        if self.interest is not None:
            z = self.interest(x, z)
        return z

    def forward(self, x, k=4):
        z = F.max_pool2d(self.feature_map(x, k), self.pool, self.pool)
        return z.flatten(1)

    def output_dim(self, image_size=52):
        """How many numbers forward() returns per image."""
        stride = self.encoder.stride[0]
        side = (image_size - 1) // stride + 1
        return self.encoder.out_channels * (side // self.pool) ** 2


class InhibitionMask():
    """Temporarily silences DG neurons that just fired.

    Paper, Section 2.4.1. Real neurons have a refractory period after they
    fire. Here a neuron that fires gets fully inhibited, and the inhibition
    decays by gamma each time a new drawing comes in. Calling the mask gives
    1 where a neuron may pass and close to 0 where it was just used, so
    drawings shown one after another end up on different neurons.
    """

    def __init__(self, dim, gamma=0.95):
        self.dim, self.gamma = dim, gamma
        self.reset()

    def update(self, top_k_sample):
        """Marks the neurons in top_k_sample (0s and 1s) as just fired."""

        # Check if values are binary, and not hopfield. The decay below
        # assumes 1 means fired.
        if not torch.all((top_k_sample == 0) | (top_k_sample == 1)):
            raise ValueError('Input must be binary (0., 1.) for inhibition mask')

        fired = top_k_sample.reshape(1, self.dim).to(self.inhibition.device)
        self.inhibition = self.gamma * torch.maximum(self.inhibition, fired)

    def reset(self):
        self.inhibition = torch.zeros(1, self.dim)

    def to(self, device):
        self.inhibition = self.inhibition.to(device)
        return self

    def __call__(self):
        return 1.0 - self.inhibition


class DG(nn.Module):
    """PS(DG), Pattern Separation. The dentate gyrus.

    Paper, Section 2.4.1. Its job is to work like a hash function: similar
    drawings should come out as very different, non-overlapping patterns. It
    is a single fully connected layer with random weights that never trains
    (like an Extreme Learning Machine). Three things push the outputs apart:
      * only the top k=10 of 225 neurons fire for each drawing,
      * 25% of the incoming weights are knocked out to zero, like the sparse
        wiring between DG and CA3,
      * neurons that just fired are inhibited for the next drawings.

    The output is a 0/1 pattern with exactly k ones. AHA stores it in CA3 and
    also uses it as the target that PR(EC-CA3) learns to produce.
    """

    def __init__(self, D_in, D_out=225, k=10, gamma=0.95, knockout=0.25):
        super(DG, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(D_in, D_out, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        with torch.no_grad():
            keep = (torch.rand_like(self.fc1.weight) >= knockout).float()
            self.fc1.weight.mul_(keep)
        self.fc1.weight.requires_grad_(False)

        self.phi = InhibitionMask(D_out, gamma)

    def reset(self):
        self.phi.reset()

    def forward(self, X):
        Z = self.fc1(X)
        self.phi.to(Z.device)

        # Drawings go through one at a time, because each one's winners
        # inhibit the next one's.
        out = torch.zeros_like(Z)
        for i in range(Z.shape[0]):
            s = get_top_k(Z[i:i + 1] * self.phi(), self.k, "binary", 1, 1)
            self.phi.update(s)
            out[i] = s[0]

        return out


class ECToCA3(nn.Module):
    """PR(EC-CA3), Pattern Retrieval. The path from EC straight into CA3.

    Paper, Section 2.4.3. DG makes a brand new pattern for every drawing, even
    a second copy of the same character, so DG can't help you find an old
    memory. This network learns to map the vision code of a drawing to the
    DG pattern that was stored for it. Because vision codes of similar
    drawings overlap, a new drawing of a studied character tends to retrieve
    the right memory. The paper finds this is where most of AHA's
    classification accuracy comes from.

    Two layers, 1000 hidden units (paper Table 3), leaky ReLU then sigmoid,
    trained as multi-label classification with binary cross-entropy.
    """

    def __init__(self, D_in, D_out=225, hidden=1000):
        super(ECToCA3, self).__init__()
        self.fc1 = nn.Linear(D_in, hidden)
        self.fc2 = nn.Linear(hidden, D_out)

    def logits(self, x):
        """Pre-sigmoid outputs. Train on these with BCE-with-logits: at the
        paper's learning rate a plain sigmoid saturates to exactly 0 and the
        gradient dies."""
        return self.fc2(F.leaky_relu(self.fc1(x)))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


class CA3():
    """PC(CA3), Pattern Completion. A Hopfield network.

    Paper, Section 2.4.2. A Hopfield network is a content-addressable memory:
    store some patterns, hand it a noisy or partial one, and it settles on
    the closest stored pattern. That's what CA3's recurrent wiring is
    thought to do. It trains in one shot.

    Stored patterns come from DG. Cues come from PR(EC-CA3). Neurons are
    graded, tanh with gain 2.7, and each of 70 steps updates 20 randomly
    chosen neurons (paper Table 3).
    """

    def __init__(self, dim1, gain=2.7, n_update=20, n_iter=70):
        self._train_method = {
            "hebbian": self._hebbian,
            "pinverse": self._pinverse
        }
        self.dim1 = dim1
        self.gain, self.n_update, self.n_iter = gain, n_update, n_iter
        self.W = torch.zeros(dim1, dim1)

    def train(self, X, method="pinverse"):
        """Stores the -1/+1 patterns in X, shape (patterns, neurons)."""
        if method not in self._train_method:
            raise ValueError("Valid methods are 'hebbian' and 'pinverse'")
        return self._train_method[method](X)

    def update(self, Y, generator=None):
        """Lets each cue in Y settle into an attractor and returns the states.

        Asynchronous updates: each step picks n_update neurons at random and
        sets each one from the weighted sum of all the others. One neuron at
        a time is guaranteed to converge. 20 at a time is faster and works
        just as well in practice, per the paper.
        """
        state = Y.clone()
        n = state.shape[1]
        for _ in range(self.n_iter):
            idx = torch.randperm(n, generator=generator)[:self.n_update].to(state.device)
            state[:, idx] = torch.tanh(self.gain * (state @ self.W[:, idx]))
        return state

    def energy(self, state):
        """Hopfield energy of each state in a batch, -1/2 s W s^T.

        Lower is more stable. Kept as a float: the old version cast it to an
        integer, which threw away the fraction.
        """
        return -0.5 * ((state @ self.W) * state).sum(dim=1)

    def _hebbian(self, X):
        """Neurons that fire together, wire together.

        Warning: Hebbian learning holds only a handful of patterns, and
        correlated patterns (all of ours share lots of -1s) interfere. Kept
        for comparison.
        """
        W = X.T @ X / X.shape[1]
        W.fill_diagonal_(0)
        self.W = W
        return self.W

    def _pinverse(self, X):
        """Pseudo-inverse (projection) rule. Recommended in nearly all cases.

        W = X^T (X X^T)^-1 X, which equals pinv(X) @ X. W projects any state
        onto the space spanned by the stored patterns, so each stored
        pattern maps to itself and is a fixed point. It holds far more
        patterns than Hebbian learning and is computed in one step.

        The old code wrote this with elementwise * instead of a matrix
        product. I compute it in float64 on the CPU because pinverse is not
        on every GPU backend and float32 loses precision here.
        """
        X64 = X.detach().cpu().double()
        self.W = (torch.linalg.pinv(X64) @ X64).float().to(X.device)
        return self.W

    def reset(self):
        """Resets weight matrix to 0."""
        self.W = torch.zeros(self.dim1, self.dim1, device=self.W.device)


class CA1(nn.Module):
    """PM(CA1), Pattern Mapping. Turns a CA3 memory back into a picture.

    Paper, Section 2.4.4. CA3 patterns are effectively random symbols. CA1
    learns to map each one back to what was seen, so a recalled memory can be
    replayed to the neocortex. Like the paper, I train it to rebuild the
    input image, which also makes the output easy to look at.

    Two layers: 100 hidden units, then one output per pixel (52 x 52 = 2704),
    leaky ReLU on both, trained with mean squared error (MSE).
    """

    def __init__(self, D_in=225, D_out=2704, hidden=100, image_size=52):
        super(CA1, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(D_in, hidden)
        self.fc2 = nn.Linear(hidden, D_out)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x.view(-1, 1, self.image_size, self.image_size)
