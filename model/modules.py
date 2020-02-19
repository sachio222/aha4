import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def get_top_k(x, k=10, mask_type="pass_through", topk_dim=0, scatter_dim=0):
    """Finds the top k values in a tensor, returns them as a tensor. 

    Accepts a tensor as input and returns a tensor of the same size. Values
    in the top k values are preserved or converted to 1, remaining values are
    floored to 0 or -1.

        Example:
            >>> a = torch.tensor([1, 2, 3])
            >>> k = 1
            >>> ans = get_top_k(a, k)
            >>> ans
            torch.tensor([0, 0, 3])

    Args:
        x: (tensor) input.
        k: (int) how many top k examples to return.
        mask_type: (string) Options: ['pass_through', 'hopfield', 'binary']
        topk_dim: (int) Which axis do you want to grab topk over? ie. batch = 0
        scatter_dim: (int) Make it the same as topk_dim to scatter the values
    """

    # Initialize zeros matrix
    zeros = torch.zeros_like(x)

    # find top k vals, indicies
    vals, idx = torch.topk(x, k, dim=topk_dim)

    # Scatter vals onto zeros
    top_ks = zeros.scatter(scatter_dim, idx, vals)

    if mask_type != "pass_through":
        # pass_through does not convert any values.

        if mask_type == "binary":
            # Converts values to 0, 1
            top_ks[top_ks > 0.] = 1.
            top_ks[top_ks < 1.] = 0.

        elif mask_type == "hopfield":
            # Converts values to -1, 1
            top_ks[top_ks >= 0.] = 1.
            top_ks[top_ks < 1.] = -1.

        else:
            raise Exception(
                'Valid options: "pass_through", "hopfield" (-1, 1), or "binary" (0, 1)')

    return top_ks


def flip_some_units(X, QTY, clone=True):
    """Randomly flips the sign of qty neurons per sample and returns matrix.
    
    Corrupts an input, returns a clone awith qty flipped units per sample
    
    Args:
        input: (n x m tensor) Accepts 2dim input tensors
        QTY: (int) Amount of flipped units per sample. 
        clone: (bool) True generates a clone; False makes changes in place.
    """

    dim0, dim1 = X.shape

    # torch.manual_seed(22)
    # Generate an (n x QTY) tensor of indicies between 0 and len(dim1)
    randomized_units = torch.randint(dim1, (dim0, QTY))

    # Return a clone of input, or operate in place.
    if not clone:
        new_X = X
    else:
        new_X = X.clone()

    # run through each sample, flip QTY units
    for i, unit in enumerate(randomized_units):
        # Flip pixels.
        new_X[i, unit] *= -1

    return new_X


def center_me_zero(x):
    x = x.clone()
    mean = x.mean()
    x -= mean
    return x


def bipolarize(x):
    """Note: If your numbers come back all ones, then all vals are positive"""
    x = x.clone()
    x[x > 0.] = 1
    x[x < 1] = -1
    return x


def all_dressed(x):
    """Zero-centers x, and polarizes values"""
    x = bipolarize(center_me_zero(x))
    return x


class InhibitionMask():
    """Creates a mask of inhibited neurons based on fired neurons. 

    Samples are multiplied by mask such that where mask = 1, a neuron is
            allowed to pass through, and where mask = 0, a neuron is 
            inhibited. Each time the mask is updated, the inhibition is
            reduced by a factor of gamma.
    """

    def __init__(self, N, dim):
        super(InhibitionMask, self).__init__()
        # set firstdim to 1 so it works on a sample not a batch
        self.phi = torch.ones(1, dim)

    def update(self, top_k_sample, gamma=1):
        """Mask value is set to 0 at location of top_k

        Args:
            top_k_sample: (binary tensor) Sample of top_k values (as 0. or 1.)
            gamma: (float) range: 0-1. Amount to decrease inhibition each timestep.
        """

        # Check if value is binary, and not hopfield. This is required for the
        # decay funcyion to work properly
        for t in top_k_sample.unique():
            if t not in (0, 1):
                raise ValueError(
                    'Input must be binary (0., 1.) for inhibition mask')

        self.phi[self.phi < 1] += gamma
        self.phi[self.phi >= 1] = 1.0

        # Elementwise mul by (1 - top k)
        # Example: 1 - 1 = 0, making active top_k index = 0 in mask.
        self.phi = self.phi * (1 - top_k_sample)

    def reset(self):
        self.phi = torch.ones(10)

    def __call__(self):
        return self.phi


class CA1(nn.Module):
    """Reconstructs the inputs that originated from EC network.

    Consists of 2 fully connected layers, recieving inputs from CA3
    and outputs to EC. 
    """

    def __init__(self, N, D_in, D_out, resize_dim):
        super(CA1, self).__init__()
        self.N, self.resize_dim = N, resize_dim

        self.fc1 = nn.Linear(D_in, 100)
        self.fc2 = nn.Linear(100, D_out)  #2704 = 52 * 52

        # Initialize uniform distribution
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        # x = torch.flatten(x)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = x.view(self.N, 1, self.resize_dim, self.resize_dim)
        return x


class DG(nn.Module):
    """Dentate Gyrus network
    
    Generates sparse output signals where neurons are inhibited directly after
    firing. 
    """

    def __init__(self, N, D_in, D_out):
        super(DG, self).__init__()
        self.fc1 = nn.Linear(D_in, D_out)
        # Initialize uniform distribution
        nn.init.xavier_uniform_(self.fc1.weight)

        # Initialize inhibition mask
        self.phi = InhibitionMask(N, D_out)

    def forward(self, X, k=10, mask_type="binary", gamma=0.01618):
        """
        Args:
            x: (tensor) sizedtorch.Size([32, 225])
        """

        X = F.leaky_relu(self.fc1(X))

        # Apply inhibition to every sample
        for i in range(len(X[:,])):
            s = X[i, :]
            s = s.clone() * self.phi().clone()
            s = get_top_k(s, k, mask_type, -1, -1)
            self.phi.update(s, gamma)
            X[i, :] = s

        return X


class EC(nn.Module):
    """Standard EC with decoder layers removed and maxpool added.
    """

    def __init__(self, N, D_in, D_out, KERNEL_SIZE, STRIDE, PADDING=0):
        super(EC, self).__init__()
        self.encoder = nn.Conv2d(D_in,
                                 D_out,
                                 kernel_size=KERNEL_SIZE,
                                 stride=STRIDE,
                                 padding=PADDING)
        self.N = N

    def forward(self, x, k):
        """
        Todo:
            Insert hooks
        """

        x = self.encoder(x)
        # print(f"x size in ec is: {x.shape}")
        x = get_top_k(x, 4, mask_type="pass_through", topk_dim=0, scatter_dim=0)
        x = F.max_pool2d(x, 4, 3)
        max_pool = x
        x = x.view(self.N, -1)
        return x, max_pool


class ECPretrain(nn.Module):
    """Pre-training... conducted on background split.

    Sparse convolutional autoencoder, develops filters that detecct a set of
            primitive visual concepts that consist of straight and curved edges, sometimes with junctions.

    Alphabets: 20
    Batches: 2000
    Batch size: 128
    """

    def __init__(self, D_in, D_out, KERNEL_SIZE, STRIDE, PADDING=0):
        """ 
        Args:
            D_in: (int) input channels, black and white, therefore 1. 
            D_out: (int) filter count, as per paper, 121
            KERNEL_SIZE: (int) 7 because ((52 - 7) / 5) + 1 = 10
                note: the paper calls for 10, but not sure what the basis
                      for using an even kernel size is. Will ask. 
            STRIDE: (int) set to 5
        """

        super(ECPretrain, self).__init__()
        self.encoder = nn.Conv2d(D_in,
                                 D_out,
                                 kernel_size=KERNEL_SIZE,
                                 stride=STRIDE,
                                 padding=PADDING)
        # self.decoder = nn.Conv2d(D_out, 1, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.encoder.weight)

        self.decoder = nn.ConvTranspose2d(D_out, D_in, KERNEL_SIZE, STRIDE,
                                          PADDING)

    def forward(self, x, k):
        """
        Todo:
            Insert hooks
        """

        x = self.encoder(x)  # Size: [64, 121, 10, 10}
        # Squeezes each character into a single pixel

        x = get_top_k(x, k, topk_dim=0,
                      scatter_dim=0)  # Size: [64, 121, 10, 10]
        # x = F.interpolate(x, 52, mode="nearest")
        x = self.decoder(x)  # Desired size: [64, 1, 52, 52]
        return torch.sigmoid(x)


class ECToCA3(nn.Module):

    def __init__(self, D_in, D_out):
        super(ECToCA3, self).__init__()

        self.fc1 = nn.Linear(D_in, 800)
        self.fch = nn.Linear(800, 1200)
        self.fc2 = nn.Linear(1200, D_out)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fch(x), 0.1)
        x = torch.sigmoid(self.fc2(x))
        return x


class CA3():
    """Hopfield architecture. Trains in one shot, content addressable memory
    """

    def __init__(self, dim1):
        super(CA3, self).__init__()

        self._train_method = {
            "hebbian": self._hebbian,
            "pinverse": self._pinverse
        }
        self.W = torch.zeros(dim1, dim1)
        self.dim1 = dim1

    def train(self, X, method="pinverse"):
        """Trains using hebbian learning rule. 
        """

        try:
            return self._train_method[method](X)
        except KeyError:
            print(f"Valid methods are 'hebbian' and 'pinverse'")

    def update(self, Y, asynch=True, n_iter=100, seed=10):
        """Update input tensor until a local minimum is reached.

        Args:
            Y: (tensor), Batch of samples.
            asych: (bool), run code asynchronously when True, synchronously
                        when false.
            n_iter: (int) Number of times to run complete updates, per sample.
                        One iteration updates all neurons once.  
            seed: (int) Debug variable for locking the randomization of neurons
                        in place for all iterations.
        """

        if not asynch:
            Y = self._synch(Y)
        else:
            Y = self._asynch(Y, n_iter, seed)

        return Y

    def energy(self, state):
        """Returns scalar energy state: E from summed state of outputs
        """

        energy = -0.5 * (state @ (self.W @ state.T)).type(torch.long)
        return energy

    def _synch(self, Y):
        """Updates all neurons in one-shot
        """

        Y = torch.mm(Y, self.W)
        Y = bipolarize(Y)
        return Y

    def _asynch(self, Y, n_iter, seed):
        """Updates each neuron in randomly selected order.

        Updates a randomly selected neuron by summing weights of all other 
        neurons and using sign of sum as new value. Selects neurons randomly
        until each neuron has been updated once. For each pass, the order in
        which each neuron is randomly selected once is changed. 

        Args:
            Y: (tensor) Input batch
            n_iter: (int) How many times to run through each batch. 
            seed: (int) Used when seeding random, for debugging, etc.
        
        Returns: Y: (tensor) output batch
        """
        for y in Y:
            E_prev = 0
            for q, _ in enumerate(range(n_iter)):

                # Generate random order to select each neuron.
                # torch.manual_seed(seed)
                idxs = torch.randperm(Y.shape[1])

                # Update each neuron once.
                for idx in idxs:

                    z = self.W[:][idx] * y.T

                    update = torch.sign(torch.sum(z))

                    # Neurons equal to 0 are quantized to 1.
                    if update == 0.:
                        update = 1.

                    y[idx] = update

                # Get energy of updated input
                E = self.energy(y)
                if E == E_prev:
                    # Go to next sample if energy is stable after seeing all
                    # neurons twice.
                    break
                else:
                    E_prev = E
        return Y

    def _hebbian(self, X):
        """Neurons that wire together, fire together.

        Neurons i and j are symmetrically connected to eeach other with a 
        weight matrix W, where W_ij = W_ji. 

        Warning: This type of learning is ineffective for this model, except
        at very small batches with very few neurons due to the sparsity of input
        signals. With many samples, results skew entirely negative.
        """
        self.W = X.T @ X
        self.W.fill_diagonal_(0)
        return self.W / X.shape[1]

    def _pinverse(self, X):
        """Dramatically increases capacity of learning model.

        Recommended in nearly all cases. 
        """
        self.W = self._hebbian(X)
        self.W = self.W.T * torch.pinverse(self.W * self.W.T)
        return self.W

    def reset(self):
        """Resets weight matrix to 0.
        """
        self.W = torch.zeros(self.dim1, self.dim1)
