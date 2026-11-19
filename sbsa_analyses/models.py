"""
The networks of Section 2.5. Both minimize the MAE; the third model type, Ridge, is
scikit-learn's Ridge regression (squared error with an L2 penalty).

make_sbsa       EBGA's EBGARegressor trained by the SBSA optimizer (Section 2.1)
MLPRegressor    the same network trained by full-batch gradient descent (PyTorch)
"""

import os

import numpy as np
import torch
from EBGA.models import EBGARegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Deterministic matrix products on CUDA (required by torch.use_deterministic_algorithms below)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------------------------------------------------------------------------
# SBSA
# ---------------------------------------------------------------------------------------------

def make_sbsa(hidden_units, activation, tau, eta, iterations, population, momentum, trust_radius,
              sigma_min, sigma_max, random_state):
    """
    One hidden layer network trained by SBSA (Algorithm 1), as EBGA's EBGARegressor.

    Full batch, MAE loss and a fixed number of iterations (early stopping off). After fit,
    the final search distribution is optimizer_.mu and optimizer_.sigma, and predict() uses mu.
    """
    return EBGARegressor(layers=[(hidden_units, activation), (1, None)], loss="mae",
                         lr_mu=tau, lr_sigma=eta, sigma_min=sigma_min, sigma_max=sigma_max,
                         calibration_size=population, max_iter=iterations, early_stopping=False,
                         momentum=momentum, trust_region_radius=trust_radius, batch_size=None,
                         random_state=random_state)


def sample_population(mu, sigma, size, seed):
    """Draw `size` parameter vectors from N(mu, diag(sigma^2)) (eq. 1)."""
    return mu + sigma * np.random.RandomState(seed).randn(size, len(mu))


def predict_population(model, X, thetas):
    """
    Predictions of each parameter vector with the network of a fitted EBGARegressor.

    Returns
    -------
    ndarray of shape (len(thetas), n_subjects)
        The network is reset to mu afterwards.
    """
    X = np.asarray(X, dtype=float)
    predictions = np.empty((len(thetas), len(X)))
    for i, theta in enumerate(thetas):
        model.network_.set_all_parameters(theta)
        predictions[i] = model.network_.forward(X).ravel()
    model.network_.set_all_parameters(model.optimizer_.mu)
    return predictions


# ---------------------------------------------------------------------------------------------
# Gradient-trained MLP
# ---------------------------------------------------------------------------------------------

def resolve_device(device):
    """Torch device for the MLPs: "cuda" when requested and available, otherwise the CPU."""
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device == "cuda":
        torch.use_deterministic_algorithms(True)  # same results in every rerun on a GPU too
    return torch.device(device)


class MLPRegressor(RegressorMixin, BaseEstimator):
    """
    One hidden layer network trained by full-batch gradient descent on the MAE.

    The update mirrors SBSA's stabilizers: the velocity v = momentum * v - lr * grad is applied
    with its norm clipped to trust_radius (momentum = 0 and trust_radius = 0 give plain gradient
    descent). The output bias starts at the mean training age, as in EBGA.

    Parameters
    ----------
    hidden_units : int, default=32
    activation : {'relu', 'linear'}, default='relu'
    lr : float, default=0.01
        Learning rate.
    epochs : int, default=1000
        Number of full-batch updates.
    alpha : float, default=0.0
        L2 penalty on the weights (not the biases), added as alpha * 0.5 * ||W||^2 / n.
    momentum : float, default=0.0
    trust_radius : float, default=0.0
        Maximum norm of each update; 0 disables the clipping.
    device : str, default='cpu'
        Device used for training; "cuda" falls back to the CPU when no GPU is available.
        Weights are initialized on the CPU, so the starting point does not depend on the
        device, and the trained network is moved back to the CPU, so that fitted models can be
        saved and loaded on any machine.
    random_state : int, default=0
    """

    def __init__(self, hidden_units=32, activation="relu", lr=0.01, epochs=1000, alpha=0.0,
                 momentum=0.0, trust_radius=0.0, device="cpu", random_state=0):
        self.hidden_units = hidden_units
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.momentum = momentum
        self.trust_radius = trust_radius
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, checkpoints=None, X_eval=None):
        """
        Train for `epochs` updates.

        If `checkpoints` (a list of epochs) is given, return instead a dictionary with the
        predictions on X_eval after each listed epoch: one run gives every epoch budget of
        the grid search.
        """
        torch.manual_seed(self.random_state)
        activation = torch.nn.ReLU() if self.activation == "relu" else torch.nn.Identity()
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden_units), activation,
                                  torch.nn.Linear(self.hidden_units, 1)).double()
        for layer in (net[0], net[2]):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        with torch.no_grad():
            net[2].bias.fill_(float(np.mean(y)))

        device = resolve_device(self.device)
        net = net.to(device)
        Xt = torch.tensor(X, dtype=torch.float64, device=device)
        yt = torch.tensor(y, dtype=torch.float64, device=device)

        params = list(net.parameters())
        velocity = [torch.zeros_like(p) for p in params]
        n_epochs = max(checkpoints) if checkpoints else self.epochs
        saved = {}
        for epoch in range(1, n_epochs + 1):
            loss = torch.mean(torch.abs(net(Xt).squeeze(1) - yt))
            if self.alpha > 0:
                weights = net[0].weight.pow(2).sum() + net[2].weight.pow(2).sum()
                loss = loss + self.alpha * 0.5 * weights / len(yt)
            grads = torch.autograd.grad(loss, params)

            with torch.no_grad():
                velocity = [self.momentum * v - self.lr * g
                            for v, g in zip(velocity, grads, strict=True)]
                scale = 1.0
                if self.trust_radius > 0:
                    norm = float(torch.sqrt(sum(v.pow(2).sum() for v in velocity)))
                    if norm > self.trust_radius:
                        scale = self.trust_radius / norm
                for p, v in zip(params, velocity, strict=True):
                    p.add_(scale * v)

            if checkpoints and epoch in checkpoints:
                saved[epoch] = _forward(net, X_eval)

        self.net_ = net.cpu()
        return saved if checkpoints else self

    def predict(self, X):
        return _forward(self.net_, X)


def _forward(net, X):
    """Predictions of a network for the rows of X, on the device where the network is."""
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float64, device=next(net.parameters()).device)
        return net(Xt).squeeze(1).cpu().numpy()
