"""
Preprocessing fitted on training subjects only (Sections 2.6 and 2.6.1).

The input X of every step holds the raw ROI features with the acquisition batch as its last
column. Age is used only by ComBat's fit, as the covariate whose effect is preserved; it is
never a model input, and transform never needs it.
"""

import neurocombat_sklearn.neurocombat_sklearn as _neurocombat
import numpy as np
from neurocombat_sklearn import CombatModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

# neurocombat-sklearn 0.1.3 (its latest release) calls OneHotEncoder(sparse=False), an argument
# scikit-learn removed in version 1.4. Map it to sparse_output; nothing else in the library changes.
_neurocombat.OneHotEncoder = lambda sparse=False, **kw: OneHotEncoder(sparse_output=sparse, **kw)


class ComBat(TransformerMixin, BaseEstimator):
    """
    ComBat harmonization (neurocombat-sklearn) as a Pipeline step.

    The last column of X is the acquisition batch. fit() estimates each batch's location and
    scale effects while preserving the effect of age (y); transform() removes only the
    location effect of the subject's batch, so it never uses the subjects' age, and training
    and test subjects are transformed in the same way.

    Parameters
    ----------
    age_model : {'spline', 'linear'}, default='spline'
        How age enters ComBat's design matrix. 'linear' is a single age column. 'spline' is a
        cubic B-spline basis of age with knots at quantiles of the training ages, so that
        non-linear age trajectories are not absorbed into the effects of batches that cover a
        narrow age range. This follows ComBat-GAM (Pomponio et al., NeuroImage 2020) with two
        differences: the spline is not penalized (on OpenBHB the batch shifts change by at
        most 0.02 SD between 4 and 9 knots), and transform removes only the batch shift,
        whereas ComBat-GAM also needs the subjects' age when it is applied.
    n_knots : int, default=5
        Number of spline knots (age_model='spline' only).
    """

    def __init__(self, age_model="spline", n_knots=5):
        self.age_model = age_model
        self.n_knots = n_knots

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        age = np.reshape(y, (-1, 1))
        if self.age_model == "spline":
            # The basis sums to one, like the batch columns of ComBat's design matrix, so one
            # basis function is dropped (include_bias=False) to keep the design full rank.
            self.age_basis_ = SplineTransformer(
                n_knots=self.n_knots, degree=3, knots="quantile", include_bias=False
            ).fit(age)
            covariates = self.age_basis_.transform(age)
        elif self.age_model == "linear":
            covariates = age
        else:
            raise ValueError(f"unknown age_model: {self.age_model}")
        self.combat_ = CombatModel().fit(X[:, :-1], X[:, -1:], continuous_covariates=covariates)
        return self

    def transform(self, X):
        check_is_fitted(self, "combat_")
        X = np.asarray(X, dtype=float)
        batches, names = X[:, -1], self.combat_.sites_names
        if not np.isin(batches, names).all():
            raise ValueError("batch not seen in fit")
        shift = self.combat_.gamma_star[np.searchsorted(names, batches)]
        return X[:, :-1] - shift * np.sqrt(self.combat_.var_pooled).T


def make_preprocessing(harmonization, age_model="spline", n_knots=5):
    """
    Preprocessing pipeline of one setting.

    Parameters
    ----------
    harmonization : {'combat', 'one_hot'}
        'combat': ComBat, then z-scoring (reference configuration).
        'one_hot': no ComBat; z-scored features plus the batch one-hot encoded (Section 2.6.1).
    age_model, n_knots
        Passed to ComBat.
    """
    if harmonization == "combat":
        return Pipeline([
            ("combat", ComBat(age_model=age_model, n_knots=n_knots)),
            ("scale", StandardScaler()),
        ])
    if harmonization == "one_hot":
        return ColumnTransformer([
            ("scale", StandardScaler(), slice(0, -1)),
            ("batch", OneHotEncoder(sparse_output=False, handle_unknown="error"), [-1]),
        ])
    raise ValueError(f"unknown harmonization: {harmonization}")
