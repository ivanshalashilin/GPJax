# %% [markdown]
# # Gaussian Processes for Vector Fields and Ocean Current Modelling
#
# In this notebook, we use Gaussian processes to learn vector valued functions. We will be
# recreating the results by [Berlinghieri et. al, (2023)](https://arxiv.org/pdf/2302.10364.pdf) by an
# application to real world ocean surface velocity data, collected via surface drifters.
#
# Surface drifters are measurement devices that measure the dynamics and circulation patterns of the world's oceans. Studying and predicting ocean currents are important to climate research, for example forecasting and predicting oil spills, oceanographic surveying of eddies and upwelling, or providing information on the distribution of biomass in ecosystems. We will be using the [Gulf Drifters Open dataset](https://zenodo.org/record/4421585), which contains all publicly available surface drifter trajectories from the Gulf of Mexico spanning 28 years.
# %%
from jax.config import config

config.update("jax_enable_x64", True)
from dataclasses import dataclass

from jax import hessian
from jax.config import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
from matplotlib import rcParams
import matplotlib.pyplot as plt
import optax as ox
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
key = jr.PRNGKey(123)
plt.style.use(
    "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
)
colors = rcParams["axes.prop_cycle"].by_key()["color"]

# %% [markdown]
# ## Data Loading and problem setting
# The dataset has binned into a $N=34\times16$ grid, equally spaced over the longitude-latitude interval $[-90.8,-83.8] \times [24.0,27.5]$. Each bin has a size $\approx 0.21\times0.21$, and contains the average velocity across all measurements that fall inside it.
#
# We will call this binned ocean data the ground truth, and label it with the vector field
# $$
# \mathbf{F} \equiv \mathbf{F}(\mathbf{x}),
# $$
# where $\mathbf{x} = (x^{(0)}$,$x^{(1)})^\text{T}$, with a vector basis in the standard Cartesian directions (dimensions will be indicated by superscripts).
#
# We shall label the dataset $D_0=\left\{ \left(\mathbf{x}_{0,i} , \mathbf{y}_{0,i} \right)\right\}_{i=1}^N$, where $\mathbf{y}_i$ is the 2 dimensional velocity vector at the $i$th location, $\mathbf{x}_i$. 20% of the data are copied and allocated to training data in a dataset $D_T=\left\{\left(\mathbf{x}_{T,i}, \mathbf{y}_{T,i} \right)\right\}_{i=1}^{N_T}$, $N_T = 109$ in this case (the subscript indicates the original dataset and the training dataset respectively).
#

# %%
# loading in data
try:
    gulf_data = pd.read_csv("data/gulfdata.csv")
except FileNotFoundError:
    gulf_data = pd.read_csv("docs/examples/data/gulfdata.csv")
shape = (30, 30)

N = len(gulf_data["lonbins"])
pos = jnp.array([gulf_data["lonbins"], gulf_data["latbins"]]).T
vel = jnp.array([gulf_data["ubar"], gulf_data["vbar"]]).T
shape = (int(gulf_data["shape"][0]), int(gulf_data["shape"][1]))  # shape = (34,16)

# split 80-20 training testing
pos_test, pos_train, vel_test, vel_train = train_test_split(
    pos, vel, test_size=0.20, random_state=43
)

pos_train = pos_train.T
pos_test = pos.T
vel_train = vel_train.T
vel_test = vel.T


fig, ax = plt.subplots(1, 1)
ax.quiver(
    pos_test[0], pos_test[1], vel_test[0], vel_test[1], color=colors[0], label="$D_0$"
)
ax.quiver(
    pos_train[0],
    pos_train[1],
    vel_train[0],
    vel_train[1],
    color=colors[1],
    alpha=0.7,
    label="$D_T$",
)
ax.legend()


# %% [markdown]
# ## Problem Setting
# Our aim is to obtain estimates for $\mathbf{F}$ at the set of points $\left\{ \mathbf{x}_{0,i} \right\}_{i=1}^N$ using Gaussian processes, followed by a comparison of the latent model to the ground truth ($D_0$). Note that $D_0$ is not used by the GPJax implementation, and is only used to compare against the model.
#
# Since $\mathbf{F}$ is a vector-valued function, we would ideally require GPs that can learn vector-valued functions[<sup>1</sup>](#fn1). Since $D_T$ contains a 2D vector measurement $\mathbf{y}_{T,i}$ at each location $\mathbf{x}_{T,i}$, we require a trick to implement this in GPJax. The problem can be changed to learn a scalar-valued function by 'massaging' the data into a  $2N\times2N$ problem, such that each dimension of our GP is associated with a *component* of $\mathbf{y}_{T,i}$.
#
# For a particular $\mathbf{y}$ (training or testing) at location $\mathbf{x}$, the components $(y^{(0)}, y^{(1)})$ are described by the latent vector field $\mathbf{F}$ such that
#
# $$
# \mathbf{y} = \mathbf{F}(\mathbf{x}) = \left(\begin{array}{l}
# f^{(0)}\left(\mathbf{x}\right) \\
# f^{(1)}\left(\mathbf{x}\right)
# \end{array}\right)
# $$
#
# where each $f^{(z)}\left(\mathbf{x}\right), z \in \{0,1\}$ is a scalar valued function. Now consider the scalar-valued function $g: \mathbb{R}^2 \times\{0,1\} \rightarrow \mathbb{R}$, such that
#
# $$
# g \left(\mathbf{x} , 0 \right) = f^{(0)} ( \mathbf{x} ), \text{and } g \left( \mathbf{x}, 1 \right)=f^{(1)}\left(\mathbf{x}\right).
# $$
#
# We have increased the input dimension by 1, from the 2D $\mathbf{x}$ to 3D $\left(\mathbf{x}, 0\right)$ or $\left(\mathbf{x}, 1\right)$
#
# By choosing the value of the third dimension, 0 or 1, we may now incorporate this information into computation of the kernel.
# We therefore make new 3D datasets $D_{T,3D} = \left\{\left( \mathbf{X}_{T,i},\mathbf{Y}_{T,i} \right) \right\} _{i=0}^{16}$ and $D_{0,3D} = \left\{\left( \mathbf{X}_{0,i},\mathbf{Y}_{0,i} \right) \right\} _{i=0}^{2N}$ that incorporates this new labelling, such that for each dataset (indicated by the subscript $D = 0$ or $D=T$),
#
# $$
# X_{D,i} = \left( \mathbf{x}_{D,i}, z \right),
# $$
# and
# $$
# Y_{D,i} = y_{D,i}^{(z)},
# $$
#
# where $z = 0$ if $i$ is odd and $z=1$ if $i$ is even.

# %%


# Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
def label_position(data):
    # introduce alternating z label
    n_points = len(data[0])
    label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
    return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T


def stack_velocity(data):
    return data.T.flatten().reshape(-1, 1)


pos_train3d = label_position(pos_train)
vel_train3d = stack_velocity(vel_train)
# we also require the testing data to be relabelled for later use, such that we can query the 2Nx2N GP at the test points
pos_test3d = label_position(pos_test)

DT = gpx.Dataset(X=pos_train3d, y=vel_train3d)


# %% [markdown]
# ## Velocity (Dimension) Decomposition
# Having labelled the data, we are now in a position to use a GP to learn the function $g$, and hence $\mathbf{F}$. A naive approach to the problem is to apply a GP prior directly onto the velocities of each dimension independently, which is called the *velocity* GP. For our prior, we choose an isotropic mean 0 over all dimensions of the GP, and a piecewise kernel that depends on the $z$ labels of the inputs, such that for two inputs $\mathbf{X} = \left( \mathbf{x}, z \right )$ and $\mathbf{X}^\prime = \left( \mathbf{x}^\prime, z^\prime \right )$,
#
# $$
# k_{\text{vel}} \left(\mathbf{X}, \mathbf{X}^{\prime}\right)= \begin{cases}k^{(z)}\left(\mathbf{x}, \mathbf{x}^{\prime}\right) & \text { if } z=z^{\prime} \\ 0 & \text { if } z \neq z^{\prime}\end{cases}
# $$
#
# where $k^{(z)}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ are the user are kernels for each dimension. What this means is that there are no correlations between the $x^{(0)}$ and $x^{(1)}$ dimensions for all choices $\mathbf{X}$ and $\mathbf{X}^{\prime}$, since there are no off-diagonal elements in the Gram matrix populated by this choice.
#
# To implement this approach in GPJax, we define `velocity_kernel` in the following cell, following the steps outlined in the creating new kernels notebook. This modular implementation takes the choice of user kernels as its class attributes: `kernel0` and `kernel1`. We must additionally pass the argument `active_dims = [0,1]`, which is an attribute of the base class `AbstractKernel`, into the chosen kernels. This is necessary such that the subsequent likelihood optimisation does not optimise over the artificial label dimension.
#

# %%


@dataclass
class velocity_kernel(gpx.kernels.AbstractKernel):
    kernel1 = gpx.kernels.RBF(active_dims=[0, 1])
    kernel0 = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0. acheive the correct value via 'Switches'

        z = X[2]
        zp = Xp[2]

        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)


# %% [markdown]
# ### GPJax Implementation
# Next, we define the model in GPJax. The prior is defined using $k_{\text{vel}}\left(\mathbf{X}, \mathbf{X}^\prime \right)$ and 0 mean. We choose a Gaussian marginal log likelihood (MLL).
#

# %%


def initialise_gp(kernel, mean, dataset):
    prior = gpx.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.Gaussian(
        num_datapoints=dataset.n, obs_noise=jnp.array([1.0e-6], dtype=jnp.float64)
    )
    posterior = prior * likelihood
    return posterior


# Define the velocity GP
mean = gpx.mean_functions.Zero()
kernel = velocity_kernel()
dataset = DT
velocity_posterior = initialise_gp(kernel, mean, dataset)


# %% [markdown]
# With a model now defined, we can proceed to optimise the hyperparameters of our likelihood over $D_0$. This is done by minimising the marginal log likelihood using `optax`. We also plot its value at each step to visually confirm that we have found the minimum. See [introduction to Gaussian Processes](https://docs.jaxgaussianprocesses.com/examples/intro_to_gps/) notebook for more information on optimising the MLL.

# %%

from functools import partial


def optimise_mll(posterior, dataset, NIters=5000, key=key, plot_history=True):
    # define the Marginal Log likelihood using DT
    objective = gpx.objectives.ConjugateMLL(negative=True)
    objective(posterior, train_data=DT)
    optimiser = ox.adam(learning_rate=0.1)
    # Optimise to minimise the MLL
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimiser,
        num_iters=NIters,
        safe=True,
        key=key,
    )
    if plot_history:
        fig, ax = plt.subplots(1, 1)
        ax.plot(history, color=colors[1])
        ax.set(xlabel="Training iteration", ylabel="Negative marginal log likelihood")

    return opt_posterior


opt_velocity_posterior = optimise_mll(velocity_posterior, dataset)

# %% [markdown]
# ### Comparison
# We next obtain the latent distribution of the GP of $g$ at $\mathbf{x}_{0,i}$, then extract its mean and standard at the test locations, $\mathbf{F}_{\text{latent}}(\mathbf{x}_{0,i})$, as well as the standard deviation (we will use it at the very end).

# %%


def latent_distribution(opt_posterior):
    latent = opt_posterior.predict(pos_test3d, train_data=DT)
    latent_mean = latent.mean()
    latent_std = latent.stddev()
    return latent_mean, latent_std


# extract x0 and x1 values over g
velocity_mean, velocity_std = latent_distribution(opt_velocity_posterior)

vel_lat = [velocity_mean[::2].reshape(shape), velocity_mean[1::2].reshape(shape)]
pos_lat = pos_test

# %% [markdown]
# We now replot the ground truth (testing data) $D_0$, the predicted latent vector field $\mathbf{F}_{\text{latent}}(\mathbf{x_i})$, and a heatmap of the residuals at each location $R(\mathbf{x}_i) = \left|\left| \mathbf{y}_{0,i} - \mathbf{F}_{\text{latent}}(\mathbf{x}_i) \right|\right|$.

# %%


# Residuals between ground truth and estimate
def plot_fields(pos_train, pos_test, vel_train, vel_test, pos_lat, vel_lat, shape):
    Y = pos_test[1]
    X = pos_test[0]
    # make figure
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    fig.tight_layout()
    # ground truth
    ax[0].quiver(
        pos_test[0],
        pos_test[1],
        vel_test[0],
        vel_test[1],
        color=colors[0],
        label="Training data",
    )
    ax[0].quiver(
        pos_train[0],
        pos_train[1],
        vel_train[0],
        vel_train[1],
        color=colors[1],
        label="Test Data",
    )
    ax[0].set(
        xlim=[X.min() - 0.1, X.max() + 0.1],
        ylim=[Y.min() + 0.1, Y.max() + 0.1],
        aspect="equal",
        title="Ground Truth",
    )

    # Latent estimate of vector field F
    ax[1].quiver(
        pos_lat[0],
        pos_lat[1],
        vel_lat[0],
        vel_lat[1],
        color=colors[3],
        label="Latent estimate of Ground Truth",
    )
    ax[1].quiver(
        pos_train[0], pos_train[1], vel_train[0], vel_train[1], color=colors[1]
    )
    ax[1].set(
        xlim=[X.min() - 0.1, X.max() + 0.1],
        ylim=[Y.min() + 0.1, Y.max() + 0.1],
        aspect="equal",
        title="GP Latent Estimate",
    )
    # residuals
    residuals_vel = jnp.sqrt(
        (vel_test[0].reshape(shape) - vel_lat[0]) ** 2
        + (vel_test[1].reshape(shape) - vel_lat[1]) ** 2
    )
    im = ax[2].imshow(
        residuals_vel, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap="hot"
    )
    ax[2].set(
        xlim=[X.min() - 0.1, X.max() + 0.1],
        ylim=[Y.min() + 0.1, Y.max() + 0.1],
        aspect="equal",
        title="Residuals",
    )
    fig.colorbar(im, fraction=0.046, pad=0.04, orientation="vertical")


plot_fields(pos_train, pos_test, vel_train, vel_test, pos_lat, vel_lat, shape)


# %% [markdown]
# From the residuals it is evident the GP does not perform well far from the training data. This is because our construction of the kernel placed an independent prior on each physical dimension. This is incorrect, as by definition $f^{(z)}$ is directly proportional to $x^{(z^\prime)}$ for both physical dimensions. Therefore, we need a different approach that can implicitly incorporate this at a fundamental level. To achieve this we will require a *Helmholtz Decomposition*.

# %% [markdown]
# ## Helmholtz Decomposition
# In 2 dimensions, a twice continuously differentiable and compactly supported vector field $\mathbf{F}: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ can be expressed as the sum of the gradient of a scalar potential $\Phi: \mathbb{R}^2 \rightarrow \mathbb{R}$, called the potential function, and the vorticity operator of another scalar potential $\Psi: \mathbb{R}^2 \rightarrow \mathbb{R}$, called the stream function ([Berlinghieri et. al, (2023)](https://arxiv.org/pdf/2302.10364.pdf)):
# $$
# \mathbf{F}=\operatorname{grad} \Phi+\operatorname{rot} \Psi
# $$
# where
# $$
# \operatorname{grad} \Phi:=\left[\begin{array}{l}
# \partial \Phi / \partial x^{(0)} \\
# \partial \Phi / \partial x^{(1)}
# \end{array}\right] \text { and } \operatorname{rot} \Psi:=\left[\begin{array}{c}
# \partial \Psi / \partial x^{(1)} \\
# -\partial \Psi / \partial x^{(0)}
# \end{array}\right]
# $$
#
# This is reminiscent of a 3 dimensional [Helmholtz decomposition](https://en.wikipedia.org/wiki/Helmholtz_decomposition).
#
# The 2 dimensional decomposition motivates a different approach: placing priors on $\Psi$ and $\Phi$, allowing us to make assumptions directly about fundamental properties of $\mathbf{F}$. If we choose independent GP priors such that $\Phi \sim \mathcal{G P}\left(0, k_{\Phi}\right)$ and $\Psi \sim \mathcal{G P}\left(0, k_{\Psi}\right)$, then $\mathbf{F} \sim \mathcal{G P} \left(0, k_\text{Helm}\right)$ (since acting linear operations on a GPs give GPs).
#
# For $\mathbf{X}, \mathbf{X}^{\prime} \in \mathbb{R}^2 \times \left\{0,1\right\}$ and $z, z^\prime \in \{0,1\}$,
#
# $$
# \boxed{ k_{\mathrm{Helm}}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)_{z,z^\prime} =  \frac{\partial^2 k_{\Phi}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(z)} \partial\left(x^{\prime}\right)^{(z^\prime)}}+(-1)^{z+z^\prime} \frac{\partial^2 k_{\Psi}\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(1-z)} \partial\left(x^{\prime}\right)^{(1-z^\prime)}}}.
# $$
#
# where $x^{(z)}$ and $(x^\prime)^{(z^\prime)}$ are the $z$ and $z^\prime$ components of $\mathbf{X}$ and ${\mathbf{X}}^{\prime}$ respectively.
#
# We compute the second derivatives using `jax.hessian`. In the following implementation, for a kernel $k(\mathbf{x}, \mathbf{x}^{\prime})$, this computes the Hessian matrix with respect to the components of $\mathbf{x}$
#
# $$
# \frac{\partial^2 k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial x^{(z)} \partial x^{(z^\prime)}}.
# $$
#
# Note that we have operated $\dfrac{\partial}{\partial x^{(z)}}$, *not* $\dfrac{\partial}{\partial \left(x^\prime \right)^{(z)}}$, as the boxed equation suggests. This is not an issue if we choose stationary kernels $k(\mathbf{x}, \mathbf{x}^{\prime}) = k(\mathbf{x} - \mathbf{x}^{\prime})$ , as the partial derivatives with respect to the components have the following exchange symmetry:
#
# $$
# \frac{\partial}{\partial x^{(z)}} = - \frac{\partial}{\partial \left( x^\prime \right)^{(z)}}
# $$
#
# for either $z$.
# %%


@dataclass
class helmholtz_kernel(gpx.kernels.AbstractKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    potetial_kernel = gpx.kernels.RBF(active_dims=[0, 1])
    stream_kernel = gpx.kernels.RBF(active_dims=[0, 1])

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        sign = (-1) ** (z + zp)
        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        potential_dvtve = -jnp.array(
            hessian(self.potetial_kernel)(X, Xp), dtype=jnp.float64
        )[z][zp]
        stream_dvtve = -jnp.array(
            hessian(self.stream_kernel)(X, Xp), dtype=jnp.float64
        )[1 - z][1 - zp]

        return potential_dvtve + sign * stream_dvtve


# %% [markdown]
# ### GPJax Implementation
# We repeat the exact same steps as with the velocity GP model, but replacing `velocity_kernel` with `helmholtz_kernel`.

# %%
# Redefine Gaussian process with Helmholtz kernel
kernel = helmholtz_kernel()
helmholtz_posterior = initialise_gp(kernel, mean, dataset)
# Optimise hyperparameters using optax
opt_helmholtz_posterior = optimise_mll(helmholtz_posterior, DT)


# %% [markdown]
# ### Comparison
# We again plot the ground truth (testing data) $D_0$, the predicted latent vector field $\mathbf{F}_{\text{latent}}(\mathbf{x_i})$, and a heatmap of the residuals at each location $R(\mathbf{x}_i) = \left|\left| \mathbf{y}_{0,i} - \mathbf{F}_{\text{latent}}(\mathbf{x}_i) \right|\right|$.

# %%

# obtain latent distribution, extract x and y values over g
helmholtz_mean, helmholtz_std = latent_distribution(opt_helmholtz_posterior)
vel_lat = [helmholtz_mean[::2].reshape(shape), helmholtz_mean[1::2].reshape(shape)]
pos_lat = pos_test

plot_fields(pos_train, pos_test, vel_train, vel_test, pos_lat, vel_lat, shape)

# %% [markdown]
# Visually, the Helmholtz model performs better than the velocity model, preserving the local structure of the $\mathbf{F}$, supportd by the residuals being much smaller than with the velocity decomposition. Since we placed priors on $\Phi$ and $\Psi$, the construction of $\mathbf{F}$ allows for correlations between the dimensions (non-zero off diagonal elements in the Gram matrix populated by $k_\text{Helm}\left(\mathbf{X},\mathbf{X}^{\prime}\right)$ ).


# %% [markdown]
# ## Negative Log Predictive Densities
# Lastly, we directly compare the velocity and Hemlholtz models by computing the [negative log predictive densities](https://en.wikipedia.org/wiki/Negative_log_predictive_density) for each model. This is a quantitative metric that measures the probability of the ground truth given the data,
#
# $$
# \mathrm{NLPD}=-\sum_{i=1}^{2N} \log \left(  p\left(\mathcal{Y}_i = Y_{0,i} \mid \mathbf{X}_{i}\right) \right)
# $$
#
# where each $p\left(\mathcal{Y}_i \mid \mathbf{X}_i \right)$ is the marginal Gaussian distribution at each test location, and $Y_{i,0}$ is the $i$th component of the (massaged) test data that we reserved at the beginning of the notebook in $D_0$. A smaller value is better, since the deviation of the ground truth and the model are small in this case.

# %%
# ensure testing data alternates between x0 and x1 components


def nlpd(mean, std, vel_test):
    vel_query = jnp.column_stack((vel_test[0], vel_test[1])).flatten()
    normal = tfp.substrates.jax.distributions.Normal(loc=mean, scale=std)
    return -jnp.sum(normal.log_prob(vel_query))


nlpd_vel = nlpd(velocity_mean, velocity_std, vel_test)
nlpd_helm = nlpd(helmholtz_mean, helmholtz_std, vel_test)

print("NLPD for Velocity: %.2f \nNLPD for Helmholtz: %.2f" % (nlpd_vel, nlpd_helm))
# %% [markdown]
# The Helmholtz model significantly outperforms the velocity model, as indicated by the lower NLPD score.

# %% [markdown]
# <span id="fn1"></span>
# ## Footnote
# Kernels for vector valued functions have been studied in the literature, see [Alvarez et. al, (2012)](https://doi.org/10.48550/arXiv.1106.6251)
# ## System Configuration
# %%
# %reload_ext watermark
# %watermark -n -u -v -iv -w -a 'Ivan Shalashilin'
