from typing import Any, Mapping, TypeVar

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant

tfd = tfp.distributions
tfpl = tfp.layers


def get_layer_builder(layer_name: str):
    return SamplingLayer.builder_for(layer_name)


class SamplingLayer(tfkl.Layer):
    def __init__(
        self,
        latent_dim: int,
        act="tanh",
        init="glorot_uniform",
        bias=1e-4,
        l1_reg=0.0,
        l2_reg=0.5,
        name="sampling",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.act = act
        self.init = init
        self.bias = bias
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    @staticmethod
    def builder_for(layer_name: str):
        if layer_name == "diagonal_normal":
            return DiagonalNormalSampling
        elif layer_name == "centered_diagonal_normal":
            return CenteredDiagonalNormalSampling
        elif layer_name == "student_t":
            return StudentTSampling
        elif layer_name == "generalized_normal":
            return GeneralizedNormalSampling
        elif layer_name == "triangle":
            return TriangleSampling
        elif layer_name == "laplace":
            return LaplaceSampling
        elif layer_name == "gumbel":
            return GumbelSampling
        elif layer_name == "polygon":
            return PolygonSampling
        else:
            raise ValueError(f"layer name {layer_name} does not correspond to an implementation")

    def sample(self, inputs, training=None):
        raise NotImplementedError("override sample() in derived classes")

    def add_kl_loss(self, samples):
        raise NotImplementedError()

    def call(self, inputs, training=None):
        z = self.sample(inputs, training=training)
        self.add_kl_loss(z)
        return z.mean(), tf.math.log(z.variance()), z

    def _layer_kwargs(self) -> Mapping[str, Any]:
        return {
            "activation": self.act,
            "kernel_initializer": self.init,
            "bias_initializer": Constant(self.bias),
            "activity_regularizer": regularizers.l1_l2(self.l1_reg, self.l2_reg),
        }

    # the **kwargs argument allows calls to this method to replace one of the default values.
    def make_dense_param_layer(self, n_params: int, **kwargs) -> tfkl.Layer:
        kwargs = self._layer_kwargs() | kwargs
        layer = tfkl.Dense(n_params, **kwargs)
        return layer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "act": self.act,
                "init": self.init,
                "bias": self.bias,
                "l1_reg": self.l1_reg,
                "l2_reg": self.l2_reg,
                "name": self.name,
            }
        )
        return config


class DiagonalNormalSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        kl_weight: float = 0.5,
        kl_mu_weight: float = 0.01,
        use_exact_kl: bool = True,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="diag_normal_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.kl_weight = kl_weight
        self.kl_mu_weight = kl_mu_weight
        self.use_exact_kl = use_exact_kl

        self.prior = tfd.MultivariateNormalDiag(
            loc=prior_loc, scale_diag=prior_scale * tf.ones(self.latent_dim)
        )

        self.dense_params = self.make_dense_param_layer(
            tfpl.IndependentNormal.params_size(self.latent_dim), name="dense_param_layer"
        )

        # Using TFPL's IndependentNormal layer apparently performs worse because the scaling
        # is forced to be positive through a softplus. Using square() solves the issue.
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., : self.latent_dim], scale_diag=tf.square(t[..., self.latent_dim :])
            ),
            activity_regularizer=None
            if self.use_exact_kl
            else tfpl.KLDivergenceRegularizer(self.prior, weight=self.kl_weight),
        )

    def add_kl_loss(self, samples):
        pass

    def sample(self, inputs, training=None):
        params = self.dense_params(inputs)
        samples = self.sampling(params)
        if training and self.use_exact_kl:
            self.add_loss(
                -self.kl_weight
                * tf.reduce_mean(
                    tf.reduce_sum(
                        tf.math.log(tf.math.pow(params[..., self.latent_dim :], 4))
                        - self.kl_mu_weight * tf.square(params[..., : self.latent_dim])
                        - tf.math.pow(params[..., self.latent_dim :], 4)
                        + 1,
                        axis=-1,
                    )
                )
            )
        return samples

    def get_config(self):
        config = super().get_config()
        # prior_loc: float = 0.0,
        # prior_scale: float = 1.0,
        # kl_weight: float = 0.5,
        # kl_mu_weight: float = 0.01,
        # use_exact_kl: bool = True,
        config.update(
            {
                "prior_loc": self.prior_loc,
                "prior_scale": self.prior_scale,
                "kl_weight": self.kl_weight,
                "kl_mu_weight": self.kl_mu_weight,
                "use_exact_kl": self.use_exact_kl,
            }
        )
        return config


class CenteredDiagonalNormalSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        kl_weight: float = 0.5,
        kl_mu_weight: float = 0.01,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="diag_normal_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.kl_weight = kl_weight
        self.kl_mu_weight = kl_mu_weight
        self.prior = tfd.MultivariateNormalDiag(
            loc=prior_loc, scale_diag=prior_scale * tf.ones(self.latent_dim)
        )

        self.dense_mean = self.make_dense_param_layer(
            2, activity_regularizer=regularizers.l1_l2(0, 0.0)
        )
        self.dense_scale = self.make_dense_param_layer(2, activation="softplus")

        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=0, scale_diag=t),
        )

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        scale = self.dense_scale(inputs)
        samples = self.sampling(scale)
        self.add_loss(
            self.kl_weight
            * tf.reduce_mean(
                tf.reduce_sum(
                    tfd.MultivariateNormalDiag(loc=0, scale_diag=scale).log_prob(samples)
                    - self.prior.log_prob(samples),
                    axis=-1,
                )
                + self.kl_mu_weight * tf.reduce_sum(tf.square(mean), axis=-1)
                # + self.scale_loss_weight
                # * tf.reduce_sum(tf.square(tf.math.log(scale_factors)), axis=-1)
            )
        )
        return mean, scale, samples + mean


class StudentTSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        df: int,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="student_t_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.df = df

        self.dense_params = self.make_dense_param_layer(
            2 * self.latent_dim,
        )
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.StudentT(
                df=self.df, loc=t[..., :2], scale=tf.square(t[..., 2:])
            ),
            activity_regularizer=tfpl.KLDivergenceRegularizer(
                tfd.StudentT(df=self.df, loc=0, scale=1), weight=0.1
            ),
        )

    def add_kl_loss(self, samples):
        pass

    def sample(self, inputs):
        params = self.dense_params(inputs)
        return self.sampling(params)


class GeneralizedNormalSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        power: float,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="gen_normal_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.power = power
        self.prior = tfd.GeneralizedNormal(
            loc=0.0,
            scale=1.0,
            power=self.power,
        )

        self.dense_params = self.make_dense_param_layer(
            2 * self.latent_dim,
        )
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=self._make_dist_from_params,
            activity_regularizer=tfpl.KLDivergenceRegularizer(
                tfd.GeneralizedNormal(
                    loc=tf.zeros(self.latent_dim), scale=tf.ones(self.latent_dim), power=self.power
                ),
                weight=0.11,
            ),
        )

    def _make_dist_from_params(self, params: tf.Tensor) -> tfd.Distribution:
        return tfd.GeneralizedNormal(
            loc=params[..., : self.latent_dim],
            scale=tf.math.square(params[..., self.latent_dim :]),
            power=self.power,
        )

    def sample(self, inputs, training):
        params = self.dense_params(inputs)
        samples = self.sampling(params)

        # self.add_loss(
        #     0.25
        #     * tf.reduce_mean(
        #         tf.reduce_sum(
        #             self._make_dist_from_params(params).log_prob(samples)
        #             - self.prior.log_prob(samples),
        #             axis=-1,
        #         )
        #     )
        # )
        return samples

    def add_kl_loss(self, samples):
        pass


class TriangleSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        use_bias: bool = True,
        loss_weight: float = 0.01,
        log_prob_loss_weight: float = 1.0,
        bias_loss_weight: float = 0.2,
        scale_loss_weight: float = 10.0,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="triangle_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.use_bias = use_bias
        self.loss_weight = loss_weight
        self.log_prob_loss_weight = log_prob_loss_weight
        self.bias_loss_weight = bias_loss_weight
        self.scale_loss_weight = scale_loss_weight

        self.triangle_corners = tf.convert_to_tensor(
            np.array([[0.0, 0.0], [0.5, np.sqrt(3) / 2], [1.0, 0.0]]).T, dtype=tf.float32
        )

        self.prior = tfd.Dirichlet(concentration=np.ones(3, dtype=np.float32))
        self.dense_bias = self.make_dense_param_layer(2)
        self.dense_rot_angle = tfk.Sequential(
            [
                self.make_dense_param_layer(1, activation="tanh", activity_regularizer=None),
                tfkl.Lambda(lambda x: x * np.pi),
            ]
        )
        self.dense_shear_x = self.make_dense_param_layer(1, activation="softplus")
        self.dense_shear_y = self.make_dense_param_layer(1, activation="softplus")
        self.dense_scale_factors = self.make_dense_param_layer(
            2,
            # activation="softplus",
            activity_regularizer=None,
            bias_initializer=Constant(0.0),
        )
        self.dense_concentration = self.make_dense_param_layer(3, activation="exponential")
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Dirichlet(concentration=t)
        )

    def make_rot_matrix(self, angle_radians: tf.Tensor) -> tf.Tensor:
        angle_radians = tf.squeeze(angle_radians, axis=-1)
        c, s = tf.cos(angle_radians), tf.sin(angle_radians)
        b = tf.eye(2) * c[:, None, None] + tf.constant([[0.0, -1.0], [1.0, 0.0]]) * s[:, None, None]
        return b  # a batch of rotation matrices (shape = [None, 2, 2])

    def make_scale_matrix(self, scaling_factors: tf.Tensor) -> tf.Tensor:
        return scaling_factors[..., None, :] * tf.eye(2)

    def make_shear_matrix(self, shear_x: tf.Tensor, shear_y: tf.Tensor) -> tf.Tensor:
        x_mat = tf.eye(2) + tf.constant([[0.0, 1.0], [0.0, 0.0]]) * shear_x[:, None]
        y_mat = tf.eye(2) + tf.constant([[0.0, 0.0], [1.0, 0.0]]) * shear_y[:, None]
        return x_mat @ y_mat

    def call(self, inputs):
        bias = self.dense_bias(inputs)
        rot_angle = self.dense_rot_angle(inputs)
        scale_factors = self.dense_scale_factors(inputs)
        concentration = self.dense_concentration(inputs)

        barycentric_coords = self.sampling(concentration)
        samples = tf.linalg.matvec(
            # self.make_shear_matrix(self.dense_shear_x(inputs), self.dense_shear_y(inputs))
            self.make_rot_matrix(rot_angle)
            @ self.make_scale_matrix(scale_factors)
            @ self.triangle_corners,
            barycentric_coords,
        ) + (bias if self.use_bias else 0.0)

        self.add_loss(
            self.loss_weight
            * tf.reduce_mean(
                self.log_prob_loss_weight
                * tf.reduce_sum(
                    tfd.Dirichlet(concentration=concentration).log_prob(barycentric_coords)
                    - self.prior.log_prob(barycentric_coords),
                    axis=-1,
                )
                + int(self.use_bias)
                * self.bias_loss_weight
                * tf.reduce_sum(tf.square(bias), axis=-1)
                + self.scale_loss_weight * tf.reduce_sum(tf.square(scale_factors - 1), axis=-1)
                # * tf.reduce_sum(tf.square(tf.math.log(tf.math.abs(scale_factors))), axis=-1)
                # * tf.reduce_sum(
                #     0.5 * tf.pow(tf.square(scale_factors) - 1, 2) / (0.25 + (scale_factors) ** 2)
                # )
            )
        )
        return tf.zeros_like(barycentric_coords), tf.zeros_like(barycentric_coords), samples

    def add_kl_loss(self, samples):
        pass


class LaplaceSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        kl_weight: float = 0.01,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="laplace_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.kl_weight = kl_weight
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.prior = tfd.Laplace(
            loc=self.prior_loc,
            scale=self.prior_scale,
        )

        # self.dense_params = self.make_dense_param_layer(2 * self.latent_dim, activity_regularization=None)
        self.dense_mean = self.make_dense_param_layer(
            self.latent_dim, activity_regularizer=regularizers.l1_l2(l1_reg, l2_reg)
        )
        self.dense_scale = self.make_dense_param_layer(self.latent_dim, activation="softplus")
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=(LaplaceSampling._make_dist_from_params),
        )

    @staticmethod
    def _make_dist_from_params(params: tf.Tensor) -> tfd.Distribution:
        # return tfd.Laplace(loc=params[..., :2], scale=tf.math.square(params[..., 2:]))
        return tfd.Laplace(loc=0.0, scale=params)

    def call(self, inputs):
        mean = self.dense_mean(inputs)
        scale = self.dense_scale(inputs)
        samples = self.sampling(scale)

        self.add_loss(
            self.kl_weight
            * tf.reduce_mean(
                tf.reduce_sum(
                    LaplaceSampling._make_dist_from_params(scale).log_prob(samples)
                    - self.prior.log_prob(samples),
                    axis=-1,
                )
                # + 0.05 * tf.reduce_sum(tf.math.square(mean), axis=-1)
                # + tf.reduce_sum(tf.math.square(tf.math.log(tf.math.square(scale))), axis=-1)
            )
        )
        return mean, scale, samples + mean

    def add_kl_loss(self, samples):
        pass


class GumbelSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        kl_weight: float = 0.1,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="gumbel_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.kl_weight = kl_weight
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.prior = tfd.Gumbel(loc=self.prior_loc, scale=self.prior_scale)

        self.dense_params = self.make_dense_param_layer(2 * self.latent_dim)
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=GumbelSampling._make_dist_from_params
        )

    @staticmethod
    def _make_dist_from_params(params: tf.Tensor) -> tfd.Distribution:
        return tfd.Gumbel(loc=params[..., :2], scale=tf.square(params[..., 2:]))

    def sample(self, inputs, training):
        params = self.dense_params(inputs)
        samples = self.sampling(params)

        self.add_loss(
            self.kl_weight
            * tf.reduce_mean(
                tf.reduce_sum(
                    GumbelSampling._make_dist_from_params(params).log_prob(samples)
                    - self.prior.log_prob(samples),
                    axis=-1,
                )
            )
        )
        return samples

    def add_kl_loss(self, samples):
        pass


class PolygonSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        polygon_vertices: np.ndarray,
        loss_weight: float = 0.2,
        log_prob_loss_weight: float = 0.1,
        bias_loss_weight: float = 0.01,
        scale_loss_weight: float = 30.0,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="polygon_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.loss_weight = loss_weight
        self.log_prob_loss_weight = log_prob_loss_weight
        self.bias_loss_weight = bias_loss_weight
        self.scale_loss_weight = scale_loss_weight
        self.prior = tfd.Dirichlet(
            concentration=np.ones(polygon_vertices.shape[0], dtype=np.float32)
        )
        self.dense_bias = self.make_dense_param_layer(2)
        self.dense_rot_angle = tfk.Sequential(
            [self.make_dense_param_layer(1, activation="tanh"), tfkl.Lambda(lambda x: x * np.pi)]
        )
        self.dense_scale_factors = self.make_dense_param_layer(2, activation="softplus")
        self.dense_concentration = self.make_dense_param_layer(
            polygon_vertices.shape[0], activation="softplus"
        )

        self.polygon_vertices = tf.convert_to_tensor(polygon_vertices.T, dtype=tf.float32)
        self.sampling = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Dirichlet(concentration=t)
        )

    def make_rot_matrix(self, angle_radians: tf.Tensor) -> tf.Tensor:
        angle_radians = tf.squeeze(angle_radians, axis=-1)
        c, s = tf.cos(angle_radians), tf.sin(angle_radians)
        b = tf.eye(2) * c[:, None, None] + tf.constant([[0.0, -1.0], [1.0, 0.0]]) * s[:, None, None]
        return b  # a batch of rotation matrices (shape = [None, 2, 2])

    def make_scale_matrix(self, scaling_factors: tf.Tensor) -> tf.Tensor:
        return scaling_factors[..., None, :] * tf.eye(2)

    def call(self, inputs):
        bias = self.dense_bias(inputs)
        rot_angle = self.dense_rot_angle(inputs)
        scale_factors = self.dense_scale_factors(inputs)
        concentration = self.dense_concentration(inputs)

        barycentric_coords = self.sampling(concentration)
        samples = (
            tf.linalg.matvec(
                self.make_scale_matrix(scale_factors)
                @ self.make_rot_matrix(rot_angle)
                @ self.polygon_vertices,
                barycentric_coords,
            )
            + bias
        )

        self.add_loss(
            self.loss_weight
            * tf.reduce_mean(
                self.log_prob_loss_weight
                * tf.reduce_sum(
                    tfd.Dirichlet(concentration=concentration).log_prob(barycentric_coords)
                    - self.prior.log_prob(barycentric_coords),
                    axis=-1,
                )
                + self.bias_loss_weight * tf.reduce_sum(tf.square(bias), axis=-1)
                + self.scale_loss_weight
                * tf.reduce_sum(tf.square(tf.math.log(scale_factors)), axis=-1)
                # + tf.reduce_sum(tf.square(concentration), axis=-1)
            )
        )
        return tf.zeros_like(barycentric_coords), tf.zeros_like(barycentric_coords), samples


class StarSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        n_points: int,
        act="tanh",
        init="glorot_uniform",
        bias=0.0001,
        l1_reg=0,
        l2_reg=0.5,
        name="star",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            init=init,
            bias=bias,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            name=name,
            **kwargs,
        )
        self.n_points = n_points
        self.prior = tfd.Uniform(low=[0.0, 0.0], high=[2 * np.pi, 1.0])

        self.dense_params = self.make_dense_param_layer(4, activation="softplus")
        self.sampled_angle = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Beta(t[..., :1], t[..., 1:2])
        )
        self.sampled_radius = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Beta(t[..., :1], t[..., 1:2])
        )
        self.dense_center = self.make_dense_param_layer(2)

    def call(self, inputs):
        params = self.dense_params(inputs)
        centers = self.dense_center(inputs)

        radii = self.sampled_radius(params[..., :2])
        angles = self.sampled_angle(params[..., 2:]) * np.pi * 2

        star_radii = self._star_radius(angles)
        # Needs to convert things to cartesian now...
        coords = tf.concat(
            [radii * star_radii * tf.cos(angles), radii * star_radii * tf.sin(angles)], axis=-1
        )

        # self.add_loss(
        #     0.1 * tf.reduce_mean(tf.reduce_sum(axis=-1))
        # )
        return tf.zeros_like(inputs), tf.zeros_like(inputs), coords + centers

    def _star_radius(self, theta: float) -> float:
        num = tf.cos(0.4 * np.pi)
        den = tf.cos(tf.asin(tf.cos(self.n_points * theta)) + np.pi * 3 / (2 * self.n_points))
        return num / den
