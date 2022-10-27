import numpy as np
import tensorflow as tf


@tf.function
def direct_proj_gradient(model, input_images) -> tf.Tensor:
    if not isinstance(input_images, tf.Tensor):
        input_images = tf.convert_to_tensor(input_images)

    with tf.GradientTape() as t:
        t.watch(input_images)
        output = model(input_images)
    # gradients = t.gradient(output, input_images)
    # tf.print(gradients.shape)
    jacobians = t.batch_jacobian(output, input_images)
    gradients = tf.linalg.norm(jacobians, axis=1)
    return tf.reduce_mean(gradients, axis=0)


def gradient_norm(model, sample_points, ord=2) -> tf.Tensor:
    if not isinstance(sample_points, tf.Tensor):
        sample_points = tf.convert_to_tensor(sample_points)
    with tf.GradientTape() as t:
        t.watch(sample_points)
        output = model.inv(sample_points)

    gradients = t.gradient(output, sample_points)  # change to Jac.
    gradient_norms = tf.linalg.norm(gradients, ord=ord, axis=-1)
    return gradient_norms


def finite_differences(model, sample_points: tf.Tensor, eps: float) -> tf.Tensor:
    if not isinstance(sample_points, tf.Tensor):
        sample_points = tf.convert_to_tensor(sample_points)
    n_sample_points = sample_points.shape[0]
    dim_points = sample_points.shape[1]
    # new_sample_points = tf.vectorized_map(
    #     lambda row: tf.stack(
    #         [
    #             row + np.array([0, eps]),
    #             row + np.array([0, -eps]),
    #             row + np.array([eps, 0]),
    #             row + np.array([-eps, 0]),
    #         ]
    #     ),
    #     sample_points,
    # )  # shape = (#sample_points, 4, point_dims = 2)
    # new_sample_points = tf.reshape(
    #     tf.transpose(
    #         sample_points + eps * np.array([[[[0, 1]], [[0, -1]]], [[[1, 0]], [[-1, 0]]]]),
    #         perm=[2, 1, 0, 3],
    #     ),
    #     (n_sample_points, -1, dim_points),
    # )
    new_sample_points = tf.reshape(
        tf.transpose(
            sample_points + eps * np.array([[[[0, -1]], [[-1, 0]]], [[[0, 1]], [[1, 0]]]]),
            perm=[2, 1, 0, 3],
        ),
        (n_sample_points, -1, 2, dim_points),
    )
    flattened_sample_points = tf.reshape(new_sample_points, (-1, dim_points))

    output = model.inverse_transform(sample_points[:1])
    output_for_new_points = tf.reshape(
        model.inverse_transform(flattened_sample_points),
        (n_sample_points, -1, 2, output.shape[1]),
    )
    diffs = tf.squeeze(tf.experimental.numpy.diff(output_for_new_points, axis=-2))

    # diffs = output_for_new_points - output[:, tf.newaxis, :]
    return tf.linalg.norm(diffs, axis=-1) / 2 * eps
