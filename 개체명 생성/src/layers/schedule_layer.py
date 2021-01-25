import tensorflow.compat.v1 as tf


def to_float(x):
  """Cast x to float; created because tf.to_float is deprecated."""
  return tf.cast(x, tf.float32)

def inverse_exp_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)
  progress = tf.minimum(step / float(max_step), 1.0)
  return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)

  def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

  def inv_sigmoid(y):
    return tf.log(y / (1 - y))

  assert min_value > 0, (
      "sigmoid's output is always >0 and <1. min_value must respect "
      "these bounds for interpolation to work.")
  assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

  # Find
  #   x  s.t. sigmoid(x ) = y_min and
  #   x' s.t. sigmoid(x') = y_max
  # We will map [0, max_step] to [x_min, x_max].
  y_min = min_value
  y_max = 1.0 - min_value
  x_min = inv_sigmoid(y_min)
  x_max = inv_sigmoid(y_max)

  x = tf.minimum(step / float(max_step), 1.0)  # [0, 1]
  x = x_min + (x_max - x_min) * x  # [x_min, x_max]
  y = sigmoid(x)  # [y_min, y_max]

  y = (y - y_min) / (y_max - y_min)  # [0, 1]
  y = y * (1.0 - y_min)  # [0, 1-y_min]
  y += y_min  # [y_min, 1]
  return y