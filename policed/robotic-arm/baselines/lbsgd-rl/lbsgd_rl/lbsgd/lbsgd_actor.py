import tensorflow as tf

from models import Actor


class LogBarrierAdaptiveStepSize(tf.Module):

  def __init__(self, m_0, m_1, base_lr, eta_0, eta_rate):
    super().__init__()
    self._m_0 = tf.constant(m_0, tf.float32)
    self._m_1 = tf.constant(m_1, tf.float32)
    self._lr = tf.Variable(base_lr, False)
    self._base_lr = base_lr
    self.eta = tf.Variable(eta_0, False)
    self._eta_rate = 1.0 + eta_rate

  def update_with_grads(self, alpha, gt, grad_vc):
    self.eta.assign(tf.convert_to_tensor(self.eta) / self._eta_rate)
    lr = self._compute_gamma(alpha, gt, grad_vc)
    # 1. NaNs; 2. Negative LRs (fallback to base);
    self._lr.assign(
        lr if tf.math.is_finite(lr) and tf.greater_equal(lr, 0.0) else self
        ._base_lr)
    return self._lr.value(), self.eta.value()

  def _compute_gamma(self, alpha, gt, grad_vc):

    def to_vec(grads):
      flat = tf.nest.map_structure(lambda x: tf.reshape(x, [
          -1,
      ]), grads)
      return tf.concat(flat, 0)

    gt = tf.cast(to_vec(gt), tf.float32)
    grad_vc = tf.cast(to_vec(grad_vc), tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    lhs = alpha / (2.0 * tf.abs(tf.tensordot(grad_vc, gt, 1)) / tf.norm(gt) +
                   tf.sqrt(alpha * self._m_1))
    eta_t = tf.convert_to_tensor(self.eta)
    m_2 = self._m_0 + 4.0 * eta_t * (self._m_1 /
                                     (alpha + 1e-8)) + 4.0 * eta_t * tf.norm(
                                         tf.tensordot(grad_vc, gt, 1))**2 / (
                                             tf.norm(gt)**2 * alpha**2)
    rhs = 1.0 / m_2
    return tf.minimum(lhs, rhs)

  def __call__(self, *args, **kwargs):
    return self._lr.value()

  def get_config(self):
    return {
        'm_0': self._m_0,
        'm_1': self._m_1,
        'base_lr': self._base_lr,
        'eta_0': self.eta,
        'eta_rate': self._eta_rate
    }


class SafeActor(Actor):

  def __init__(self, config, size, layers):
    super().__init__(config, size, layers)
    self.step_size = LogBarrierAdaptiveStepSize(config.m_0, config.m_1,
                                                config.actor_learning_rate,
                                                config.eta_0, config.eta_rate)
    self._optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.step_size,
        clipnorm=self._config.actor_grad_clip_norm)

  def train(self, grads): # noqa
    norm = tf.linalg.global_norm(grads)
    grads, _ = tf.clip_by_global_norm(grads, self._config.actor_grad_clip_norm,
                                      norm)
    if not tf.math.is_nan(norm):
      self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
    return norm

  def update_safety(self, alpha, gt, grad_vc):
    return self.step_size.update_with_grads(alpha, gt, grad_vc)

  @property
  def eta(self):
    return self.step_size.eta.value()
