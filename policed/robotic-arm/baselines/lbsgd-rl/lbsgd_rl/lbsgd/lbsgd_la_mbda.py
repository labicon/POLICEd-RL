import tensorflow as tf

import la_mbda
import utils
from lbsgd.lbsgd_actor import SafeActor


class LogBarrierSafeAgent(la_mbda.LAMBDA):

  def __init__(self, config, logger, observation_space, action_space):
    super(LogBarrierSafeAgent, self).__init__(config, logger, observation_space,
                                              action_space)
    self.actor = SafeActor(config, action_space.shape[0], 4)

  def _compute_safety_penalty(self, sequence, cost_values):
    cost_lambda_values = utils.compute_lambda_values(
        cost_values[:, 1:], sequence['cost'] * self._config.action_repeat,
        sequence['terminal'], self._config.safety_discount,
        self._config.safety_lambda)
    vc = tf.reduce_mean(cost_lambda_values)
    return tf.constant(self._config.cost_threshold, self._dtype) - vc, vc

  def _train_actor_critic(self, posterior_beliefs):
    posterior_beliefs = {
        k: tf.reshape(v, [-1, tf.shape(v)[-1]])
        for k, v in posterior_beliefs.items()
    }
    discount = tf.math.cumprod(
        self._config.discount * tf.ones([self._config.horizon], self._dtype),
        exclusive=True)
    with tf.GradientTape() as actor_tape:
      posterior_sequences = self.model.generate_sequences_posterior(
          posterior_beliefs, self._config.horizon, actor=self.actor)
      shape = tf.shape(posterior_sequences['features'])
      ravel_features = tf.reshape(posterior_sequences['features'],
                                  tf.concat([[-1], shape[2:]], 0))
      values = tf.reshape(self.critic(ravel_features).mode(), shape[:3])
      optimistic_sample, optimistic_value, pessimistic_sample, _ = \
          la_mbda.gather_optimistic_pessimistic_sample(posterior_sequences, values)
      lambda_values = self._compute_objective(
          optimistic_sample, tf.cast(optimistic_value, self._dtype))
      actor_loss = -tf.reduce_mean(lambda_values * discount[:-1])
      if self._config.safety:
        cost_values = tf.reshape(
            self.safety_critic(ravel_features).mode(), shape[:3])
        (
            pessimistic_cost_sample,
            pessimistic_cost_value,
            optimistic_cost_sample,
            _,
        ) = la_mbda.gather_optimistic_pessimistic_sample(
            posterior_sequences, cost_values)
        alpha, vc = self._compute_safety_penalty(
            pessimistic_cost_sample, tf.cast(pessimistic_cost_value,
                                             self._dtype))
        eta = tf.cast(self.actor.eta, self._dtype)
        if tf.greater(alpha, 0.0):
          actor_loss -= eta * tf.math.log(alpha + 0.1)
        else:
          actor_loss = -self._config.backup_lr * alpha
    gt = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
    grad_vc = tf.gradients(vc, self.actor.trainable_variables)
    lr, eta = self.actor.update_safety(alpha, gt, grad_vc)
    actor_grads_norm = self.actor.train(gt)
    critic_loss, critic_grads_norm = self.critic.train(
        pessimistic_sample['features'], pessimistic_sample['reward'],
        pessimistic_sample['terminal'])
    safety_critic_loss, safety_critic_grads_norm = self.safety_critic.train(
        pessimistic_cost_sample['features'],
        pessimistic_cost_sample['cost'] * self._config.action_repeat,
        pessimistic_cost_sample['terminal'])
    metrics = {
        'agent/actor_loss':
            actor_loss,
        'agent/actor_grads_norm':
            actor_grads_norm,
        'agent/critic_loss':
            critic_loss,
        'agent/critic_grads_norm':
            critic_grads_norm,
        'agent/pi_entropy':
            self.actor(posterior_sequences['features']).entropy(),
        'agent/lr':
            lr,
        'agent/eta':
            eta,
        'agent/alpha':
            alpha,
        'agent/safety_critic_loss':
            safety_critic_loss,
        'agent/safety_critic_grads_norm':
            safety_critic_grads_norm,
        'agent/average_safety_cost':
            tf.reduce_mean(pessimistic_cost_value)
    }
    self._log_metrics(**metrics)
