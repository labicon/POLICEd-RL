import argparse
import os.path
import pathlib
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

import env_wrappers as env_wrappers
import utils as utils
from la_mbda import LAMBDA
from lbsgd.lbsgd_la_mbda import LogBarrierSafeAgent


def define_config():
  return {
      # MBPO
      'horizon': 15,
      'sequence_length': 50,
      'update_steps': 100,
      'pretrain_steps': 100,
      'discount': 0.99,
      'lambda_': 0.95,
      'steps_per_update': 1000,
      'steps_per_critic_clone': 1000,
      'batch_size': 32,
      'warmup_training_steps': 5000,
      # MODELS
      'kl_scale': 1.0,
      'kl_mix': 0.8,
      'free_nats': 3.0,
      'deterministic_size': 200,
      'stochastic_size': 30,
      'sampling_scale': 1.0,
      'units': 400,
      'posterior_samples': 5,
      'model_learning_rate': 1e-4,
      'model_learning_rate_factor': 5.0,
      'actor_learning_rate': 8e-5,
      'critic_learning_rate': 8e-5,
      'model_grad_clip_norm': 100.0,
      'actor_grad_clip_norm': 5.0,
      'critic_grad_clip_norm': 1.0,
      'swag_burnin': 500,
      'swag_period': 200,
      'swag_models': 20,
      'swag_decay': 0.8,
      # SAFETY
      'cost_threshold': 25.0,
      'penalty_mu': 5e-9,
      'lagrangian_mu': 1e-6,
      'penalty_power_factor': 1e-5,
      'm_0': 1e4,
      'm_1': 1e4,
      'eta_0': 0.1,
      'eta_rate': 8e-06,
      'backup_lr': 1e-2,
      'safety_critic_learning_rate': 2e-4,
      'safety_critic_grad_clip_norm': 50.0,
      'safety_lambda': 0.95,
      'safety_discount': 0.995,
      'cost_imbalance_weight': 100.0,
      # TRAINING
      'total_training_steps': 500000,
      'action_repeat': 2,
      'robot': 'Point',
      'task': 'Goal',
      'start_lagrangian': False,
      'lagrangian': False,
      'stop_training': False,
      'safety': False,
      'observation_type': 'rgb_image',
      'seed': 314,
      'episode_length': 1000,
      'training_steps_per_epoch': 25000,
      'evaluation_steps_per_epoch': 10000,
      'log_dir': 'runs',
      'render_episodes': 0,
      'evaluate_model': False,
      'cuda_device': '-1',
      'precision': 16
  }


def make_summary(summaries, prefix):
  epoch_summary = {
      prefix + '/average_return':
          np.asarray([sum(episode['reward']) for episode in summaries]).mean(),
      prefix + '/average_episode_length':
          np.asarray([episode['steps'][0] for episode in summaries]).mean()
  }
  if 'cost' in summaries[-1]['info'][-1].keys():
    average_cost_return = np.asarray([
        sum(list(map(lambda info: info['cost'], episode['info'])))
        for episode in summaries
    ]).mean()
    epoch_summary[prefix + '/average_cost_return'] = average_cost_return
  return epoch_summary


def evaluate(agent, train_env, logger, config, steps):
  evaluation_steps, evaluation_episodes_summaries = utils.interact(
      agent,
      train_env,
      config.evaluation_steps_per_epoch,
      config,
      training=False)
  if config.render_episodes:
    videos = list(
        map(lambda episode: episode.get('image'),
            evaluation_episodes_summaries[:config.render_episodes]))
    logger.log_video(
        np.array(videos, copy=False).transpose([0, 1, 4, 2, 3]), steps)
    if config.observation_type in ['rgb_image', 'binary_image']:
      videos = list(
          map(lambda episode: episode.get('observation'),
              evaluation_episodes_summaries[:config.render_episodes]))
      logger.log_video(
          np.array(videos, copy=False).transpose([0, 1, 4, 2, 3]) + 0.5,
          steps,
          name='observation')
  if config.evaluate_model:
    utils.evaluate_model(evaluation_episodes_summaries, agent.model, logger,
                         config.observation_type, steps)
  return make_summary(evaluation_episodes_summaries, 'evaluation')


def on_episode_end(episode_summary, logger, global_step, steps_count):
  episode_return = sum(episode_summary['reward'])
  steps = global_step + steps_count
  print("\nFinished episode with return: {}".format(episode_return))
  summary = {'training/episode_return': episode_return}
  if 'cost' in episode_summary['info'][-1].keys():
    sum_costs = sum(
        list(map(lambda info: info['cost'], episode_summary['info'])))
    summary['training/episode_cost_return'] = sum_costs
    print("Finished episode with cost return: {}".format(sum_costs))
  logger.log_evaluation_summary(summary, steps)


def train_easy(config):
  tf.get_logger().setLevel('ERROR')
  logger = utils.TrainingLogger(config)
  random.seed(config.seed)
  tf.random.set_seed(config.seed)
  np.random.seed(config.seed)
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  suffix = 'Easy-v0'
  env_name = 'sgym_Safexp-' + config.robot + config.task + suffix
  train_env = env_wrappers.make_env(env_name, config.episode_length,
                                    config.action_repeat, config.seed,
                                    config.observation_type)
  config = define_config()
  # LAMBDA and LBSGD-LAMBDA use different safety discount.
  config['safety_discount'] = 0.99 if not config[
      'start_lagrangian'] else config['safety_discout']
  config = make_config(config)
  ctor = LogBarrierSafeAgent if not config.start_lagrangian else LAMBDA
  agent = ctor(config, logger, train_env.observation_space,
               train_env.action_space)
  checkpoint = tf.train.Checkpoint(agent=agent)
  steps = 0
  if pathlib.Path(config.log_dir, 'agent_data').exists():
    checkpoint.restore(os.path.join(config.log_dir, 'agent_data', 'checkpoint'))
    steps = agent.training_step
    print("Loaded {} steps. Continuing training from {}".format(
        steps, config.log_dir))
  while steps < config.total_training_steps:
    print("Performing a training epoch.")
    training_steps, training_episodes_summaries = utils.interact(
        agent,
        train_env,
        config.training_steps_per_epoch,
        config,
        training=True,
        on_episode_end=lambda episode_summary, steps_count: on_episode_end(
            episode_summary,
            logger=logger,
            global_step=steps,
            steps_count=steps_count))
    steps += training_steps
    training_summary = make_summary(training_episodes_summaries, 'training')
    if config.evaluation_steps_per_epoch and agent.warm and \
        agent.pretrained_model:
      print("Evaluating.")
      evaluation_summaries = evaluate(agent, train_env, logger, config, steps)
      training_summary.update(evaluation_summaries)
    logger.log_evaluation_summary(training_summary, steps)
    checkpoint.write(os.path.join(config.log_dir, 'agent_data', 'checkpoint'))
  train_env.close()
  return agent, steps, logger


def train_hard(agent, steps, config, logger, train_env):
  checkpoint = tf.train.Checkpoint(agent=agent)
  if pathlib.Path(config.log_dir, 'agent_data_second_trial').exists():
    checkpoint.restore(
        os.path.join(config.log_dir, 'agent_data_second_trial', 'checkpoint'))
    steps = agent.training_step
    print("Loaded {} steps. Continuing training from {}".format(
        steps, config.log_dir))
  else:
    evaluation_summaries = evaluate(agent, train_env, logger, config, steps + 1)
    logger.log_evaluation_summary(evaluation_summaries, steps + 1)
  while steps < 2 * config.total_training_steps:
    print("Performing a training epoch.")
    training_steps, training_episodes_summaries = utils.interact(
        agent,
        train_env,
        config.training_steps_per_epoch,
        config,
        training=not config.stop_training,
        on_episode_end=lambda episode_summary, steps_count: on_episode_end(
            episode_summary,
            logger=logger,
            global_step=steps,
            steps_count=steps_count))
    steps += training_steps
    if config.stop_training:
      agent._training_step.assign(steps)
    training_summary = make_summary(training_episodes_summaries, 'training')
    if config.evaluation_steps_per_epoch:
      print("Evaluating.")
      evaluation_summaries = evaluate(agent, train_env, logger, config, steps)
      training_summary.update(evaluation_summaries)
    logger.log_evaluation_summary(training_summary, steps)
    checkpoint.write(
        os.path.join(config.log_dir, 'agent_data_second_trial', 'checkpoint'))
  train_env.close()
  return agent


def prepare_new_agent_and_task(agent, config):
  suffix = 'Hard-v0'
  env_name = 'sgym_Safexp-' + config.robot + config.task + suffix
  train_env = env_wrappers.make_env(env_name, config.episode_length,
                                    config.action_repeat, config.seed,
                                    config.observation_type)
  config = define_config()
  # Change hparams based on the used agent.
  config['safety_discount'] = 0.99 if not config['lagrangian'] else config[
      'safety_discout']
  config = make_config(config)
  ctor = LogBarrierSafeAgent if not config.lagrangian else LAMBDA
  new_agent = ctor(config, agent._logger, train_env.observation_space,
                   train_env.action_space)
  new_agent._training_step = agent._training_step
  new_agent.actor._policy = agent.actor._policy
  # Copy optimizer with its state. If needed, take the step size functionality
  # as well.
  new_agent.actor._optimizer = agent.actor._optimizer
  if not config.lagrangian and config.start_lagrangian:
    new_agent.actor._optimizer._hyper[
        'learning_rate'] = new_agent.actor.step_size
  new_agent._experience = agent._experience
  new_agent.model = agent.model
  new_agent.critic._optimizer = agent.critic._optimizer
  new_agent.critic._value = agent.critic._value
  new_agent.critic._delayed_value = agent.critic._delayed_value
  new_agent.safety_critic._optimizer = agent.safety_critic._optimizer
  new_agent.safety_critic._value = agent.safety_critic._value
  new_agent.safety_critic._delayed_value = agent.safety_critic._delayed_value
  return new_agent, train_env


def train(config):
  # Train the agent on
  agent, steps, logger = train_easy(config)
  # Migrate all the relevant parts of the old agent't internal state to the
  # new agent.
  new_agent, train_env = prepare_new_agent_and_task(agent, config)
  # Train the new agent on the harder task.
  agent = train_hard(new_agent, steps, config, logger, train_env)


def make_config(config):
  parser = argparse.ArgumentParser()
  for key, value in config.items():
    if type(value) == bool:
      assert not value, "Default bool params should be set to false."
      parser.add_argument('--{}'.format(key), action='store_true')
    else:
      parser.add_argument(
          '--{}'.format(key),
          type=type(value) if value is not None else str,
          default=value)
  return parser.parse_args()


if __name__ == '__main__':
  config = make_config(define_config())
  train(config)
