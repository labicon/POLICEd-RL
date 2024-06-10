import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
  value = str(value)
  parts = numbers.split(value)
  parts[1::2] = map(int, parts[1::2])
  return parts


def parse_tf_event_file(file_path):
  print('Parsing event file {}'.format(file_path))
  ea = event_accumulator.EventAccumulator(file_path)
  ea.Reload()
  if any(
      map(lambda metric: metric not in ea.scalars.Keys(), [
          'evaluation/average_return', 'evaluation/average_cost_return',
          'training/episode_cost_return'
      ])):
    return [], [], [], []
  rl_objective, safety_objective, timesteps = [], [], []
  for i, (objective, cost_objective) in enumerate(
      zip(
          ea.Scalars('evaluation/average_return'),
          ea.Scalars('evaluation/average_cost_return'))):
    rl_objective.append(objective.value)
    safety_objective.append(cost_objective.value)
    timesteps.append(objective.step)
  costs = []
  costs_iter = iter(ea.Scalars('training/episode_cost_return'))
  for step in timesteps:
    sum_costs = 0.0
    while True:
      cost = next(costs_iter)
      sum_costs += cost.value
      if cost.step >= step:
        break
    costs.append(sum_costs)
  return rl_objective, safety_objective, costs, timesteps


def parse(experiment_path, run, max_steps):
  run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps = [], \
                                                                       [], \
                                                                       [], []
  files = list(
      Path(experiment_path).glob(os.path.join(run, 'events.out.tfevents.*')))
  last_time = -1
  for file in sorted(files, key=numerical_sort):
    objective, cost_objective, sum_costs, timestamps = parse_tf_event_file(
        str(file))
    if not all([objective, cost_objective, sum_costs, timestamps]):
      print("Not all metrics are available!")
      continue
    # Filter out time overlaps, taking the first event file.
    run_rl_objective += [
        obj for obj, stamp in zip(objective, timestamps)
        if last_time < stamp <= max_steps
    ]
    run_cost_objective += [
        obj for obj, stamp in zip(cost_objective, timestamps)
        if last_time < stamp <= max_steps
    ]
    run_sum_costs += [
        obj for obj, stamp in zip(sum_costs, timestamps)
        if last_time < stamp <= max_steps
    ]
    run_timesteps += [
        stamp for stamp in timestamps if last_time < stamp <= max_steps
    ]
    last_time = timestamps[-1]

  return run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps


def parse_experiment_data(experiment_path, max_steps=2e6):
  rl_objectives, cost_objectives, sum_costs, all_timesteps = [], [], [], []
  for metrics in map(parse, itertools.repeat(experiment_path),
                     next(os.walk(experiment_path))[1],
                     itertools.repeat(max_steps)):
    run_rl_objective, run_cost_objective, run_sum_costs, run_timesteps = metrics
    rl_objectives.append(run_rl_objective)
    cost_objectives.append(run_cost_objective)
    sum_costs.append(run_sum_costs)
    all_timesteps.append(run_timesteps)
  return dict(
      objective=rl_objectives,
      cost=cost_objectives,
      regret=sum_costs,
      timesteps=all_timesteps)


def load_experiments(config):
  root, algos, _ = next(os.walk(config.data_path))
  all_results = defaultdict(dict)
  for algo in algos:
    environments = next(os.walk(os.path.join(root, algo)))[1]
    for environment in environments:
      steps = 4e6 if 'doggo' in str(environment) else 2e6
      experiment = os.path.join(root, algo, environment)
      print('Processing experiment {}...'.format(experiment))
      results = parse_experiment_data(experiment, steps)
      all_results[experiment] = results
  with open(config.output_basename + '_results.json', 'w') as file:
    json.dump(all_results, file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--output_basename', type=str, default='output')
  args = parser.parse_args()
  load_experiments(args)
