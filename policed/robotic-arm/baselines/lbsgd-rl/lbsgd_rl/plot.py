import argparse
import json
import re
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

numbers = re.compile(r'(\d+)')

BENCHMARK_THRESHOLD = 25.0

TASKS = ['point_goal', 'car_goal', 'doggo_goal']

ALGOS = [
    'lagrangian_lagrangian', 'lagrangian_interior', 'lagrangian_no_update',
    'interior_no_update', 'interior_interior'
]


def median_percentiles(metric):
  median = np.median(metric, axis=0)
  upper_percentile = np.percentile(metric, 95, axis=0, interpolation='linear')
  lower_percentile = np.percentile(metric, 5, axis=0, interpolation='linear')
  return median, upper_percentile, lower_percentile


def make_statistics(eval_rl_objectives, eval_mean_sum_costs, sum_costs,
                    timesteps):
  (objectives_median, objectives_upper,
   objectives_lower) = median_percentiles(eval_rl_objectives)
  (mean_sum_costs_median, mean_sum_costs_upper,
   mean_sum_costs_lower) = median_percentiles(eval_mean_sum_costs)
  (average_costs_median, average_costs_upper,
   average_costs_lower) = median_percentiles(sum_costs)
  return dict(
      objectives_median=objectives_median,
      objectives_upper=objectives_upper,
      objectives_lower=objectives_lower,
      mean_sum_costs_median=mean_sum_costs_median,
      mean_sum_costs_upper=mean_sum_costs_upper,
      mean_sum_costs_lower=mean_sum_costs_lower,
      average_costs_median=average_costs_median,
      average_costs_upper=average_costs_upper,
      average_costs_lower=average_costs_lower,
      timesteps=timesteps[0])


def draw(ax, timesteps, median, upper, lower, label):
  ax.plot(timesteps, median, label=label, linewidth=1)
  ax.fill_between(timesteps, lower, upper, alpha=0.2)
  ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  ax.set_xlim([0, timesteps[-1]])
  ax.xaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))


def resolve_name(name):
  if 'lagrangian_no_update' in name:
    return 'Lagrangian' + u"\u2192" + 'No Update'
  elif 'lagrangian_interior' in name:
    return 'Lagrangian' + u"\u2192" + 'LB-SGD'
  elif 'lagrangian_lagrangian' in name:
    return 'Lagrangian' + u"\u2192" + 'Lagrangian'
  elif 'interior_interior' in name:
    return 'LB-SGD' + u"\u2192" + 'LB-SGD'
  elif 'interior_no_update' in name:
    return 'LB-SGD' + u"\u2192" + 'No Update'
  else:
    return ""


def resolve_task(name):
  robot, goal = name.split('_')
  return robot.capitalize()


def draw_threshold(axes):
  for env_axes in axes:
    env_axes.axhline(BENCHMARK_THRESHOLD, ls='--', color='orangered')


def draw_experiment(metric_axes, experiment_statistics, algo):
  for ax, metric_name, label in zip(metric_axes,
                                    ['objectives', 'mean_sum_costs'], [
                                        r'$-f^0(x)$'
                                        '\nObjective', r'$f^1(x)$'
                                        '\nConstraint'
                                    ]):
    draw(
        ax,
        experiment_statistics['timesteps'],
        experiment_statistics[metric_name + '_median'],
        experiment_statistics[metric_name + '_upper'],
        experiment_statistics[metric_name + '_lower'],
        label=resolve_name(algo))
    if ax.is_first_col():
      ax.set_ylabel(label, fontsize=10)


def standardize_data(data):
  make_gen = lambda: ((*(len(data[key][i])
                         for key in data.keys()), i)
                      for i in range(len(data['objective'])))
  print(
      *('Found {} objective datapoints, {} cost datapoints, {} regret '
        'datapoints for {} timesteps in run with id {}.'.format(*elem)
        for elem in make_gen()),
      sep='\n')
  prev = None
  padding_needed = False
  max_length = 0
  for item in make_gen():
    prev = prev or item[0]
    max_length = max(max_length, item[0])
    if item[0] != prev:
      print("Found experiments with different number of data points.")
      padding_needed = True
    prev = item[0]
  if padding_needed:
    for value in data.values:
      for array in value:
        while len(array) < max_length:
          array.append(array[-1])
  return data


def aggregate_sum_costs(experiment_data):
  for i, (timesteps, regrets) in enumerate(
      zip(experiment_data['timesteps'], experiment_data['regret'])):
    df = pd.DataFrame(regrets, index=timesteps)
    mid = 1e6
    left = df[df.index <= mid]
    right = df[df.index > mid]
    compute_regret = lambda df, timesteps: df.cumsum()[0] / timesteps
    regret_left = compute_regret(left, left.index)
    regret_right = compute_regret(right, right.index - right.index[0])
    experiment_data['regret'][i] = list(regret_left) + list(regret_right)
  return experiment_data


def summarize_experiments(config):
  tasks = [task for task in TASKS if task not in config.exclude_tasks]
  algos = set(ALGOS)
  algos.difference_update(config.exclude_algos)
  algos = sorted(algos)
  fig, axes = plt.subplots(
      nrows=2,
      ncols=len(tasks),
      figsize=(6.5, 8.5 * 0.33),
      sharex='col')
  axes = axes[None,] if len(tasks) < 2 else axes
  task_to_ax_col = {task: ax_row for task, ax_row in zip(tasks, axes.T)}
  data = pd.read_json(config.data_path)
  all_results = defaultdict(dict)
  all_errors = defaultdict(dict)
  annnotations = []
  for algo, task in product(algos, tasks):
    print('Processing experiment {} {}...'.format(algo, task))
    experiment_data = data['{}/{}'.format(algo, task)]
    experiment_data = standardize_data(experiment_data)
    experiment_data = aggregate_sum_costs(experiment_data)
    experiment_statistics = make_statistics(*experiment_data.values)
    all_results[task][resolve_name(algo)] = (
        experiment_statistics['objectives_median'][-1],
        experiment_statistics['mean_sum_costs_median'][-1],
        experiment_statistics['average_costs_median'][-1])
    all_errors[task][resolve_name(algo)] = (
        ((experiment_statistics['objectives_lower'][-1],
          experiment_statistics['objectives_upper'][-1])),
        (experiment_statistics['mean_sum_costs_lower'][-1],
         experiment_statistics['mean_sum_costs_upper'][-1]),
        (experiment_statistics['average_costs_lower'][-1],
         experiment_statistics['average_costs_upper'][-1]))
    ax_col = task_to_ax_col[task]
    draw_experiment(ax_col, experiment_statistics, algo)
    ann = ax_col[0].annotate(
        resolve_task(task), (0.95, 0.1),
        ha='right',
        va='center',
        fontsize=10,
        xycoords='axes fraction')
    annnotations.append(ann)
  if not config.remove_threshold:
    draw_threshold(axes[-1, :])
  for ax in axes.flatten():
    max_xdata = max(np.asarray(line.get_xdata()).max() for line in ax.lines)
    ax.axvline(max_xdata // 2, linestyle='--', color='black')
  for ax in axes[-1, :]:
    ax.set_xlabel('Environment steps', fontsize=10, labelpad=10)
    ax.set_ylim([0, 210])
  leg = fig.legend(
      *axes[0, 0].get_legend_handles_labels(),
      loc='center',
      bbox_to_anchor=(0.5, 1.05),
      ncol=5,
      frameon=False,
      fontsize=10,
      numpoints=1,
      labelspacing=0.2,
      columnspacing=0.8,
      handlelength=1.2,
      handletextpad=0.5)
  fig.tight_layout(h_pad=1., w_pad=0.5)
  artist_to_keep = [leg] + annnotations
  plt.savefig(
      config.output_basename + '_curves.pdf',
      bbox_extra_artists=artist_to_keep,
      bbox_inches='tight')
  with open(config.output_basename + '_results.json', 'w') as file:
    json.dump(all_results, file)
  with open(config.output_basename + '_errors.json', 'w') as file:
    json.dump(all_errors, file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--remove_threshold', action='store_true')
  parser.add_argument('--output_basename', type=str, default='output')
  parser.add_argument('--exclude_tasks', nargs='+', default=[])
  parser.add_argument('--exclude_algos', nargs='+', default=[])
  args = parser.parse_args()
  import matplotlib as mpl

  mpl.rcParams["font.family"] = "serif"
  mpl.rcParams["font.serif"] = "Times New Roman"
  mpl.rcParams["text.usetex"] = True
  summarize_experiments(args)
