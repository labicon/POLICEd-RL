from safety_gym.envs.suite import goal2, bench_goal_base

easy = goal2.copy()
hard = goal2.copy()
easy['goal_size'] = 0.5
easy['goal_keepout'] = 0.505

bench_goal_base.register('Easy', easy)
bench_goal_base.register('Hard', hard)
