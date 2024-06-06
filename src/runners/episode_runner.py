from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import os
import json

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        print(self.batch_size)
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.episode_total = 0
        
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, if_test=None, cur_time=None):
        self.batch = self.new_batch()
        if if_test==None:
            self.env.reset(args=self.args, cur_time=cur_time)
        else:
            self.env.reset(if_test=if_test,args=self.args, cur_time=cur_time)

        self.t = 0

    def run(self, test_mode=False, cur_time=None):
        monster_hp_json_file_path = os.path.join('./results/sacred/hok/monster_last_hp/', f'{cur_time}_monster_lasthp_{self.args.name}.txt')
        if self.args.env=='hok':
            self.reset(if_test=test_mode, cur_time=cur_time)
        else:
            self.reset(cur_time=cur_time)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
           
            #actions = actions[0] # for ippo
            # Fix memory leak
            cpu_actions = actions[0].to("cpu").numpy()

            # 这是跟环境交互的最重要的一步
            reward, terminated, env_info = self.env.step(actions[0], if_test=test_mode)
           
            episode_return += reward
            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
            #print(f'self.t {self.t}, terminated {terminated}')

            # 加入暴龙血量的文件保存
            if env_info['monster_last_hp'] is not None:
                '''此时到达了最后一帧，可以拿到当前帧的大龙血量了 -> [1, 2, ...]'''
                # 检查目标文件夹是否存在，如果不存在则创建
                if not os.path.exists(os.path.dirname(monster_hp_json_file_path)):
                    os.makedirs(os.path.dirname(monster_hp_json_file_path))
                # 将数据存储到与当前文件夹相同的文件中，如果没有这个文件，则先创建
                if not os.path.exists(monster_hp_json_file_path):
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write('[]')
                    
                    with open(monster_hp_json_file_path, 'r') as file:
                        content = file.read()
                        numbers = eval(content)
                        numbers.append(env_info['monster_last_hp'])
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write(str(numbers))
                # 如果有了，则采用追加格式
                else:
                    with open(monster_hp_json_file_path, 'r') as file:
                        content = file.read()
                        numbers = eval(content)
                        numbers.append(env_info['monster_last_hp'])
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write(str(numbers))
            
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        } # last_data应该没有作用，这个是充数的
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.episode_total += 1

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, env_info

    def _log(self, returns, stats, prefix):
        # stats: {'battle_won': 0, 'n_episodes': 1, 'ep_length': 45} battle_won从哪里来的
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
