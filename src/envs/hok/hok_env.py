
import path, sys
from pathlib import Path
folder = path.Path(__file__).abspath()
current_dir = Path(__file__).parent # 获得当前目录路径
sys.path.append(str(current_dir))# 如果使用相对路径,并且添加当前目录的上两级目录

from envs.multiagentenv import MultiAgentEnv
import numpy as np
import torch as th
from hok_game.nature_client import NatureClient
import time
import zmq
import sys
import logging
from hok_game.agent.actor import get_action
from hok_game.client.gamecore_controller import GameCoreController
from hok_game.protocol.python.sgame_ai_server_pb2 import StepFrameRsp, FightStartRsp, FightOverReq, FightOverRsp, StepFrameReq
from hok_game.conf.config import NC_CONFIG, GC_CONFIG

# actor
from hok_game.protocol.python.command_pb2 import AICommandInfo
from hok_game.protocol.python.command_pb2 import *
from hok_game.protocol.python.common_pb2 import *
from hok_game.conf.config import NC_CONFIG
import random
import time
import math


class HokEnv(MultiAgentEnv, NatureClient):
    def __init__(
        self,
        map_name='hok',
        num_agents=5,
        time_step=0,
        seed=123,
        episode_limit=150,
        #episode_limit=300,
        client_id=f"debug-train",
        logger=None,
        max_frame=60000
    ):
        super().__init__(client_id ,logger, max_frame)

        self.client_id=client_id
        self.map_name = map_name
        self.n_agents = num_agents
        #self.time_step = time_step
        self.seed = seed
        self.episode_limit = episode_limit
        self._episode_steps = 0 # 这是代表第几个episode
        self.action_space = 13 # up down left right attack + 4 技能
        self.observation_space = 6 # 4 # every unit location (x,z) + Monster location (xm,zm) = [x,z,agenthp,xm,zm,mHP]

        self.n_actions = self.action_space

        # self.agents = {}
        # self.enemies = {}   

        self.monster_hp_total = [] # 存储当前episode的暴龙血量,直至最后一个step
        self.obs = None
        self.state = None

        self.pos_delta = 1500
        
        self.SKILL_TYPE = ["obj_skill", "dir_skill", "pos_skill", "talent_skill"] # 动作施展方式

        # 用于输出结果,胜率
        # self.battles_won = 0
        # self.battles_game = 0

    def get_obs(self):
        """Returns all agent observations in a list.
            启动游戏拿到当前帧的信息
            所有Agent的Obs是(5,6)的张量
            先拿到当前帧的画面,然后读取坐标信息
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        '''return the size of the observation'''
        return self.observation_space

    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        ## 现在加入技能之后,并不是所有的技能都能使用,5678技能键要判断calm_down==0,如果为0才是1
        all_actor_legal_skill = []
        for hero in self.heroes:
            if hero.actor_id == 6: # 暴龙直接退出
                break 
            hero_legal_skill = [1,1,1,1]
            # hero就是当前的英雄,只选取合法的技能
            cur_legal_skill = [1 if slot.cooldown==0 else 0 for slot in hero.skill_state.slot_states]
            hero_legal_skill.extend(cur_legal_skill)
            hero_legal_skill.extend([1,1,1,1])
            all_actor_legal_skill.append(hero_legal_skill)
        
        return all_actor_legal_skill
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.action_space
        
    def reset(self, if_test=False, args=None, cur_time=None):
        """Reset the environment. Required after each full episode.
           Returns initial observations and states. 
           Starting game in every episode phase of sampling
        """
        # self.battles_won = 0
        # self.battles_game = 0

        self.monster_hp_total = [] 

        self._episode_steps = 0
        self.info_re = {'win_rate': 0., 'test_win_rate': 0., 'battle_won': None, 'monster_last_hp': None}

        # 启动游戏并返回初始游戏帧self.obs
        # start game
        self.start_point = time.time()
        if if_test:
            self.game_id = f"{args.name}-hok-test-{cur_time}" # 这里可以写执行的第几次训练。
        else:
            self.game_id = f"{args.name}-hok-train-{cur_time}"

        self.logger.info('New episode game starting!')
        self.reset_game()
        # run game just one time for getting original observation
        '''
            NatureClient 主循环, 负责运行游戏并输出统计信息
            - 首先, 调用 start_game 开始游戏.
            - 然后, 进入一个主循环,直到游戏结束. 在每次循环中, 函数会调用 on_update 方法来更新游戏状态.\n
            - 当游戏结束时, 函数会输出一些统计信息, 并调用stop_game方法停止游戏.
        '''
        
        t = time.time()
        ret = self.controller.start_game()
        if not ret:
            self.logger.error("start_game failed")
            try:
                ret = self.controller.start_game()
                if not ret:
                    raise Exception("start game failed even after retry")
            except Exception as e:
                self.logger.error(str(e))

        self.logger.debug(f"Nature client start_game time cost = {time.time() - t} s")

        '''先进行首帧处理,拿到初始状态obs,接下来通过step来获取下一步的交互动作
        '''
        # 更新时间
        ti = time.time() - self.start_point
        self.step_time_cost+=ti # 运行一个episode游戏总共花费多长时间(包括启动时间)

        # on_update() 拆开,只运行初始帧
        self.message_name, self.message_proto = self.recv_request()
        if self.message_name == "FightStartReq":
            self.rsp = FightStartRsp()
            #self.send_response(self.rsp)

        elif self.message_name == "StepFrameReq":
            self.rsp = StepFrameRsp()
            # Parse step frame request from game core
            self._parse_frame_info(self.message_proto) # 对于初始状态我只需要解析当前传过来的消息即可
            #self._print_debug_log(freq=1)

        elif self.message_name == "FightOverReq":
            self.rsp = FightOverRsp()
            gameover_state = self.message_proto.gameover_state
            self.game_over = True
            
            # 通过 game core 返回的状态码来判断游戏结束的状态
            if gameover_state == 1:
                self.game_status = NC_CONFIG["game_status"]["win"]
            elif gameover_state == 2:
                self.game_status = NC_CONFIG["game_status"]["fail"]
            elif gameover_state == 4:
                self.game_status = NC_CONFIG["game_status"]["error"]
            #self.send_response(self.rsp)
        
        else:
            self.logger.warning("Warning: receiving message fails")
            
        # 定义初始帧 state=(5, 6)的矩阵,5智能体,4为[x1,z1,agenthp,xm,zm,monsterhp]
        '''
            首帧获取信息,拿到gamecore返回的三个信息
            self.hero_hp: {'1': 16772, '2': 5706, '3': 8409, '4': 8743, '5': 9885}
            self.monster_hp: 30000 (某个int数值)
            self.agent_loc: [[   842   2956  agenthp 14501  -3754 monsterhp]
                             [   861  -1095  agenthp 14501  -3754 monsterhp]
                             [   769  -4468  agenthp 14501  -3754 monsterhp]
                             [   639  -7465  agenthp 14501  -3754 monsterhp]
                             [   515 -10797  agenthp 14501  -3754 monsterhp]] np.array()
        '''

        self.start_point = time.time()
        
        # 大龙的血量要记录,用于计算reward
        self.monster_hp_total.append(self.monster_hp)

        self.obs = self.agent_loc_np 
        # normalize [0.1]
        # min_val = np.min(self.obs)
        # max_val = np.max(self.obs)
        # self.obs = (self.obs-min_val) / (max_val-min_val)
        
        # normalize [-1.1]
        self.obs = self.obs - np.mean(self.obs)
        self.obs = self.obs / np.max(np.abs(self.obs))

        #print(f'self.obs {self.obs}')

        return self.get_obs(), self.get_state()

    def get_curr_reward(self):
        '''
            根据暴龙的血量返回奖励
            先读取到当前帧的暴龙血量为self.curr_monster_hp和上一帧的暴龙血量self.last_monster_hp
            或者设一个列表存储整个episode的Monster HP 变化
            当前帧的奖励reward = (curr_monster_hp - last_monster_hp) * -0.01
        '''
        
        return (self.monster_hp_total[-1] - self.monster_hp_total[-2]) * -0.01

    def position_change(self, agent_id, act):
        '''用于计算偏移量即
           act=0就是沿x上移Δ=10的偏移量
           act=1就是沿x下移Δ=10的偏移量
           act=2就是沿z上移Δ=10的偏移量
           act=3就是沿z下移Δ=10的偏移量
        '''
        # self.pos_delta = 10
        # self.agent_loc_np 是上一次保存下来的状态信息
        dst_pos=VInt3()
        last_loc_x = self.agent_loc_np[agent_id][0] # 拿到上一轮的x坐标位置
        last_loc_z = self.agent_loc_np[agent_id][1] # 拿到上一轮的z坐标位置
        if act==0:
            dst_pos.x = last_loc_x + self.pos_delta # 新的坐标是（x+10, z)
            dst_pos.z = last_loc_z
            dst_pos.y = 100

        elif act==1:
            dst_pos.x = last_loc_x - self.pos_delta # 新的坐标是（x-10, z)
            dst_pos.z = last_loc_z
            dst_pos.y = 100

        elif act==2:
            dst_pos.x = last_loc_x # 新的坐标是（x, z+10)
            dst_pos.z = last_loc_z + self.pos_delta
            dst_pos.y = 100

        elif act==3:
            dst_pos.x = last_loc_x # 新的坐标是（x, z-10)
            dst_pos.z = last_loc_z - self.pos_delta
            dst_pos.y = 100
        
        # """细化移动动作，左上、右上、右下、左下"""
        elif act==9:
            # 左上
            dst_pos.x = last_loc_x + self.pos_delta
            dst_pos.z = last_loc_z - self.pos_delta
            dst_pos.y = 100
            
        elif act==10:
            # 右上
            dst_pos.x = last_loc_x + self.pos_delta
            dst_pos.z = last_loc_z + self.pos_delta
            dst_pos.y = 100
            
        elif act==11:
            # 右下
            dst_pos.x = last_loc_x - self.pos_delta
            dst_pos.z = last_loc_z + self.pos_delta
            dst_pos.y = 100
            
        elif act==12:
            # 左下
            dst_pos.x = last_loc_x - self.pos_delta
            dst_pos.z = last_loc_z - self.pos_delta
            dst_pos.y = 100
            
        # 边界处理
        if dst_pos.x > 28700:
            dst_pos.x = 28700
        elif dst_pos.x < -28700:
            dst_pos.x = -28700
        
        if dst_pos.z > 9000:
            dst_pos.z = 9000
        elif dst_pos.z < -16800:
            dst_pos.z = -16800
        
        return dst_pos

    def get_skill_id_by_slot_type(self, hero, slot_type):
        '''obj_skill 寻找unit坐标'''
        skill_id = [slot.configId for slot in hero.skill_state.slot_states if slot.slot_type == slot_type][0]
        return skill_id

    def random_position(self):
        dst_pos=VInt3()
        dst_pos.x=random.randint(-28700,28700)
        dst_pos.y=100
        dst_pos.z=random.randint(-16800,9000)
    
        return dst_pos
    
    def get_angle(self, point_1=None, point_2=None):
        """通过两个坐标，计算point_2在point_1的哪个角度
        """
        x1, y1 = point_1[0], point_1[1]
        x2, y2 = point_2[0], point_2[1]

        delta_x = x2 - x1
        delta_y = y2 - y1

        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        # 调整角度到 0 到 360 度范围内
        angle_deg = angle_deg % 360
        return angle_deg


    ## 技能定义
    def __obj_skill_command(self, target, hero, skill_slot_type):
        cmd_pkg = CmdPkg()
        objSkill = ObjSkill()
        objSkill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
        objSkill.actorID = target
        objSkill.slotType = skill_slot_type
        cmd_pkg.command_type = CommandType.COMMAND_TYPE_ObjSkill
        cmd_pkg.obj_skill.CopyFrom(objSkill)

        return cmd_pkg
    
    def __dir_skill_command(self, target, hero, skill_slot_type, angle=None):
        cmd_pkg = CmdPkg()
        skill = DirSkill()
        skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
        skill.actorID = target
        skill.slotType = skill_slot_type
        skill.degree = int(angle)
        cmd_pkg.command_type = CommandType.COMMAND_TYPE_DirSkill
        cmd_pkg.dir_skill.CopyFrom(skill)
        
        return cmd_pkg

    def __pos_skill_command(self, hero, skill_slot_type):
        cmd_pkg = CmdPkg()
        skill = PosSkill()
        skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
        skill.destPos.CopyFrom(self.random_position())
        skill.slotType = skill_slot_type
        cmd_pkg.command_type = CommandType.COMMAND_TYPE_PosSkill
        cmd_pkg.pos_skill.CopyFrom(skill)
        
        return cmd_pkg

    def __talent_skill_command(self, target):
        cmd_pkg = CmdPkg()
        skill = TalentSkill()
        skill.degree = 90
        skill.actorID = target
        cmd_pkg.command_type = CommandType.COMMAND_TYPE_TalentSkill
        cmd_pkg.talent_skill.CopyFrom(skill)
        
        return cmd_pkg

    def act_2_cmd(self, actions):
        ''' 
            actions应该是类似于 [2,3,1,4,1]的数值列表
            将动作转换为cmd指令,然后传入gamecore中
            [agent1_cmd, agent2_cmd, ... agent5_cmd]
            0: up
            1: down
            2: left
            3: right
            4: normal attack
            5: skill1 目前所有的1,2,3技能都是以obj_skill的形式施展出来
            6: skill2
            7: skill3
            8: skill4
            self.heroes: 包含了上一帧中所有英雄的信息,这里主要是基于上一帧用技能攻击暴龙
            
            actions = [a1, a2, a3, a4, a5] 分别对应5个智能体
            
            庄周:技能1(方向型), 技能2(自身释放型), 技能3(自身释放型)
            狄仁杰:技能1(方向型), 技能2(方向型), 技能3(方向型)
            貂蝉:技能1(方向型), 技能2(方向型), 技能3(自身释放型)
            孙悟空:技能1(自身释放型), 技能2(方向型), 技能3(自身释放型)
            曹操:技能1(方向型), 技能2(方向型), 技能3(自身释放型)
            
        '''
        cmd_list, stop_game = [], False
        skill_type = None
        target = 6
        for _id, act in enumerate(actions): # [0,1,2,3,4]
            '''actor_id = _id+1'''
            cmd_pkg = CmdPkg()
            if _id == 0:
                """庄周"""
                if act == 4:
                    target = 6
                elif act == 5:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 6:
                    skill_type = 'obj_skill'
                    target = _id+1  
                elif act == 7:
                    skill_type = 'obj_skill'
                    target = _id+1  
                elif act == 8:
                    skill_type = 'talent_skill'
                    target = _id+1  
            elif _id == 1:
                """狄仁杰"""
                if act == 4:
                    target = 6
                elif act == 5:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 6:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 7:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 8:
                    skill_type = 'talent_skill'
                    target = _id+1
            elif _id == 2:
                """貂蝉"""
                if act == 4:
                    target = 6
                elif act == 5:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 6:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 7:
                    skill_type = 'obj_skill'
                    target = _id+1
                elif act == 8:
                    skill_type = 'talent_skill'
                    target = _id+1
            elif _id == 3:
                """孙悟空"""
                if act == 4:
                    target = 6
                elif act == 5:
                    skill_type = 'obj_skill'
                    target = _id+1
                elif act == 6:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 7:
                    skill_type = 'obj_skill'
                    target = _id+1
                elif act == 8:
                    skill_type = 'talent_skill'
                    target = _id+1
            elif _id == 4:
                """曹操"""
                if act == 4:
                    target = 6
                elif act == 5:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 6:
                    skill_type = 'dir_skill'
                    target = 6
                elif act == 7:
                    skill_type = 'obj_skill'
                    target = _id+1
                elif act == 8:
                    skill_type = 'talent_skill'
                    target = _id+1
            
            hero = self.heroes[_id]
            # 计算暴龙此刻在英雄的哪个角度，通过两个坐标
            hero_x = self.agent_loc_np[_id][0]
            hero_z = self.agent_loc_np[_id][1]
            hero_point = [hero_x, hero_z]

            monster_x = self.agent_loc_np[_id][3]
            monster_z = self.agent_loc_np[_id][4]
            monster_point = [monster_x, monster_z]

            angle_ = self.get_angle(hero_point, monster_point)

            if act == 4:
                # 普攻
                attack = AttackCommon()
                attack.actorID = target
                attack.start = 1
                cmd_pkg.command_type = CommandType.COMMAND_TYPE_AttackCommon
                cmd_pkg.attack_common.CopyFrom(attack)
            
            ## 加入4个英雄技能
            elif act == 5:
                skill_slot_type = 1 # 1技能
                if skill_type == "obj_skill":
                    cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
                elif skill_type == "dir_skill":
                    cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type, angle=angle_)
                elif skill_type == "pos_skill":
                    cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
                elif skill_type == "talent_skill":
                    cmd_pkg = self.__talent_skill_command(target)

            elif act == 6:
                skill_slot_type = 2 # 2技能
                if skill_type == "obj_skill":
                    cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
                elif skill_type == "dir_skill":
                    cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type, angle=angle_)
                elif skill_type == "pos_skill":
                    cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
                elif skill_type == "talent_skill":
                    cmd_pkg = self.__talent_skill_command(target)

            elif act == 7:
                skill_slot_type = 3 # 3技能
                if skill_type == "obj_skill":
                    cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
                elif skill_type == "dir_skill":
                    cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type, angle=angle_)
                elif skill_type == "pos_skill":
                    cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
                elif skill_type == "talent_skill":
                    cmd_pkg = self.__talent_skill_command(target)
            
            elif act == 8: 
                cmd_pkg = self.__talent_skill_command(target)
            
            else:
                # 0上 1下 2左 3右
                move_pos = MoveToPos()
                move_pos.destPos.CopyFrom(self.position_change(_id, act))
                cmd_pkg.command_type = CommandType.COMMAND_TYPE_MovePos
                cmd_pkg.move_pos.CopyFrom(move_pos)


            cmd = AICommandInfo(
                actor_id=int(_id+1),
                cmd_info=cmd_pkg
            )
            cmd_list.append(cmd)

        return cmd_list, stop_game

    def step(self, _actions, if_test=False):
        """ Returns reward, terminated, info """
        if th.is_tensor(_actions):
            #actions = _actions.cpu.numpy()
            actions = [int(a) for a in _actions]
        else:
            actions = _actions

        self._episode_steps+=1
        print(f'当前帧:{self._episode_steps},分别执行了哪些动作: {_actions}')
        # 开始每帧进行交互
        # ti总是代表当前时间 - 上一个记录节点
        ti = time.time() - self.start_point
        self.step_time_cost+=ti
        self.start_point = time.time()
        self.timestep+=1
        
        # 每帧训练,先给对面动作,看对面反馈,因为reset()已经拿到初始信息了,现在做的动作相当于基于初始状态下做的动作
        if self.message_name == "FightStartReq":
            self.send_response(self.rsp)

        elif self.message_name == "StepFrameReq":
            # get actions from agent 
            cmd_list, stop_game = self.act_2_cmd(actions) # 要重新编辑

            if stop_game:
                self.game_over = True
                self.game_status = NC_CONFIG["game_status"]["error"]
                self.rsp.gameover_ai_server = 1
                self.controller.stop_game()
                self.logger.info("Send game over request to game core")
                

            # senf action cmd
            # send step frame response
            self.rsp.cmd_list.extend(cmd_list)
            self.send_response(self.rsp)

        elif self.message_name == "FightOverReq":
            self.send_response(self.rsp)

        reward = 0.
        terminated = False
        if (self._episode_steps >= self.episode_limit) or \
            ((self._episode_steps < self.episode_limit) and self.game_status==1):
            '''两种结束条件:
                1.运行steps超过了episode_limit,此时应该结束条件,并判断游戏状态
                2.运行step未超过episode_limit,但是获胜了
            '''
            terminated = True # 如果超过了每个episode的步数,即使没有训练完也要停止本episode的采用
            ## 关闭本次游戏连接
            # self.stop()
            self.game_over = True
            if (self._episode_steps >= self.episode_limit):
                self.game_status = NC_CONFIG["game_status"]["overtime"]

        # receive next step obs, and interact with the gamecore again 
        if not self.game_over:
            #print('nextnextnext')
            # 更新时间
            ti = time.time() - self.start_point
            self.step_time_cost+=ti # 运行一个episode游戏总共花费多长时间

            # on_update() 拆开,只运行初始帧
            self.message_name, self.message_proto = self.recv_request()
            #print('received next states!!!!!!')
            if self.message_name == "FightStartReq":
                self.rsp = FightStartRsp()
                #self.send_response(self.rsp)

            elif self.message_name == "StepFrameReq":
                self.rsp = StepFrameRsp()
                # Parse step frame request from game core
                self._parse_frame_info(self.message_proto) # 对于初始状态我只需要解析当前传过来的消息即可
                self._print_debug_log(freq=1)

            elif self.message_name == "FightOverReq":
                self.rsp = FightOverRsp()
                gameover_state = self.message_proto.gameover_state
                self.game_over = True
                
                # 通过 game core 返回的状态码来判断游戏结束的状态
                if gameover_state == 1:
                    self.game_status = NC_CONFIG["game_status"]["win"]
                elif gameover_state == 2:
                    self.game_status = NC_CONFIG["game_status"]["fail"]
                elif gameover_state == 4:
                    self.game_status = NC_CONFIG["game_status"]["error"]

                #self.send_response(self.rsp)
        

            self.start_point = time.time()

            # 大龙的血量要记录,用于计算reward
            self.monster_hp_total.append(self.monster_hp)
            
            self.obs = self.agent_loc_np # 这是下一轮的obs, self.obs 赋予新一帧的观测
            # normalize [-1,1]
            self.obs = self.obs - np.mean(self.obs)
            self.obs = self.obs / np.max(np.abs(self.obs))
            reward = self.get_curr_reward()
            terminated = False

        else:
            '''表示游戏已经结束了,但是我还要拿到最后一帧画面（最后一帧画面没什么用,其实不拿也行）
                同时我需要拿到game status
            '''
            # 存储最后一帧的暴龙血量
            # 按理说应该只会出现在最后一轮
            avg_time = self.step_time_cost / self.timestep
            self.logger.info(f"******* Game Over with status {self.game_status} *******")
            self.logger.info("0: error, 1: win, 2: fail, 3: overtime")
            self.logger.info(
                f"FrameNo = [{self.frame_no}], StepNo = [{self.timestep}], avg time = [{avg_time}]")
            
            self.controller.stop_game()
            
            # self.battles_game+=1

            ## 记录结果:battle_won: 
            if self.game_status == 1:
                #self.battles_won+=1
                self.info_re['battle_won'] = True

            elif self.game_status == 2 or 3:
                self.info_re['battle_won'] = False
                
            self.info_re['monster_last_hp'] = self.monster_hp_total[-1]
            
        return reward, terminated, self.info_re

    def save_replay(self):
        pass

    def render(self):
        pass

    def close(self):
        '''close honor of kings game'''
        pass

    def seed(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(), # 20=4*5
                    "obs_shape": self.get_obs_size(), # 4
                    "n_actions": self.get_total_actions(), # 5
                    "n_agents": self.n_agents, # 5
                    "episode_limit": self.episode_limit} # 这个episode_limit可能是buffer的存储时间长度？？
        return env_info
    
    def get_stats(self):
        stats={
            # "battles_won": self.battles_won,
            # "battles_game": self.battles_game,
            #"battles_draw": self.timeouts,
            # "win_rate": self.battles_won / self.battles_game,
            #"timeouts": self.timeouts,
            #"restarts": self.force_restarts,
        }
        
        return stats