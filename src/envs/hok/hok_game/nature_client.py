#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import path, sys
from pathlib import Path
folder = path.Path(__file__).abspath()
current_dir = Path(__file__).parent ## 获得当前目录路径
sys.path.append(str(current_dir))# 如果使用相对路径，并且添加当前目录的上两级目录

import zmq
import time
import sys
import logging
from agent.actor import get_action
from client.gamecore_controller import GameCoreController
from protocol.python.sgame_ai_server_pb2 import StepFrameRsp, FightStartRsp, FightOverReq, FightOverRsp, StepFrameReq
from conf.config import NC_CONFIG, GC_CONFIG


class NatureClient:
    """
    NatureClient is a middleware that bridges AI Server and Game Core:
        - receives frame information from game core and send it to ai server
        - receives command from ai server and send it back to game core
    """

    def __init__(self, client_id, logger=None, max_frame=20000):

        super(NatureClient, self).__init__()

        # 日志相关
        if logger is None:
            self.logger = logging
            logging.basicConfig(format='%(levelname)s - %(module)s - %(message)s', level=logging.DEBUG)

        # 通信相关
        self.battlesrv_port = GC_CONFIG["battlesrv_port"]
        self.connect_to_gamecore(self.battlesrv_port)
        
        # 游戏相关
        self.client_id = client_id
        self.game_id = self.client_id
        self.max_frame = max_frame

    # 与 gamecore 建立端口连接
    def connect_to_gamecore(self, battlesrv_port):
        self.context = zmq.Context()
        self.GCconn = self.context.socket(zmq.REP)
        if not battlesrv_port:
            self.battlesrv_port = GC_CONFIG["battlesrv_port"]
        else:
            self.battlesrv_port = battlesrv_port
        endpoint = f"tcp://*:{self.battlesrv_port}"
        try:
            self.GCconn.bind(endpoint)
            self.logger.info(f"Socket connect to gamecore: {endpoint} ... Succeed")
            return True
        except Exception:
            self.logger.error(f"Socket connect to gamecore: {endpoint} ... Failed")
            
            return False
    
    def reset_game(self):
        """
        游戏重置，开启新的一局游戏前的必要操作
            - 重置游戏信息
            - 重置游戏数据
            - 初始化 GameCoreController
        """
        self.logger.info(f"Game reset ... New game id = {self.game_id}")
        
        # 游戏信息重置
        self.game_over = False
        self.frame_no = 0
        self.game_status = NC_CONFIG["game_status"]["pending"]
        self.monster_hp = 0
        self.hero_hp = {}
        self.agent_loc = {}
        for key, _ in NC_CONFIG["hero_id"].items():
            self.hero_hp[key] = 0
            self.agent_loc[key] = []

        self.agent_loc['6'] = []
        
            
        # 记录发送的请求数目和时延
        self.timestep = 1
        self.step_time_cost = 0.0
        
        # 初始化GameCoreController
        self.controller = GameCoreController(self.game_id, self.logger, self.battlesrv_port)
        
        return True
    
    def stop(self):
        # self.GCconn.setsockopt(zmq.LINGER,0)
        self.GCconn.close()
        self.context.term()
        self.logger.info("Socket connection closed")
        print("Socket connection closed")
        # sys.exit()

        
    def run(self):
        """
        NatureClient 主循环, 负责运行游戏并输出统计信息
        - 首先, 调用 start_game 开始游戏.
        - 然后, 进入一个主循环，直到游戏结束. 在每次循环中, 函数会调用 on_update 方法来更新游戏状态.\n
        - 当游戏结束时, 函数会输出一些统计信息, 并调用stop_game方法停止游戏.
        """
        # 通过 GameCoreController 开始游戏
        t = time.time()
        ret = self.controller.start_game()
        if not ret:
            self.logger.error("start_game failed")
            return False

        self.logger.debug(f"Nature client start_game time cost = {time.time() - t} s")
        
        # 游戏主循环，直到游戏结束
        while not self.game_over: # 这是一个回合(episode)的训练，当大龙被打败的时候，self.game_over=False
            # 更新时间
            ti = time.time() - t
            t = time.time()
            
            self.on_update()

            if not self.game_over:
                self.timestep += 1
            if self.timestep > 1:
                self.step_time_cost += ti

            # 按照一定数量打印日志, 减少日志输出
            if self.timestep % 100 == 0:
                self.logger.info(
                    f"Game on updating: FrameNo = [{self.frame_no}], StepNo = [{self.timestep}]")

        # 游戏结束，输出统计信息
        if self.game_over:            
            avg_time = self.step_time_cost / self.timestep
            self.logger.info(f"******* Game Over with status {self.game_status} *******")
            self.logger.info("0: error, 1: win, 2: fail, 3: overtime")
            self.logger.info(
                f"FrameNo = [{self.frame_no}], StepNo = [{self.timestep}], avg time = [{avg_time}]")
            self.controller.stop_game()

        return True

    def on_update(self):
        """
        主循环里每一帧的处理, 主要是更新游戏状态, 并与 Agent 交互获取下一步的动作.
        - 首先, 从 game core 接收消息, 按照消息类型进行处理.
            - 如果是 FightStartReq, 则发送 FightStartRsp 给 game core.
            - 如果是 StepFrameReq, 则解析消息, 更新游戏状态, 并发送 StepFrameRsp 给 game core.
            - 如果是 FightOverReq, 则判断游戏结束的状态, 并发送 FightOverRsp 给 game core.
        - 在获取到下一步的行动后, 将响应发送回 game core.
        """
        # Receive message from game core
        message_name, message_proto = self.__recv_request()

        '''
        message_name: ...
        message_proto: 
            frame_state {
            frameNo: 1 # 这个指标是训练的轮数?
            hero_states {
                actor_id: 1
                actor_state {
                config_id: 17101
                runtime_id: 1
                actor_type: ACTOR_HERO
                camp: PlayerCamp_1
                behav_mode: State_Idle
                location {
                    x: 842
                    y: 100
                    z: 2956
                }
                hp: 16772
                max_hp: 16772
                values {
                    phy_atk: 293
                    phy_def: 1055
                    mgc_atk: 0
                    mgc_def: 472
                    mov_spd: 4111
                    atk_spd: 1402
                    ep: 1
                    max_ep: 100
                    hp_recover: 402
                    ep_recover: 5
                    phy_armor_hurt: 0
                    mgc_armor_hurt: 0
                    crit_rate: 0
                    crit_effe: 10000
                    phy_vamp: 0
                    mgc_vamp: 0
                    cd_reduce: 0
                    ctrl_reduce: 0
                }
                attack_range: 3000
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 17100
                    slot_type: SLOT_SKILL_0
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 877
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 3000
                }
                slot_states {
                    configId: 17110
                    slot_type: SLOT_SKILL_1
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 6000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 4000
                }
                slot_states {
                    configId: 17120
                    slot_type: SLOT_SKILL_2
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 10000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 6500
                }
                slot_states {
                    configId: 17130
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: false
                    cooldown: 0
                    cooldown_max: 40000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 10000
                }
                slot_states {
                    configId: 80110
                    slot_type: SLOT_SKILL_6
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 60000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            hero_states {
                actor_id: 2
                actor_state {
                config_id: 13301
                runtime_id: 2
                actor_type: ACTOR_HERO
                camp: PlayerCamp_1
                behav_mode: State_Idle
                location {
                    x: 861
                    y: 100
                    z: -1095
                }
                hp: 5706
                max_hp: 5706
                values {
                    phy_atk: 770
                    phy_def: 333
                    mgc_atk: 0
                    mgc_def: 162
                    mov_spd: 3895
                    atk_spd: 15800
                    ep: 1770
                    max_ep: 1770
                    hp_recover: 54
                    ep_recover: 29
                    phy_armor_hurt: 126
                    mgc_armor_hurt: 0
                    crit_rate: 4500
                    crit_effe: 10000
                    phy_vamp: 3536
                    mgc_vamp: 0
                    cd_reduce: 0
                    ctrl_reduce: 0
                }
                attack_range: 8000
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 13303
                    slot_type: SLOT_SKILL_0
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 387
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 8000
                }
                slot_states {
                    configId: 13310
                    slot_type: SLOT_SKILL_1
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 6000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 10000
                }
                slot_states {
                    configId: 13320
                    slot_type: SLOT_SKILL_2
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 12000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 9000
                }
                slot_states {
                    configId: 13330
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 16000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 10000
                }
                slot_states {
                    configId: 80110
                    slot_type: SLOT_SKILL_6
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 60000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            hero_states {
                actor_id: 3
                actor_state {
                config_id: 14101
                runtime_id: 3
                actor_type: ACTOR_HERO
                camp: PlayerCamp_1
                behav_mode: State_Idle
                location {
                    x: 769
                    y: 100
                    z: -4468
                }
                hp: 8409
                max_hp: 8409
                values {
                    phy_atk: 279
                    phy_def: 685
                    mgc_atk: 666
                    mgc_def: 402
                    mov_spd: 3500
                    atk_spd: 1400
                    ep: 2460
                    max_ep: 2460
                    hp_recover: 171
                    ep_recover: 31
                    phy_armor_hurt: 0
                    mgc_armor_hurt: 42
                    crit_rate: 0
                    crit_effe: 10000
                    phy_vamp: 0
                    mgc_vamp: 0
                    cd_reduce: 5010
                    ctrl_reduce: 0
                }
                attack_range: 6000
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 14100
                    slot_type: SLOT_SKILL_0
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 877
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 6000
                }
                slot_states {
                    configId: 14110
                    slot_type: SLOT_SKILL_1
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 2495
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 8800
                }
                slot_states {
                    configId: 14120
                    slot_type: SLOT_SKILL_2
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 4990
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 4500
                }
                slot_states {
                    configId: 14130
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 14970
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 6000
                }
                slot_states {
                    configId: 80110
                    slot_type: SLOT_SKILL_6
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 60000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            hero_states {
                actor_id: 4
                actor_state {
                config_id: 16701
                runtime_id: 4
                actor_type: ACTOR_HERO
                camp: PlayerCamp_1
                behav_mode: State_Idle
                location {
                    x: 639
                    y: 100
                    z: -7465
                }
                hp: 8743
                max_hp: 8743
                values {
                    phy_atk: 905
                    phy_def: 374
                    mgc_atk: 0
                    mgc_def: 162
                    mov_spd: 3800
                    atk_spd: 3900
                    ep: 2160
                    max_ep: 2160
                    hp_recover: 78
                    ep_recover: 29
                    phy_armor_hurt: 126
                    mgc_armor_hurt: 0
                    crit_rate: 6527
                    crit_effe: 5042
                    phy_vamp: 0
                    mgc_vamp: 0
                    cd_reduce: 2500
                    ctrl_reduce: 0
                }
                attack_range: 3000
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 16700
                    slot_type: SLOT_SKILL_0
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 719
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 3000
                }
                slot_states {
                    configId: 16710
                    slot_type: SLOT_SKILL_1
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 7500
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 3000
                }
                slot_states {
                    configId: 16720
                    slot_type: SLOT_SKILL_2
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 3750
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 1
                }
                slot_states {
                    configId: 16730
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 22500
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 4000
                }
                slot_states {
                    configId: 80110
                    slot_type: SLOT_SKILL_6
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 60000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            hero_states {
                actor_id: 5
                actor_state {
                config_id: 12801
                runtime_id: 5
                actor_type: ACTOR_HERO
                camp: PlayerCamp_1
                behav_mode: State_Idle
                location {
                    x: 515
                    y: 100
                    z: -10797
                }
                hp: 9885
                max_hp: 9885
                values {
                    phy_atk: 672
                    phy_def: 1172
                    mgc_atk: 0
                    mgc_def: 512
                    mov_spd: 3807
                    atk_spd: 1423
                    ep: 0
                    max_ep: 0
                    hp_recover: 196
                    ep_recover: 0
                    phy_armor_hurt: 126
                    mgc_armor_hurt: 0
                    crit_rate: 0
                    crit_effe: 10000
                    phy_vamp: 0
                    mgc_vamp: 0
                    cd_reduce: 3500
                    ctrl_reduce: 0
                }
                attack_range: 2800
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 12800
                    slot_type: SLOT_SKILL_0
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 875
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 2800
                }
                slot_states {
                    configId: 12810
                    slot_type: SLOT_SKILL_1
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 4550
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                slot_states {
                    configId: 12820
                    slot_type: SLOT_SKILL_2
                    level: 6
                    usable: true
                    cooldown: 0
                    cooldown_max: 4550
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 8000
                }
                slot_states {
                    configId: 12830
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 19500
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 4000
                }
                slot_states {
                    configId: 80110
                    slot_type: SLOT_SKILL_6
                    level: 1
                    usable: true
                    cooldown: 0
                    cooldown_max: 60000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5000
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            hero_states {
                actor_id: 6
                actor_state {
                config_id: 12202
                runtime_id: 6
                actor_type: ACTOR_MONSTER
                camp: PlayerCamp_2
                location {
                    x: 14501
                    y: 100
                    z: -3754
                }
                hp: 30000
                max_hp: 30000
                values {
                    phy_atk: 600
                    phy_def: 1500
                    mgc_atk: 204
                    mgc_def: 1500
                    mov_spd: 3500
                    atk_spd: 0
                    ep: 0
                    max_ep: 0
                    hp_recover: 0
                    ep_recover: 0
                    phy_armor_hurt: 0
                    mgc_armor_hurt: 0
                    crit_rate: 0
                    crit_effe: 0
                    phy_vamp: 0
                    mgc_vamp: 0
                    cd_reduce: 0
                    ctrl_reduce: 0
                }
                attack_range: 3000
                attack_target: 0
                }
                skill_state {
                slot_states {
                    configId: 21250
                    slot_type: SLOT_SKILL_0
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 0
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 3000
                }
                slot_states {
                    configId: 21253
                    slot_type: SLOT_SKILL_1
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 10000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 2500
                }
                slot_states {
                    configId: 21252
                    slot_type: SLOT_SKILL_2
                    level: 3
                    usable: true
                    cooldown: 0
                    cooldown_max: 12000
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 5500
                }
                slot_states {
                    configId: 0
                    slot_type: SLOT_SKILL_3
                    level: 3
                    usable: false
                    cooldown: 0
                    cooldown_max: 0
                    usedTimes: 0
                    hitHeroTimes: 0
                    attack_range: 100
                }
                }
                killCnt: 0
                deadCnt: 0
                totalHurtToHero: 0
                totalBeHurtByHero: 0
            }
            }
        '''
        if message_name == "FightStartReq":
            rsp = FightStartRsp()
            self.__send_response(rsp)

        elif message_name == "StepFrameReq":
            '''
                这个地方获取当前帧，然后得到状态和Obs，并通过get_actions输入到Agent网络中
                batch是EpisodeBatch
                self.heroes 包含英雄信息验证和英雄技能数据验证,具体包括:actor_id, actor_state, skill_state{slot_states(普通+技能*3+额外技能)}
            '''
            rsp = StepFrameRsp()
            # Parse step frame request from game core
            self._parse_frame_info(message_proto)
            
            self._print_debug_log(freq=100)

            # get action from agent
            cmd_list, stop_game = get_action(self.heroes)

            # actions = select_actions(batch, timestep) -> Q = forward(batch, step) -> batch: obs, action_onehot, agent_id, a_t
            
            # 设计状态, 奖励
            # reward = env.step(actions)

            # stop_game 为 True 表示 agent 主动结束游戏
            if stop_game:
                self.game_over = True
                self.game_status = NC_CONFIG["game_status"]["error"]
                rsp.gameover_ai_server = 1
                self.logger.info("Send game over request to game core")
        
            # send step frame response
            rsp.cmd_list.extend(cmd_list)
            self.__send_response(rsp)

        elif message_name == "FightOverReq":
            rsp = FightOverRsp()
            
            gameover_state = message_proto.gameover_state
            self.game_over = True
            
            # 通过 game core 返回的状态码来判断游戏结束的状态
            if gameover_state == 1:
                self.game_status = NC_CONFIG["game_status"]["win"]
            elif gameover_state == 2:
                self.game_status = NC_CONFIG["game_status"]["fail"]
            elif gameover_state == 4:
                self.game_status = NC_CONFIG["game_status"]["error"]
            
            self.__send_response(rsp)

        else:
            self.logger.warning("Warning: receiving message fails")

    def _parse_frame_info(self, msg_proto):
        """
        - parse step frame request from game core
        - check game over and update game status
        """
        frame_state = msg_proto.frame_state
        # self.logger.error(f"FrameState is {frame_state}")
        self.frame_no = frame_state.frameNo
        self.heroes = frame_state.hero_states                                   # 这个self.heros里边包含了所有智能体的信息,具体参考on_update()的注释部分
        self.game_status = NC_CONFIG["game_status"]["running"]
        for hero in self.heroes:
            # 解析英雄血量和暴君血量
            actor_type = hero.actor_state.actor_type
            actor_hp = hero.actor_state.hp
            if actor_type == 1:
                self.monster_hp = actor_hp
            elif actor_type == 0:
                hero_id = str(hero.actor_id)
                if hero_id not in self.hero_hp:
                    self.logger.error(f"hero_id {hero_id} not in hero_hp")
                    #continue
                self.hero_hp[hero_id] = actor_hp

            # 解析英雄位置和暴君位置
            actor_id = str(hero.actor_id)
            actor_loc = hero.actor_state.location
            cur_loc = []
            pos_x = actor_loc.x
            cur_loc.append(pos_x)
            pos_y = actor_loc.y
            cur_loc.append(pos_y)
            pos_z = actor_loc.z
            cur_loc.append(pos_z)
            self.agent_loc[actor_id] = cur_loc
            
            # if actor_type != 1:
            #     # 打印当前帧英雄的最大血量和恢复血量
            #     print(f'当前帧 {self.frame_no}, 英雄 agent id:{hero.actor_id}, hp:{hero.actor_state.hp}, \
            #         max_hp: {hero.actor_state.max_hp}, hp_recover: {hero.actor_state.values.hp_recover} \n \
            #             技能 1 冷却信息: {hero.skill_state.slot_states[0].cooldown}, 技能 2 冷却信息: {hero.skill_state.slot_states[1].cooldown}, 技能 3 冷却信息: {hero.skill_state.slot_states[2].cooldown} \
            #                 技能 4 冷却信息: {hero.skill_state.slot_states[3].cooldown}, 技能 5 冷却信息: {hero.skill_state.slot_states[4].cooldown}')
            # else:
            #     print(f'当前帧 {self.frame_no}, 暴龙 hp:{hero.actor_state.hp}, \
            #         max_hp: {hero.actor_state.max_hp}, hp_recover: {hero.actor_state.values.hp_recover}')
                
        import numpy as np 
        import torch as th
        self.agent_loc_np = []
        for i in range(1,6):
            hero_obs = []
            hero_obs.append(self.agent_loc[f'{i}'][0])
            hero_obs.append(self.agent_loc[f'{i}'][2])
            # 加入英雄血量
            hero_obs.append(self.hero_hp[f'{i}'])
            hero_obs.append(self.agent_loc['6'][0])
            hero_obs.append(self.agent_loc['6'][2])
            # 加入暴龙血量
            hero_obs.append(self.monster_hp)
            self.agent_loc_np.append(hero_obs)
        self.agent_loc_np = np.array(self.agent_loc_np)

        
        ## self.agent_loc: {'1': [842, 100, 2956], '2': [861, 100, -1095], '3': [769, 100, -4468], '4': [639, 100, -7465], '5': [515, 100, -10797], '6': [14501, 100, -3754]} 
        
        ## self.agent_hp_info: {'1': [max_hp, hp_recover], ...}
        
        # 此处的最大帧数是 client 自己设置的, 用于单局的超时控制
        if self.frame_no > self.max_frame:
            self.game_over = True
            self.game_status = NC_CONFIG["game_status"]["overtime"]

    def _print_debug_log(self, freq):
        if self.timestep % freq == 0:
            self.logger.debug(f"-------------------------------------------")
            self.logger.debug(
                f"FrameNo = [{self.frame_no}], StepNo = [{self.timestep}]")
            self.logger.debug(f"Game Status = {self.game_status}")
            self.logger.debug(f"Monster HP = {self.monster_hp}")
            self.logger.debug(f"Hero HP = {self.hero_hp}")

    # Zmq receive request from game core
    
    def recv_request(self):
        return self.__recv_request()

    def __recv_request(self):
        message_name = None
        req = None

        try:
            message_parts = self.GCconn.recv_multipart()
            message_name = message_parts[0].decode("utf-8")
            message_content = message_parts[1]

            if message_name == "StepFrameReq":
                req = StepFrameReq()
                req.ParseFromString(message_content)
            elif message_name == "FightOverReq":
                req = FightOverReq()
                req.ParseFromString(message_content)
            else:
                req = None

        except Exception as e:
            self.logger.error(f"Receiving request fails, err is {str(e)}")

        return message_name, req

    def send_response(self, rsp):
        self.__send_response(rsp=rsp)

    # Zmq send response to game core
    def __send_response(self, rsp):
        try:
            self.GCconn.send(rsp.SerializeToString())

        except Exception as e:
            self.logger.error(f"Sending response fails, err is {str(e)}")


if __name__ == "__main__":
#def game_start():
    '''执行每个episode的game'''
    i = 0
    game = NatureClient(f"debug-test")
    
    # 每执行一次就是一个episode
    # while True:
    i += 1
    t1 = time.time()

    game.game_id = f"debug-test-{i}"
    game.reset_game()

    ret = game.run()

    if not ret:
        game.logger.error(f"game.run() failed")
        game.stop()

    game.logger.debug(
        "Time cost: {} s".format(round(time.time() - t1)))
        # game.stop()
