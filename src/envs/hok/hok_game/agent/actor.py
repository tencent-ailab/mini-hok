#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from protocol.python.command_pb2 import AICommandInfo
from protocol.python.command_pb2 import *
from protocol.python.common_pb2 import *
from conf.config import NC_CONFIG
import random

#技能类型
SKILL_TYPE = ["obj_skill", "dir_skill", "pos_skill", "talent_skill"]


def get_action(heroes):
    """对每个英雄生成随机指令

    Args:
        heroes (list of HeroState): 所有英雄信息的列表

    Returns:
        cmd_list (list of AICommandInfo): 英雄指令的列表
        stop_game (bool): 智能体是否主动结束游戏

    =======================================================

    修改后的get_actions应该接收的是obs, a_t-1, agent_id输入到Agent网络中,输出Q值,根据Q值在探索阶段和训练阶段来拿到动作
    
    """
    cmd_list, stop_game = [], False
    for (key, _), hero in zip(NC_CONFIG["hero_id"].items(), heroes):
        '''
        cmd: 
            [actor_id: 1
            cmd_info {
            command_type: COMMAND_TYPE_DirSkill
            dir_skill {
                skillID: 17110
                actorID: 5
                slotType: SLOT_SKILL_1
                degree: 4
            }
            }, ... ]
        '''
        #print(f"curr actor: {key}"),
        cmd = AICommandInfo(
            actor_id=int(key),
            cmd_info=_get_random_command(hero, heroes)
        )

        cmd_list.append(cmd)

    return cmd_list, stop_game


def _get_random_command(hero, heroes):
    """获取单个英雄的随机指令
    Args:
        hero (HeroState): 单个英雄的信息
        heroes (list of HeroState): 所有英雄的信息的列表
    Returns:
        cmd_info (CmdPkg): 英雄指令信息
    """

    # 30% 概率英雄随机移动， 或者随机技能
    if 0 < random.random() < 0.8:
        return __random_move_cmd()
    else:
        # 随机获取一个可用的合法技能
        skill_slot_type = __get_random_skill(hero) # 先随机到一个合法的技能
        # print(f"hero is {hero}")
        # print(f"heros are {heroes}")
        # 随机选择一个合法的目标
        target = __get_random_target(hero.actor_id, heroes)

        ## 选择大龙为攻击目标 actor_id: 6
        #target = __get_specific_target(hero.actor_id, heroes)

        if skill_slot_type == 0:
            return __normal_attack_command(target)
        else:
            skill_type = random.choice(SKILL_TYPE)
            #skill_type = 'obj_skill'
            if skill_type == "obj_skill":
                return __obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                return __dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                return __pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                return __talent_skill_command(target)
          
            
def __get_random_skill(hero):
    legal_skill = [slot.slot_type for slot in hero.skill_state.slot_states if slot.cooldown == 0]
    return random.choice(legal_skill)


def get_skill_id_by_slot_type(hero, slot_type):
    skill_id = [slot.configId for slot in hero.skill_state.slot_states if slot.slot_type == slot_type][0]
    return skill_id
    
    
def __get_random_target(hero_id, heroes):
    legal_target_id = [target.actor_id for target in heroes if target.actor_id != hero_id]
    return random.choice(legal_target_id)

def __get_specific_target(hero_id, heroes):
    '''重新定义规则,使所有的英雄都对大龙发起进攻目标'''
    legal_target_id = 6
    return legal_target_id



def random_position():
    dst_pos=VInt3()
    dst_pos.x=random.randint(-28700,28700)
    dst_pos.y=100
    dst_pos.z=random.randint(-16800,9000)
    
    return dst_pos

def __obj_skill_command(target, hero, skill_slot_type):
    cmd_pkg = CmdPkg()
    objSkill = ObjSkill()
    objSkill.skillID = get_skill_id_by_slot_type(hero, skill_slot_type)
    objSkill.actorID = target
    objSkill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_ObjSkill
    cmd_pkg.obj_skill.CopyFrom(objSkill)
    
    return cmd_pkg

def __dir_skill_command(target, hero, skill_slot_type):
    cmd_pkg = CmdPkg()
    skill = DirSkill()
    skill.skillID = get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.actorID = target
    skill.slotType = skill_slot_type
    skill.degree = 4
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_DirSkill
    cmd_pkg.dir_skill.CopyFrom(skill)
    
    return cmd_pkg


def __pos_skill_command(hero, skill_slot_type):
    cmd_pkg = CmdPkg()
    skill = PosSkill()
    skill.skillID = get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.destPos.CopyFrom(random_position())
    skill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_PosSkill
    cmd_pkg.pos_skill.CopyFrom(skill)
    
    return cmd_pkg


def __talent_skill_command(target):
    cmd_pkg = CmdPkg()
    skill = TalentSkill()
    skill.degree = 90
    skill.actorID = target
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_TalentSkill
    cmd_pkg.talent_skill.CopyFrom(skill)
    
    return cmd_pkg
    

def __random_move_cmd():
    cmd_pkg = CmdPkg()
    move_pos = MoveToPos()
    move_pos.destPos.CopyFrom(random_position())
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_MovePos
    cmd_pkg.move_pos.CopyFrom(move_pos)
    return cmd_pkg


def __normal_attack_command(target):
    cmd_pkg = CmdPkg()
    attack = AttackCommon()
    attack.actorID = target
    attack.start = 1
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_AttackCommon
    cmd_pkg.attack_common.CopyFrom(attack)
    
    return cmd_pkg
    

