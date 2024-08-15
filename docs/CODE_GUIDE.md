
## Code Directory Introduction
```python
├── src
│   ├── components
│   │   ├── action_selectors.py
│   │   ├── episode_buffer.py
│   │   ├── epsilon_schedules.py
│   │   ├── __init__.py
│   │   ├── segment_tree.py
│   │   └── transforms.py
│   ├── config
│   │   ├── algs
│   │   │   └── vdn.yaml
│   │   ├── default.yaml
│   │   └── envs
│   │       └── hok.yaml
│   ├── controllers
│   │   ├── basic_controller.py
│   │   ├── __init__.py
│   │   └── n_controller.py
│   ├── envs
│   │   ├── hok
│   │   │   ├── hok_env.py
│   │   │   ├── hok_game
│   │   │   │   ├── agent
│   │   │   │   │   ├── actor.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── client
│   │   │   │   │   ├── gamecore_controller.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── conf
│   │   │   │   │   ├── config.py
│   │   │   │   │   ├── gamecore_conf.json
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── natureclient_conf.json
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nature_client.py
│   │   │   │   ├── protocol
│   │   │   │   │   ├── command.proto
│   │   │   │   │   ├── common.proto
│   │   │   │   │   ├── easy.proto
│   │   │   │   │   ├── hero.proto
│   │   │   │   │   ├── python
│   │   │   │   │   │   ├── build_py.sh
│   │   │   │   │   │   ├── command_pb2.py
│   │   │   │   │   │   ├── common_pb2.py
│   │   │   │   │   │   ├── hero_pb2.py
│   │   │   │   │   │   ├── scene_pb2.py
│   │   │   │   │   │   ├── sgame_ai_server_pb2.py
│   │   │   │   │   │   └── sgame_state_pb2.py
│   │   │   │   │   ├── scene.proto
│   │   │   │   │   ├── sgame_ai_server.proto
│   │   │   │   │   └── sgame_state.proto
│   │   │   │   └── README.md
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── multiagentenv.py
│   ├── __init__.py
│   ├── learners
│   │   ├── __init__.py
│   │   └── nq_learner.py
│   ├── main.py
│   ├── modules
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   └── n_rnn_agent.py
│   │   ├── __init__.py
│   │   └── mixers
│   │       ├── __init__.py
│   │       ├── nmix.py
│   │       ├── qatten.py
│   │       └── vdn.py
│   ├── run
│   │   ├── __init__.py
│   │   └── run.py
│   ├── runners
│   │   ├── episode_runner.py
│   │   └── __init__.py
│   └── utils
│       ├── logging.py
│       ├── rl_utils.py
│       ├── th_utils.py
│       └── timehelper.py
└── train.sh
The main file explanations are as follows:
- The code in this environment draws on the code implementation of the SMAC environment.
- `./src/envs/hok/hok_game/client/gamecore_controller.py` is responsible for controlling the engine, and it starts/stops the gamecore by sending HTTP requests to ugc_game_core_server.
- `./src/envs/hok/hok_game/conf/gamecore_conf.json` is used to configure the server's IP and port access.
- `./src/envs/hok/hok_game/conf/gamecore_conf.json` is used to set hero attributes and configID.


## Environment Interface File

The reinforcement learning algorithm will interact with the gamecore through the environment interface file.

### Environment Class Attribute Parameters

```python
# Inherits from the MultiAgentEnv class and Tencent GameCore class
class HokEnv(MultiAgentEnv, NatureClient):
    def __init__(
        self,
        map_name='hok',
        num_agents=5,
        time_step=0,
        seed=123,
        episode_limit=150,
        client_id=f"debug-train",
        logger=None,
        max_frame=60000
    ):
        super().__init__(client_id ,logger, max_frame)
```

> Note: The above are the default parameters for initialization
>- `NatureClient` is the GameCore rule class provided by Tencent
>- `MultiAgentEnv` is the basic environment class for multi-agent reinforcement learning

#### Map-related Parameters
```python
self.map_name = map_name # Map name
self.n_agents = num_agents # Number of agents modeled
```

#### Training-related Parameters
```python
self.episode_limit = episode_limit # Maximum number of steps for episode_limit
self._episode_steps = 0 # Step count for each episode
```

#### Action-related Parameters
```python
self.action_space = 13 # Dimension of action space
self.n_actions = self.action_space
```

> Note:
>- Actions include movement actions: up, down, left, right, up-left, up-right, down-right, down-left
>- Skill actions: normal attack, 1, 2, 3, summoner skills (usage)

#### State Space Parameters
```python
self.observation_space = 6 # The observation space for each agent is 6

# Initialize state and observation variables
self.obs = None
self.state = None
```

> Note:
>- Each agent's observation is: [own x-coordinate, own z-coordinate, own health, Tyrannosaurus' x-coordinate, Tyrannosaurus' z-coordinate, Tyrannosaurus' health]

#### Other Related Parameters
```python
self.seed = seed # Random seed
self.monster_hp_total = [] # Stores the Tyrannosaurus' health for the current episode
self.pos_delta = 1500 # Offset after discretizing movement actions, move 1500 based on the original coordinates
self.SKILL_TYPE = ["obj_skill", "dir_skill", "pos_skill", "talent_skill"] # Action execution methods
```

> Note:
>- self.SKILL_TYPE is the action execution method for skills
>- obj_skill is a target-based execution method, when released by oneself, target should be the hero's own ID
>- dir_skill is a direction-based execution method, default is to release towards the Tyrannosaurus direction
>- pos_skill is a position-based execution method, currently this setting does not involve this skill type
>- talent_skill is a summoner skill, target should be the hero's own ID

### Get State Function
```python
def get_obs(self):
    """Returns all agent observations in a two-dim list."""
    agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
    return agents_obs
    
def get_obs_agent(self, agent_id):
    return self.obs[agent_id].reshape(-1)

def get_obs_size(self):
    """Return the size of the observation"""
    return self.observation_space

def get_global_state(self):
    """Return Global State"""
    return self.obs.flatten()
def get_state(self):
    """Returns the global state."""
    return self.get_global_state()

def get_state_size(self):
   """Returns the size of the global state."""
   return self.get_obs_size() * self.n_agents
```
### Get Available Actions
```python
def get_avail_actions(self):
    """Returns the available actions of all agents in a list."""
    ## Now that skills have been added, not all skills can be used, the 5678 skill keys need to check calm_down==0, if it is 0, then it is 1
    all_actor_legal_skill = []
    for hero in self.heroes:
        if hero.actor_id == 6: # Tyrannosaurus exits directly
            break 
        hero_legal_skill = [1,1,1,1] # Directional movement does not need to consider calm_down
        # hero is the current hero, only select legal skills
        cur_legal_skill = [1 if slot.cooldown==0 else 0 for slot in hero.skill_state.slot_states]
        hero_legal_skill.extend(cur_legal_skill)
        hero_legal_skill.extend([1,1,1,1]) # Directional movement does not need to consider calm_down
        all_actor_legal_skill.append(hero_legal_skill)
    
    return all_actor_legal_skill
        
def get_avail_agent_actions(self, agent_id):
    """Returns the available actions for agent_id."""
    return self.get_avail_actions()[agent_id]

def get_total_actions(self):
    """Returns the total number of actions an agent could ever take"""
    return self.action_space
```
> Note:
>- Filter the legal actions that the agent can execute in the current frame
>- Provide a list of length equal to the action space for each agent, where each index represents whether the action can be executed, with 0/1 representing cannot/be executed respectively

### Action Conversion Function
#### Skill Definition Related
```python
def get_skill_id_by_slot_type(self, hero, slot_type):
    """Get skill slot by slot_type
    
    params:
        hero: Hero
        slot_type: Skill 1 / 2 / 3
        
    return:    
        Skill ID
    """
    skill_id = [slot.configId for slot in hero.skill_state.slot_states if slot.slot_type == slot_type][0]
    return skill_id
        
def __obj_skill_command(self, target, hero, skill_slot_type):
    """Targeted skill
    
    params:
        target: Target ID
        hero: Hero ID
        skill_slot_type: Skill
    
    return:
        Targeted skill command
    
    """
    cmd_pkg = CmdPkg()
    objSkill = ObjSkill()
    objSkill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    objSkill.actorID = target
    objSkill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_ObjSkill
    cmd_pkg.obj_skill.CopyFrom(objSkill)

    return cmd_pkg
        
def __dir_skill_command(self, target, hero, skill_slot_type):
    """Directional skill
    
    params:
        target: Target ID
        hero: Hero ID
        skill_slot_type: Skill
    
    return:
        Directional skill command
    
    """
    cmd_pkg = CmdPkg()
    skill = DirSkill() 
    skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.actorID = target 
    skill.slotType = skill_slot_type
    skill.degree = 4
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_DirSkill
    cmd_pkg.dir_skill.CopyFrom(skill)
    
    return cmd_pkg
        
def __pos_skill_command(self, hero, skill_slot_type):
    """Positional skill
    
    params:
        hero: Hero ID
        skill_slot_type: Skill
    
    return:
        Positional skill command
    
    """
    cmd_pkg = CmdPkg()
    skill = PosSkill()
    skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.destPos.CopyFrom(self.random_position())
    skill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_PosSkill
    cmd_pkg.pos_skill.CopyFrom(skill)
    
    return cmd_pkg
        
def __talent_skill_command(self, target):
    """Summoner skill
    
    params:
        hero: Hero ID
        skill_slot_type: Skill
    
    return:
        Summoner skill command
    
    """
    cmd_pkg = CmdPkg()
    skill = TalentSkill()
    skill.degree = 90
    skill.actorID = target
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_TalentSkill
    cmd_pkg.talent_skill.CopyFrom(skill)
    
    return cmd_pkg
```
#### Action to Command
```python
def act_2_cmd(self, actions):
    """Convert agent actions to commands that gamecore can execute
    
    params:
        actions: List of agent actions -> [2,3,1,4,1]
    
    return:
        Generate a command list for each agent
    """
    cmd_list, stop_game = [], False
    for _id, act in enumerate(actions): # [0,1,2,3,4]
        '''actor_id = _id+1'''
        cmd_pkg = CmdPkg()
        # Adapt the skill casting type of each hero
        if _id == 0:
            """Zhuang Zhou"""
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
            """Di Renjie"""
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
            """Diao Chan"""
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
            """Sun Wukong"""
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
            """Cao Cao"""
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
        if act == 4:
            # Normal attack
            attack = AttackCommon()
            attack.actorID = target
            attack.start = 1
            cmd_pkg.command_type = CommandType.COMMAND_TYPE_AttackCommon
            cmd_pkg.attack_common.CopyFrom(attack)

        elif act == 5:
            skill_slot_type = 1 # Skill 1
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)

        elif act == 6:
            skill_slot_type = 2 
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)

        elif act == 7:
            skill_slot_type = 3 
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)
        
        elif act == 8: 
            '''obj_skill talent skill __talent_skill_command(target)'''
            cmd_pkg = self.__talent_skill_command(target)
        
        else:
            # # 0 up 1 down 2 left 3 right 9 up left 10 up right 11 down right 12 down left
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
```

1. Action mapping relationship:
<table>
  <tr>
    <th>0</th>
    <td>up</td>
  </tr>
  <tr>
    <th>1</th>
    <td>down</td>
  </tr>
  <tr>
    <th>2</th>
    <td>left</td>
  </tr>
  <tr>
    <th>3</th>
    <td>right</td>
  </tr>
  <tr>
    <th>4</th>
    <td>normal attack</td>
  </tr>
  <tr>
    <th>5</th>
    <td>skill1 Currently, all 1, 2, and 3 skills are cast in the form of obj_skill</td>
  </tr>
  <tr>
    <th>6</th>
    <td>skill2</td>
  </tr>
  <tr>
    <th>7</th>
    <td>skill3</td>
  </tr>
  <tr>
    <th>8</th>
    <td>skill4</td>
  </tr>
  <tr>
    <th>9</th>
    <td>left_up</td>
  </tr>
  <tr>
    <th>10</th>
    <td>right_up</td>
  </tr>
  <tr>
    <th>11</th>
    <td>right_down</td>
  </tr>
  <tr>
    <th>12</th>
    <td>left_down</td>
  </tr>
</table>
         
2. actions = [a1, a2, a3, a4, a5] correspond to 5 agents:
   - Zhuang Zhou: Skill 1 (Directional), Skill 2 (Self-release), Skill 3 (Self-release)        
   - Di Renjie: Skill 1 (Directional), Skill 2 (Directional), Skill 3 (Directional)               
   - Diao Chan: Skill 1 (Directional), Skill 2 (Directional), Skill 3 (Self-release)               
   - Sun Wukong: Skill 1 (Self-release), Skill 2 (Directional), Skill 3 (Self-release)                   
   - Cao Cao: Skill 1 (Directional), Skill 2 (Directional), Skill 3 (Self-release)

### Environment initialization function reset()
```python
def reset(self, if_test=False, args=None, cur_time=None):
    """Reset the environment. Required after each full episode.
        Returns initial observations and states. 
        Starting game in every episode phase of sampling
    """
    # Determine the victory condition of the game
    self.battles_won = 0
    self.battles_game = 0
    # Store the blood volume of Tyrannosaurus at each frame
    self.monster_hp_total = [] 
    # The initial step of the game is 0
    self._episode_steps = 0
    # Additional information returned for each frame
    self.info_re = {'win_rate': 0., 'test_win_rate': 0., 'battle_won': None, 'monster_last_hp': None}
    
    ## Start the game engine
    # Record the start time node
    self.start_point = time.time()
    # Define the id of the game startup for this time, used to generate abs files
    if if_test:
        self.game_id = f"{args.name}-hok-test-{cur_time}" 
    else:
        self.game_id = f"{args.name}-hok-train-{cur_time}"
    
    # logger log record
    self.logger.info('New episode game starting!')
    # Initialize the game
    self.reset_game()
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
    ti = time.time() - self.start_point
    # Record the game startup time
    self.step_time_cost+=ti
    # Receive the initial frame game information
    self.message_name, self.message_proto = self.recv_request()
    # Receive game judgment
    if self.message_name == "FightStartReq":
        self.rsp = FightStartRsp()

    elif self.message_name == "StepFrameReq":
        self.rsp = StepFrameRsp()
        # Parse step frame request from game core
        self._parse_frame_info(self.message_proto) 
        #self._print_debug_log(freq=1)

    elif self.message_name == "FightOverReq":
        self.rsp = FightOverRsp()
        gameover_state = self.message_proto.gameover_state
        self.game_over = True
        if gameover_state == 1:
            self.game_status = NC_CONFIG["game_status"]["win"]
        elif gameover_state == 2:
            self.game_status = NC_CONFIG["game_status"]["fail"]
        elif gameover_state == 4:
            self.game_status = NC_CONFIG["game_status"]["error"]
        #self.send_response(self.rsp)
    
    else:
        self.logger.warning("Warning: receiving message fails")
    
    self.start_point = time.time()
    self.monster_hp_total.append(self.monster_hp)
    self.obs = self.agent_loc_np 
    # 状态归一化 -> [-1, 1]
    self.obs = self.obs - np.mean(self.obs)
    self.obs = self.obs / np.max(np.abs(self.obs))
    
    return self.get_obs(), self.get_state()
```      

Get the information of the first frame, get the three pieces of information returned by gamecore, namely:
```python
self.hero_hp: {'1': 16772, '2': 5706, '3': 8409, '4': 8743, '5': 9885}  
self.monster_hp: 30000 (some int value)   
self.agent_loc: [[   842   2956  agenthp 14501  -3754 monsterhp]   
                 [   861  -1095  agenthp 14501  -3754 monsterhp]   
                 [   769  -4468  agenthp 14501  -3754 monsterhp]         
                 [   639  -7465  agenthp 14501  -3754 monsterhp]         
                 [   515 -10797  agenthp 14501  -3754 monsterhp]]       
np.array()   
``` 

### Interaction progression function step()
The interactive deduction function is responsible for the frame-by-frame deduction of the agent and the simulation.
```python
def step(self, _actions, if_test=False):
    """Returns reward, terminated, info
    
    params:
        _actions: List of agent actions
        if_test: Whether it is in test mode
        
    return:
        Returns the reward, terminated, info of the current frame
    """
    
    if th.is_tensor(_actions):
        #actions = _actions.cpu.numpy()
        actions = [int(a) for a in _actions]
    else:
        actions = _actions
        
    self._episode_steps+=1
    print(f'Current frame: {self._episode_steps}, which actions have been performed: {_actions}')
    ti = time.time() - self.start_point
    self.step_time_cost+=ti
    self.start_point = time.time()
    # Record the number of requests sent
    self.timestep+=1
    # Each frame training, first give the opposite action, see the opposite feedback, because reset() has already got the initial information, the action now done is equivalent to the action done based on the initial state
    if self.message_name == "FightStartReq":
        self.send_response(self.rsp)

    elif self.message_name == "StepFrameReq":
        # get actions from agent 
        cmd_list, stop_game = self.act_2_cmd(actions) # Need to re-edit

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
        
    # Initialize reward    
    reward = 0.
    terminated = False
    """
    Two termination conditions:
        1. If the running steps exceed the episode_limit, this should be the termination condition, and judge the game status
        2. The running step is less than episode_limit, but it has won
    """
    if (self._episode_steps >= self.episode_limit) or \
        ((self._episode_steps < self.episode_limit) and self.game_status==1):
        terminated = True 
        self.game_over = True
        if (self._episode_steps >= self.episode_limit):
            self.game_status = NC_CONFIG["game_status"]["overtime"]

    if not self.game_over:
        ti = time.time() - self.start_point
        self.step_time_cost+=ti 
        self.message_name, self.message_proto = self.recv_request()
        #print('received next states!!!!!!')
        if self.message_name == "FightStartReq":
            self.rsp = FightStartRsp()
            #self.send_response(self.rsp)

        elif self.message_name == "StepFrameReq":
            self.rsp = StepFrameRsp()
            # Parse step frame request from game core
            self._parse_frame_info(self.message_proto) 可
            self._print_debug_log(freq=1)

        elif self.message_name == "FightOverReq":
            self.rsp = FightOverRsp()
            gameover_state = self.message_proto.gameover_state
            self.game_over = True
            
            if gameover_state == 1:
                self.game_status = NC_CONFIG["game_status"]["win"]
            elif gameover_state == 2:
                self.game_status = NC_CONFIG["game_status"]["fail"]
            elif gameover_state == 4:
                self.game_status = NC_CONFIG["game_status"]["error"]

        self.start_point = time.time()

        self.monster_hp_total.append(self.monster_hp)
        
        self.obs = self.agent_loc_np 
        # obs normalize -> [-1,1]
        self.obs = self.obs - np.mean(self.obs)
        self.obs = self.obs / np.max(np.abs(self.obs))
        reward = self.get_curr_reward()
        terminated = False

    else:
        avg_time = self.step_time_cost / self.timestep
        self.logger.info(f"******* Game Over with status {self.game_status} *******")
        self.logger.info("0: error, 1: win, 2: fail, 3: overtime")
        self.logger.info(
            f"FrameNo = [{self.frame_no}], StepNo = [{self.timestep}], avg time = [{avg_time}]")
        
        self.controller.stop_game() 
        self.battles_game+=10
        
        if self.game_status == 1:
            self.info_re['battle_won'] = True
        elif self.game_status == 2 or 3:
            self.info_re['battle_won'] = False
            
        self.info_re['monster_last_hp'] = self.monster_hp_total[-1]
        
    return reward, terminated, self.info_re
```  

### Calculate Reward Function
```python
def get_curr_reward(self):
    """
    Return reward based on the Tyrannosaurus's health
    """
    return (self.monster_hp_total[-1] - self.monster_hp_total[-2]) * -0.01
```  

### Other Functions
#### Calculate the Coordinates After Moving
```python
def position_change(self, agent_id, act):
    '''Used to calculate the position coordinates after moving
    '''
    # self.agent_loc_np is the state information saved last time
    dst_pos=VInt3()
    last_loc_x = self.agent_loc_np[agent_id][0] # Get the x coordinate position of the last round
    last_loc_z = self.agent_loc_np[agent_id][1] # Get the z coordinate position of the last round
    if act==0:
        dst_pos.x = last_loc_x + self.pos_delta 
        dst_pos.z = last_loc_z
        dst_pos.y = 100

    elif act==1:
        dst_pos.x = last_loc_x - self.pos_delta 
        dst_pos.z = last_loc_z
        dst_pos.y = 100

    elif act==2:
        dst_pos.x = last_loc_x 
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100

    elif act==3:
        dst_pos.x = last_loc_x
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
    
    # """Refine moving actions: upper left, upper right, lower right, lower left"""
    elif act==9:
        # Upper left
        dst_pos.x = last_loc_x + self.pos_delta
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
        
    elif act==10:
        # Upper right
        dst_pos.x = last_loc_x + self.pos_delta
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100
        
    elif act==11:
        # Lower right
        dst_pos.x = last_loc_x - self.pos_delta
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100
        
    elif act==12:
        # Lower left
        dst_pos.x = last_loc_x - self.pos_delta
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
        
    # Boundary handling
    if dst_pos.x > 28700:
        dst_pos.x = 28700
    elif dst_pos.x < -28700:
        dst_pos.x = -28700
    
    if dst_pos.z > 9000:
        dst_pos.z = 9000
    elif dst_pos.z < -16800:
        dst_pos.z = -16800
    
    return dst_pos
```  
Note:
By recording the coordinates of the last frame and adding the offset of the current frame, the latest coordinates are obtained
- act=0 is moving upwards along x  
- act=1 is moving downwards along x  
- act=2 is moving upwards along z  
- act=3 is moving downwards along z  
- act=9 is moving upwards along the upper left  
- act=10 is moving upwards along the upper right  
- act=11 is moving downwards along the lower right  
- act=12 is moving downwards along the lower left

