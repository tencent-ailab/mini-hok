# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hero.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import common_pb2 as common__pb2
from . import command_pb2 as command__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='hero.proto',
  package='',
  syntax='proto2',
  serialized_pb=_b('\n\nhero.proto\x1a\x0c\x63ommon.proto\x1a\rcommand.proto\"\xcb\x01\n\x0eSkillSlotState\x12\x10\n\x08\x63onfigId\x18\x01 \x02(\x05\x12!\n\tslot_type\x18\x02 \x02(\x0e\x32\x0e.SkillSlotType\x12\r\n\x05level\x18\x03 \x02(\x05\x12\x0e\n\x06usable\x18\x04 \x02(\x08\x12\x10\n\x08\x63ooldown\x18\x05 \x02(\x05\x12\x14\n\x0c\x63ooldown_max\x18\x06 \x02(\x05\x12\x11\n\tusedTimes\x18\x07 \x01(\x05\x12\x14\n\x0chitHeroTimes\x18\x08 \x01(\x05\x12\x14\n\x0c\x61ttack_range\x18\t \x02(\x05\"2\n\nSkillState\x12$\n\x0bslot_states\x18\x01 \x03(\x0b\x32\x0f.SkillSlotState\"F\n\x0bProtectInfo\x12!\n\x0bprotectType\x18\x01 \x01(\x0e\x32\x0c.ProtectType\x12\x14\n\x0cprotectValue\x18\x02 \x01(\r\"\xda\x01\n\tHeroState\x12\x10\n\x08\x61\x63tor_id\x18\x01 \x01(\r\x12 \n\x0b\x61\x63tor_state\x18\x02 \x02(\x0b\x32\x0b.ActorState\x12 \n\x0bskill_state\x18\x03 \x02(\x0b\x32\x0b.SkillState\x12\x0f\n\x07killCnt\x18\x04 \x02(\x05\x12\x0f\n\x07\x64\x65\x61\x64\x43nt\x18\x05 \x02(\x05\x12\x17\n\x0ftotalHurtToHero\x18\x06 \x02(\x05\x12\x19\n\x11totalBeHurtByHero\x18\x07 \x02(\x05\x12!\n\x0bprotectInfo\x18\x08 \x03(\x0b\x32\x0c.ProtectInfo')
  ,
  dependencies=[common__pb2.DESCRIPTOR,command__pb2.DESCRIPTOR,])




_SKILLSLOTSTATE = _descriptor.Descriptor(
  name='SkillSlotState',
  full_name='SkillSlotState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='configId', full_name='SkillSlotState.configId', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='slot_type', full_name='SkillSlotState.slot_type', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='level', full_name='SkillSlotState.level', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='usable', full_name='SkillSlotState.usable', index=3,
      number=4, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cooldown', full_name='SkillSlotState.cooldown', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cooldown_max', full_name='SkillSlotState.cooldown_max', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='usedTimes', full_name='SkillSlotState.usedTimes', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hitHeroTimes', full_name='SkillSlotState.hitHeroTimes', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attack_range', full_name='SkillSlotState.attack_range', index=8,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=247,
)


_SKILLSTATE = _descriptor.Descriptor(
  name='SkillState',
  full_name='SkillState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='slot_states', full_name='SkillState.slot_states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=249,
  serialized_end=299,
)


_PROTECTINFO = _descriptor.Descriptor(
  name='ProtectInfo',
  full_name='ProtectInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='protectType', full_name='ProtectInfo.protectType', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='protectValue', full_name='ProtectInfo.protectValue', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=301,
  serialized_end=371,
)


_HEROSTATE = _descriptor.Descriptor(
  name='HeroState',
  full_name='HeroState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='actor_id', full_name='HeroState.actor_id', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='actor_state', full_name='HeroState.actor_state', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='skill_state', full_name='HeroState.skill_state', index=2,
      number=3, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='killCnt', full_name='HeroState.killCnt', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='deadCnt', full_name='HeroState.deadCnt', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='totalHurtToHero', full_name='HeroState.totalHurtToHero', index=5,
      number=6, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='totalBeHurtByHero', full_name='HeroState.totalBeHurtByHero', index=6,
      number=7, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='protectInfo', full_name='HeroState.protectInfo', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=374,
  serialized_end=592,
)

_SKILLSLOTSTATE.fields_by_name['slot_type'].enum_type = common__pb2._SKILLSLOTTYPE
_SKILLSTATE.fields_by_name['slot_states'].message_type = _SKILLSLOTSTATE
_PROTECTINFO.fields_by_name['protectType'].enum_type = common__pb2._PROTECTTYPE
_HEROSTATE.fields_by_name['actor_state'].message_type = common__pb2._ACTORSTATE
_HEROSTATE.fields_by_name['skill_state'].message_type = _SKILLSTATE
_HEROSTATE.fields_by_name['protectInfo'].message_type = _PROTECTINFO
DESCRIPTOR.message_types_by_name['SkillSlotState'] = _SKILLSLOTSTATE
DESCRIPTOR.message_types_by_name['SkillState'] = _SKILLSTATE
DESCRIPTOR.message_types_by_name['ProtectInfo'] = _PROTECTINFO
DESCRIPTOR.message_types_by_name['HeroState'] = _HEROSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SkillSlotState = _reflection.GeneratedProtocolMessageType('SkillSlotState', (_message.Message,), dict(
  DESCRIPTOR = _SKILLSLOTSTATE,
  __module__ = 'hero_pb2'
  # @@protoc_insertion_point(class_scope:SkillSlotState)
  ))
_sym_db.RegisterMessage(SkillSlotState)

SkillState = _reflection.GeneratedProtocolMessageType('SkillState', (_message.Message,), dict(
  DESCRIPTOR = _SKILLSTATE,
  __module__ = 'hero_pb2'
  # @@protoc_insertion_point(class_scope:SkillState)
  ))
_sym_db.RegisterMessage(SkillState)

ProtectInfo = _reflection.GeneratedProtocolMessageType('ProtectInfo', (_message.Message,), dict(
  DESCRIPTOR = _PROTECTINFO,
  __module__ = 'hero_pb2'
  # @@protoc_insertion_point(class_scope:ProtectInfo)
  ))
_sym_db.RegisterMessage(ProtectInfo)

HeroState = _reflection.GeneratedProtocolMessageType('HeroState', (_message.Message,), dict(
  DESCRIPTOR = _HEROSTATE,
  __module__ = 'hero_pb2'
  # @@protoc_insertion_point(class_scope:HeroState)
  ))
_sym_db.RegisterMessage(HeroState)


# @@protoc_insertion_point(module_scope)
