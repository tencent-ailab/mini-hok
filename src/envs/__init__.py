from functools import partial

from .multiagentenv import MultiAgentEnv


from .hok import HokEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["hok"] = partial(env_fn, env=HokEnv)
