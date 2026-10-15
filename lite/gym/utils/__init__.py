"""Env-facing helpers, grouped by *what an env author reaches for*.

This package deliberately exports nothing: the concept lives one level down and
every import names a child — ``backend`` (host resources: docker, ports, key
and coordinate projection, image freshness), ``config`` (an env's declared
configuration, identity and container naming), ``feedback`` (step tool-feedback
and action-surface helpers), ``server`` (env-server lifecycle and routing
helpers). A re-export barrel here would give each symbol two spellings; the
four children are the whole surface.

The empty-file version of this module was read as "the path segment carries no
concept, so hoist the four children to ``lite.gym``". Deferred to the package
owner, with the measurement that makes it a bigger change than it looks:

  * the segment IS reached as an object, not only traversed — ``from
    lite.gym.utils import config as env_config`` is the tree's standard config
    idiom (~39 production sites, several of them comments in ``configs/*.yaml``
    and heredocs in ``scripts/*.sh`` that no symbol rename would follow), and
    four discipline gates do ``import lite.gym.utils as gym_utils`` and assert
    ``not hasattr(gym_utils, <retired flat name>)`` — that guard's subject is
    this package object, so a hoist deletes the thing being guarded;
  * ``lite.gym.utils.server`` hoisted to ``lite.gym.server`` would sit beside
    ``lite.gym.remote.server``, trading a vague segment for a collided one.

Run: not directly.
"""
