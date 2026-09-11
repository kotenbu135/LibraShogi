# SPDX-License-Identifier: Apache-2.0
"""librasearch: 自己対局エンジン（libra-search）の Python バインディング。"""
from ._search import SelfPlay, solve  # noqa: F401

__all__ = ["SelfPlay", "solve"]
