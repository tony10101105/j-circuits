# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified from anthropics/jacobian-lens by Tung-Yu (Tony) Wu and his Claude.
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._env import load_dotenv
from jlens._logging import configure_logging
from jlens.circuit import Edge, JCircuit, Node, build_jcircuit
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.interventions import (
    Ablate,
    Intervention,
    Steer,
    Swap,
    TopKAblation,
    greedy_generate,
    lens_vector,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.pursuit import gradient_pursuit, pursue_lens

__all__ = [
    "Ablate",
    "ActivationRecorder",
    "Edge",
    "HFLensModel",
    "Intervention",
    "JCircuit",
    "JacobianLens",
    "Layout",
    "LensModel",
    "Node",
    "Steer",
    "Swap",
    "TopKAblation",
    "build_jcircuit",
    "configure_logging",
    "fit",
    "from_hf",
    "gradient_pursuit",
    "greedy_generate",
    "jacobian_for_prompt",
    "lens_vector",
    "load_dotenv",
    "pursue_lens",
]
