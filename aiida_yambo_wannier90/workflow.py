#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base class for Yambo+Wannier90 workflow."""
import typing as ty

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes import WorkChain, ToContext, if_, ProcessBuilder

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin


class YamboWannier90WorkChain(ProtocolMixin, WorkChain):
    """Workchain to obtain GW-corrected maximally localised Wannier functions (MLWF)."""
