"""Exact 2D pattern geometry, interface solving, and assembly compilation."""

from .assembler import compile_assembly
from .inputs import GeometryInputs, load_geometry_inputs
from .interface_solver import solve_interfaces
from .library import assembly_recipe, build_geometry_library, component_interfaces, component_registry

__all__ = [
    "GeometryInputs",
    "assembly_recipe",
    "build_geometry_library",
    "compile_assembly",
    "component_interfaces",
    "component_registry",
    "load_geometry_inputs",
    "solve_interfaces",
]
