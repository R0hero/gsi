from enum import Enum

class ReflectionType(Enum):
    """Reflection types for use in determining how drawn signal will look like"""
    SPECULAR = 'Specular'
    DIFFUSE = 'Diffuse'

class GroundType(Enum):
    """Ground types for use in determining which ground to compute for the figure"""
    COMMON = 'Common'
    RIVER = 'River' # WIP
    COAST = 'Coast' # WIP

class AntennaType(Enum):
    """Antenna types for use to determine radiation pattern"""
    GEODETIC = 'Geodetic'
    HELIX = 'Helix'
    PATCH = 'Patch' # WIP