"""
errors module with definition of Customized Errors

The classes and subclasses in this module defines a "tree" relation of exceptions,
that can be used thoughout the code for a consistent error handling pattern.
"""

class Error(Exception):
    pass

class EngineError(Error):
    pass

class TeraChemEngineError(EngineError):
    pass

class OpenMMEngineError(EngineError):
    pass

class Psi4EngineError(EngineError):
    pass

class QChemEngineError(EngineError):
    pass

class GromacsEngineError(EngineError):
    pass

class MolproEngineError(EngineError):
    pass

class QCEngineAPIEngineError(EngineError):
    pass

class ConicalIntersectionEngineError(EngineError):
    pass

class GeomOptNotConvergedError(Error):
    pass
