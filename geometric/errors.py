"""
errors.py : errors module with definition of Customized Errors

The classes and subclasses in this module defines a "tree" relation of exceptions,
that can be used thoughout the code for a consistent error handling pattern.

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

class Error(Exception):
    pass

class InputError(Error):
    pass

class HessianExit(Error):
    pass

class EngineError(Error):
    pass

class ParamError(Error):
    pass

class FrequencyError(Error):
    pass

class CheckCoordError(Error):
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

class GaussianEngineError(EngineError):
    pass

class QCEngineAPIEngineError(EngineError):
    pass

class ConicalIntersectionEngineError(EngineError):
    pass

class GeomOptNotConvergedError(Error):
    pass

class GeomOptStructureError(Error):
    pass

class LinearTorsionError(GeomOptStructureError):
    pass
