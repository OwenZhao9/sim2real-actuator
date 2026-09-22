"""Library error type.

Contract 0.4: every library defines its own ``<Lib>Error`` base class and uses it
only for construction-time / argument errors.  Contract 0.4 also says construction
errors must be ``ValueError``.  Both hold at once here because
:class:`Sim2RealActuatorError` *is* a ``ValueError`` -- callers that only know about
``ValueError`` keep working, callers that want to tell this library's errors apart
can catch the narrower type.
"""

from __future__ import annotations

__all__ = ["Sim2RealActuatorError"]


class Sim2RealActuatorError(ValueError):
    """Raised for invalid arguments to this library (construction time only).

    Subclasses :class:`ValueError`, so ``except ValueError`` also catches it.
    """
