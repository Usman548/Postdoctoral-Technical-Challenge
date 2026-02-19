"""
Core abstractions and protocols for SOLID compliance.
Shared interfaces so high-level modules depend on abstractions, not concretions.
"""

from core.protocols import DatasetProvider, ModelLoaderProtocol

__all__ = ["DatasetProvider", "ModelLoaderProtocol"]
