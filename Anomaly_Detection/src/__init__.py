"""
I moduli in `src/` sono funzioni pure - non scrivono
su disco e non hanno side effect. Chi orchestra e salva sta in `scripts/`.
"""

__all__ = ["config", "io", "features"]
