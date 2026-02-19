"""
Protocols (abstract interfaces) for dependency inversion.
Consumers depend on these; implementors (data_loader, models, etc.) satisfy them.
"""

from typing import List, Tuple, Any, Protocol, runtime_checkable

# Avoid circular import: use TYPE_CHECKING for torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

@runtime_checkable
class DatasetProvider(Protocol):
    """Protocol for data providers. Train/Evaluate depend on this, not concrete datasets."""

    def get_dataloaders(
        self,
    ) -> Tuple[Any, Any, Any]:  # DataLoader, DataLoader, DataLoader
        ...

    def get_class_names(self) -> List[str]:
        ...

    @property
    def class_weights(self) -> Any:  # List[float] or torch.Tensor
        ...


@runtime_checkable
class ModelLoaderProtocol(Protocol):
    """Protocol for loading a classification model (architecture + weights)."""

    def load(
        self,
        model_name: str,
        num_classes: int,
        checkpoint_path: str,
        device: Any,
        **kwargs: Any,
    ) -> Any:  # nn.Module
        ...
