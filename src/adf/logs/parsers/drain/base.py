# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    @property
    @abstractmethod
    def templates(self) -> Any:
        """Return the templates.

        Returns
        -------
            Any : The templates.

        """
        pass

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the clustering score.

        Args:
            *args : tuple
                Variable length argument list.
            **kwargs : dict
                Arbitrary keyword arguments.

        Returns:
            Any : The clustering score.

        """
        pass

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit the model according to the given training data.

        Args:
            *args : tuple
                Variable length argument list.
            **kwargs : dict
                Arbitrary keyword arguments.

        Returns:
            None

        """
        pass

    @abstractmethod
    def save_model(self, *args: Any, **kwargs: Any) -> None:
        """Save the model to disk.

        Args:
            *args : tuple
                Variable length argument list.
            **kwargs : dict
                Arbitrary keyword arguments.

        Returns:
            None

        """
        pass
