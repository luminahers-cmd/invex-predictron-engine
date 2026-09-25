"""Validator registry — Phase 8 Sprint 4.

:class:`ValidatorRegistry` is the single place validators are registered,
unregistered, resolved, and enumerated.  The validation engine never
knows concrete validator classes: every lookup flows through this
registry, making the registry the extension point for future validators.

The registry is deterministic and isolated:

* registration order is preserved,
* topic resolution is stable (first registered validator wins),
* duplicate registration is rejected,
* separate registry instances share no state,
* the registry can serialize its registered validator ids and rebuild
  itself from a canonical validator catalog.
"""

from __future__ import annotations

from collections.abc import Sequence

from predictron_engine.research.exceptions import (
    DuplicateValidator,
    RegistryError,
    UnsupportedValidation,
)
from predictron_engine.research.validation_models import (
    VALIDATION_SCHEMA_VERSION,
    ValidatorMetadata,
    _require_str,
)
from predictron_engine.research.validators import (
    DEFAULT_VALIDATORS,
    VALIDATOR_CATALOG,
    EvidenceValidator,
)


class ValidatorRegistry:
    """A mutable catalog of :class:`EvidenceValidator` instances.

    Parameters
    ----------
    validators:
        Optional initial validators registered in the given order.  When
        ``None`` the registry starts empty.
    """

    def __init__(
        self,
        validators: Sequence[EvidenceValidator] | None = None,
    ) -> None:
        self._validators: list[EvidenceValidator] = []
        if validators is not None:
            for validator in validators:
                self.register(validator)

    def __len__(self) -> int:
        """Return the number of registered validators."""
        return len(self._validators)

    def __contains__(self, validator_id: object) -> bool:
        """Return whether a validator with ``validator_id`` is registered."""
        return isinstance(validator_id, str) and self.has_validator(
            validator_id
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, validator: EvidenceValidator) -> EvidenceValidator:
        """Register ``validator`` and return it.

        Registration order is preserved and determines resolution order.

        Raises
        ------
        RegistryError:
            When ``validator`` is not an :class:`EvidenceValidator`.
        DuplicateValidator:
            When a validator with the same ``validator_id`` is already
            registered.
        """
        if not isinstance(validator, EvidenceValidator):
            raise RegistryError(
                "register expects an EvidenceValidator instance"
            )
        validator_id = validator.metadata().validator_id
        if self.has_validator(validator_id):
            raise DuplicateValidator(
                f"validator already registered: {validator_id!r}"
            )
        self._validators.append(validator)
        return validator

    def unregister(self, validator_id: str) -> EvidenceValidator | None:
        """Remove and return the validator for ``validator_id``.

        Returns ``None`` when no such validator is registered (the
        operation is idempotent).
        """
        for position, validator in enumerate(self._validators):
            if validator.metadata().validator_id == validator_id:
                del self._validators[position]
                return validator
        return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list_validators(self) -> tuple[EvidenceValidator, ...]:
        """Return registered validators in registration order."""
        return tuple(self._validators)

    def get(self, validator_id: str) -> EvidenceValidator | None:
        """Return the validator for ``validator_id``, or ``None``."""
        for validator in self._validators:
            if validator.metadata().validator_id == validator_id:
                return validator
        return None

    def has_validator(self, validator_id: str) -> bool:
        """Return whether a validator with ``validator_id`` is registered."""
        return self.get(validator_id) is not None

    def validators_for(self, topic_id: str) -> tuple[EvidenceValidator, ...]:
        """Return validators supporting ``topic_id``, in registration order."""
        return tuple(
            validator
            for validator in self._validators
            if validator.supports(topic_id)
        )

    def resolve(self, topic_id: str) -> EvidenceValidator:
        """Resolve a single validator for ``topic_id``.

        The first registered validator that supports ``topic_id`` wins,
        which makes resolution deterministic.

        Raises
        ------
        UnsupportedValidation:
            When no registered validator supports ``topic_id``.
        """
        validators = self.validators_for(topic_id)
        if not validators:
            raise UnsupportedValidation(
                f"no registered validator supports topic: {topic_id!r}"
            )
        return validators[0]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize registered validator ids to a JSON-ready dictionary.

        The payload is a list of :class:`ValidatorMetadata` descriptors in
        registration order; :meth:`from_dict` reconstructs instances from
        the canonical validator catalog.
        """
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "validators": [
                validator.metadata().to_dict()
                for validator in self._validators
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ValidatorRegistry:
        """Deserialize a registry from :meth:`to_dict` output.

        Raises
        ------
        RegistryError:
            When the payload is malformed or references a validator id
            that is not part of the canonical validator catalog.
        """
        version = data.get("schema_version")
        if version != VALIDATION_SCHEMA_VERSION:
            raise RegistryError(
                f"unsupported registry schema version: {version!r}"
            )
        descriptors = data.get("validators")
        if not isinstance(descriptors, list):
            raise RegistryError("expected a 'validators' list")
        registry = cls()
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise RegistryError(
                    "validator descriptors must be dictionaries"
                )
            validator_id = _require_str(descriptor, "validator_id")
            catalog_entry = VALIDATOR_CATALOG.get(validator_id)
            if catalog_entry is None:
                raise RegistryError(
                    f"validator not in catalog: {validator_id!r}"
                )
            registry.register(catalog_entry)
        return registry

    def validator_ids(self) -> tuple[str, ...]:
        """Return registered validator ids in deterministic order."""
        return tuple(
            validator.metadata().validator_id
            for validator in self._validators
        )

    def metadata(self) -> tuple[ValidatorMetadata, ...]:
        """Return registered validator metadata in registration order."""
        return tuple(
            validator.metadata() for validator in self._validators
        )


# Default registry shared by the validation engine and convenience API.
VALIDATOR_REGISTRY = ValidatorRegistry(DEFAULT_VALIDATORS)


def default_validator_registry() -> ValidatorRegistry:
    """Return a fresh registry preloaded with the default validators."""
    return ValidatorRegistry(DEFAULT_VALIDATORS)


def register_validator(validator: EvidenceValidator) -> EvidenceValidator:
    """Register ``validator`` on the shared registry."""
    return VALIDATOR_REGISTRY.register(validator)


def unregister_validator(validator_id: str) -> EvidenceValidator | None:
    """Unregister ``validator_id`` from the shared registry."""
    return VALIDATOR_REGISTRY.unregister(validator_id)


def resolve_validator(topic_id: str) -> EvidenceValidator:
    """Resolve a validator for ``topic_id`` on the shared registry."""
    return VALIDATOR_REGISTRY.resolve(topic_id)


def list_validators() -> tuple[EvidenceValidator, ...]:
    """List the shared registry's validators in registration order."""
    return VALIDATOR_REGISTRY.list_validators()


def has_validator(validator_id: str) -> bool:
    """Return whether ``validator_id`` is registered on the shared registry."""
    return VALIDATOR_REGISTRY.has_validator(validator_id)


__all__ = [
    "DEFAULT_VALIDATORS",
    "VALIDATOR_CATALOG",
    "VALIDATOR_REGISTRY",
    "ValidatorRegistry",
    "default_validator_registry",
    "has_validator",
    "list_validators",
    "register_validator",
    "resolve_validator",
    "unregister_validator",
]
