"""Registry manifest schema — metadata for test collections."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CollectionManifest:
    """Metadata describing a saved test resource collection.

    Attributes:
        name: Unique identifier for the collection.
        version: Schema version of the collection (default ``"1.0"``).
        description: Human-readable description of what the collection contains.
        author: Creator of the collection.
        created: ISO-8601 timestamp of when the collection was saved.
        files: List of relative file paths included in the collection.
        tags: Arbitrary labels for filtering/searching collections.
    """

    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize manifest to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created": self.created,
            "files": list(self.files),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CollectionManifest:
        """Deserialize a manifest from a dictionary (e.g. loaded from JSON).

        Args:
            data: Dictionary with manifest fields.

        Returns:
            A :class:`CollectionManifest` instance.
        """
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created=data.get("created", datetime.now().isoformat()),
            files=list(data.get("files", [])),
            tags=list(data.get("tags", [])),
        )
