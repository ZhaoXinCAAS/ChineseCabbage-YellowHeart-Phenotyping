"""Shared TypedDict contracts for X-AnyLabeling annotation JSON.

``json.load`` results are annotated directly at the read site
(``data: AnnotationFile``); ``ShapeRecord`` stays ``total=False`` because
label/points are always read defensively.

.. note::
   A structural ``Classifier`` ``Protocol`` for sklearn estimators was tried
   and removed: sklearn's own stubs type ``fit``/``predict`` too loosely for
   one Protocol to satisfy *both* checkers (precise parameters are rejected
   by contravariance in basedpyright; permissive ones are rejected in
   pyrefly). The closed estimator union in ``baseline_compare`` is the
   precise tool for a fixed model set.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ShapeRecord(TypedDict, total=False):
    """One X-AnyLabeling polygon (label/points always read defensively)."""

    label: str
    points: list[list[float]]
    shape_type: str
    group_id: str | None
    description: str | None
    difficult: bool
    flags: dict[str, object]
    attributes: dict[str, object]


class AnnotationFile(TypedDict):
    """An X-AnyLabeling annotation JSON (image size + head/yellow shapes)."""

    imageHeight: int
    imageWidth: int
    imagePath: NotRequired[str]
    shapes: list[ShapeRecord]
