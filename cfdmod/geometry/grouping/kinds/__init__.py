"""Per-kind grouping spec + apply implementations.

Each module here defines a Pydantic spec with a unique ``kind`` literal
and an ``apply_<kind>(spec, mesh, allowed)`` function returning
``dict[str, np.ndarray]`` of triangle indices into the parent mesh.
"""
