#!/usr/bin/python
# -*- coding: utf-8 -*-
def iter_widgets(base, typename, direction=0):
    """Yields all widgets of type `typename` starting from `base` widget.
    The search can be restricted to upward (`direction`=-1), downward (`direction`=1) or both (`direction`=0)."""
    if isinstance(typename, str):
        typename = [typename]
    if base.typename in typename:
        yield base
    # Search down in the object tree
    if direction >= 0:
        for obj in base.children:
            if obj.typename in typename:
                yield obj

        for obj in base.children:
            found = searchFirstOccurrence(obj, typename, direction=1)
            if found is not None:
                yield found
        # Continue upwards
        if direction == 0:
            direction = -1
    # Search up in the object tree
    # Note: exclude siblings - just look for parents
    if direction < 0:
        found = searchFirstOccurrence(base.parent, typename, direction=-1)
        if found is not None:
            yield found
    # Nothing found
    yield None


def searchFirstOccurrence(base, typename, direction=0):
    """Search for the nearest occurrence of a widget of type `typename` starting from `base`.
    The search can be restricted to upward (`direction`=-1), downward (`direction`=1) or both (`direction`=0)."""
    for wg in iter_widgets(base, typename, direction):
        if wg:
            return wg