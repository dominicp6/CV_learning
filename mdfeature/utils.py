#!/usr/bin/env python

"""
   Author: Dominic Phillips (dominicp6)
"""

import numpy as np


def remove_corresponding_entries_with_nans(x: list, y: list) -> [list, list]:
    """
    Given two lists, removes the indexes of entries that are nan in either list.

    Parameters
    ----------
    x: first list
    y: second list

    Returns
    -------
    x_: first list processed
    y_: second list processed

    """
    nans_in_x = set(
        [index for sublist in np.argwhere(np.isnan(x)) for index in sublist]
    )
    nans_in_y = set(
        [index for sublist in np.argwhere(np.isnan(y)) for index in sublist]
    )

    if len(nans_in_x) > 0 or len(nans_in_y) > 0:
        nans_in_either = list(nans_in_x.union(nans_in_y))
        x_ = np.delete(x, nans_in_either)
        y_ = np.delete(y, nans_in_either)

        return x_, y_
    else:
        return x, y
