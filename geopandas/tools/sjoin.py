import numpy as np
import pandas as pd
from shapely import prepared


def sjoin(left_df, right_df, how='inner', op='intersects',
          lsuffix='left', rsuffix='right'):
    """Spatial join of two GeoDataFrames.

    Parameters
    ----------
    left_df, right_df : GeoDataFrames
    how : string, default 'inner'
        The type of join:

        * 'left': use keys from left_df; retain only left_df geometry column
        * 'right': use keys from right_df; retain only right_df geometry column
        * 'inner': use intersection of keys from both dfs; retain only
          left_df geometry column
    op : string, default 'intersection'
        Binary predicate, one of {'intersects', 'contains', 'within'}.
        See http://toblerity.org/shapely/manual.html#binary-predicates.
    lsuffix : string, default 'left'
        Suffix to apply to overlapping column names (left GeoDataFrame).
    rsuffix : string, default 'right'
        Suffix to apply to overlapping column names (right GeoDataFrame).

    """
    import rtree
    left_df = left_df.copy()
    right_df = right_df.copy()

    allowed_hows = ['left', 'right', 'inner']
    if how not in allowed_hows:
        raise ValueError('`how` was "%s" but is expected to be in %s' %
                         (how, allowed_hows))

    allowed_ops = ['contains', 'within', 'intersects']
    if op not in allowed_ops:
        raise ValueError('`op` was "%s" but is expected to be in %s' %
                         (op, allowed_ops))

    if how == "right":
        drop_index_name = False
    elif op == "within":
        drop_index_name = right_df.index.names == [None]
    else:
        drop_index_name = left_df.index.names == [None]

    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df

    if left_df.crs != right_df.crs:
        print('Warning: CRS does not match!')

    l_index = 'tmp_index_%s' % lsuffix
    r_index = 'tmp_index_%s' % rsuffix

    r_index_names = _reset_df_index(right_df, suffix='right')
    l_index_names = _reset_df_index(left_df, suffix='left')

    tree_idx = rtree.index.Index()
    right_df_bounds = right_df.geometry.apply(lambda x: x.bounds)
    for i in right_df_bounds.index:
        tree_idx.insert(i, right_df_bounds[i])

    idxmatch = (left_df.geometry.apply(lambda x: x.bounds)
                .apply(lambda x: list(tree_idx.intersection(x))))
    idxmatch = idxmatch[idxmatch.apply(len) > 0]

    if idxmatch.shape[0] > 0:
        # if output from  join has overlapping geometries
        r_idx = np.concatenate(idxmatch.values)
        l_idx = np.concatenate([[i] * len(v) for i, v in idxmatch.iteritems()])

        # Vectorize predicate operations
        def find_intersects(a1, a2):
            return a1.intersects(a2)

        def find_contains(a1, a2):
            return a1.contains(a2)

        predicate_d = {'intersects': find_intersects,
                       'contains': find_contains,
                       'within': find_contains}

        check_predicates = np.vectorize(predicate_d[op])

        result = (
                  pd.DataFrame(
                      np.column_stack(
                          [l_idx,
                           r_idx,
                           check_predicates(
                               left_df.geometry
                               .apply(lambda x: prepared.prep(x))[l_idx],
                               right_df[right_df.geometry.name][r_idx])
                           ]))
                   )

        result.columns = [l_index, r_index, 'match_bool']
        result = result[result['match_bool'] == 1].drop('match_bool', axis=1)

    else:
        # when output from the join has no overlapping geometries
        result = pd.DataFrame(columns=[l_index, r_index])

    if op == "within":
        # within implemented as the inverse of contains; swap names
        left_df, right_df = right_df, left_df
        to_rename = {l_index: r_index, r_index: l_index}
        to_rename.update(dict(zip(l_index_names, r_index_names)))
        to_rename.update(dict(zip(r_index_names, l_index_names)))
        result.rename(columns=to_rename, inplace=True)

    if how == 'inner' or how == 'left':
        result.set_index(l_index, inplace=True)
        result = left_df.merge(result, left_index=True, right_index=True,
                               how=how)
        result = (
                result.merge(
                            right_df.drop(right_df.geometry.name, axis=1),
                            how=how, left_on=r_index, right_index=True,
                            suffixes=('_%s' % lsuffix, '_%s' % rsuffix))
                )
        result.set_index(l_index_names, inplace=True)
        result.drop(r_index, axis=1, inplace=True)

    elif how == 'right':
        result = (
                left_df.drop(left_df.geometry.name, axis=1)
                .merge(result.merge(
                                   right_df, left_on=r_index,
                                   right_index=True, how=how),
                       left_index=True, right_on=l_index, how=how)
                )
        result.set_index(r_index_names, inplace=True)
        result.drop([r_index, l_index], axis=1, inplace=True)

    if drop_index_name:
        result.index.set_names(None, inplace=True)
    return result


def _reset_df_index(pd_df, suffix=''):
    '''
    Parameters
    ----------
    pd_df : pandas dataframe
    suffix : string
        optional suffix to add to index names

    modifies pd_df in place by moving the index columns to df columns,
    renaming the index columns via the suffix

    Returns list of newly named columns
    '''

    if pd_df.index.names == [None]:
        pd_df.index.set_names('index', inplace=True)
    if suffix != '':
        suffix = '_%s' % suffix

    index_names = ["%s%s" % (name, suffix) for name in pd_df.index.names]
    for i, name in enumerate(index_names):
        new_name = name
        j = 0
        while new_name in pd_df.columns:
            new_name = '%s_%s' % (name, j)
            j += 1
        if new_name != name:
            index_names[i] = new_name

    pd_df.index.set_names(index_names, inplace=True)
    pd_df.reset_index(inplace=True)

    return index_names
