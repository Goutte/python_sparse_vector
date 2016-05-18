"""

A "sparse vector" is basically a 1D `numpy.ndarray` where most (say, more
than 95% of) values will be None (or some other default) and for reasons
of memory efficiency you don't wish to store these. cf. Sparse array:

    http://en.wikipedia.org/wiki/Sparse_array

"""

import numpy as np
from future.builtins import range
from six.moves import zip_longest


class SparseVector(object):
    """
    This implementation has a similar interface to `numpy`'s `ndarray` but
    stores the indices and values in two `ndarray`s to preserve memory.

    default_value : numerical, optional
        The default value that fills most of this vector.
        The value should be compatible with `dtype`.
    size : int, optional
        When not provided, the sparse vector will be as big as it needs.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    """

    def __init__(self, arg, default_value=0, size=None, dtype=np.float):
        self.default = default_value
        self.dtype = dtype
        self.indices = np.array([], dtype=np.int)
        self.values = np.array([], dtype=self.dtype)
        if isinstance(arg, (int, float)):  # 1e6 is a float
            self.size = int(arg)
        elif isinstance(arg, dict):
            self.__initialise_from_dict(arg)
        elif isinstance(arg, tuple):
            self.__initialise_from_tuple(arg)
        else:
            self.__initialise_from_iterable(arg)
        if size is not None:
            self.size = int(size)

    def __len__(self):
        return self.size

    def __setitem__(self, index, value):
        def _is_array(_v):
            return isinstance(_v, (list, np.ndarray))
        is_array_value = _is_array(value)

        if _is_array(index):

            new = np.setdiff1d(index, self.indices, assume_unique=False)
            old = np.intersect1d(self.indices, index, assume_unique=False)

            if old.size > 0:
                internal_indices = self.__internal_indices_of_indices(old)

                if internal_indices.size > 0:
                    if is_array_value:
                        ei = np.where(np.in1d(index, old))[0]
                        val = np.array(value)[ei]
                    else:
                        val = value
                    self.values[internal_indices] = val

            if new.size > 0:
                ei = np.where(np.in1d(index, new))[0]
                only_new_values = []
                if is_array_value:
                    only_new_values = np.array(value)[ei]
                new_indices = []
                new_values = []
                for k, v in enumerate(np.array(index)[ei]):
                    if is_array_value:
                        val = only_new_values[k]
                    else:
                        val = value
                    new_indices.append(v)
                    new_values.append(val)
                    self.size = max(v + 1, self.size)

                self.indices = np.append(self.indices, new_indices)
                self.values = np.append(self.values, new_values)
        else:
            i = self.__internal_index_of_index(index)
            if i is None:
                self.indices = np.append(self.indices, [index])
                self.values = np.append(self.values, [value])
            else:
                self.values[i] = value
            self.size = max(index + 1, self.size)

    def __getitem__(self, index):
        try:  # [start:stop:step]
            s = slice(index.start, index.stop, index.step).indices(self.size)
            return [self[i] for i in range(*s)]
        except AttributeError:
            pass
        try:  # [iterable]
            return [self[i] for i in index]
        except TypeError:
            pass
        i = slice(index).indices(self.size)[1]
        k = np.where(self.indices == i)[0]
        return self.values[k[0]] if k.size > 0 else self.default

    def __delitem__(self, index):
        try:
            s = slice(index.start, index.stop, index.step).indices(self.size)
            for j in range(*s):
                i = self.__internal_index_of_index(j)
                self.indices = np.delete(self.indices, i)
                self.values = np.delete(self.values, i)
        except AttributeError:
            i = self.__internal_index_of_index(index)
            if i is not None:
                self.indices = np.delete(self.indices, i)
                self.values = np.delete(self.values, i)

    def __delslice__(self, start, stop):
        for index in range(start, stop):
            self.__delitem__(index)

    def __iter__(self):
        return np.nditer(self.densify())

    def __contains__(self, value):
        return value in self.values

    def __repr__(self):
        return '[{}]'.format(', '.join([str(e) for e in self]))

    def __add__(self, other):
        result = self[:]
        return result.__iadd__(other)

    def __iadd__(self, other):
        for element in other:
            self.append(element)
        return self

    def __initialise_from_dict(self, arg):
        self.values = np.array(arg.values(), dtype=self.dtype)
        self.indices = np.array(arg.keys(), dtype=np.int)
        self.size = np.max(self.indices) + 1

    def __initialise_from_tuple(self, arg):
        indices, values = arg
        assert len(indices) == len(values), \
            "You must provide a tuple of two vectors (indices, values),\n" \
            "and indices must be integers."
        self.values = np.array(values, dtype=self.dtype)
        self.indices = np.array(indices, dtype=np.int)
        self.size = np.max(self.indices) + 1

    def __initialise_from_iterable(self, arg):
        self.values = np.array(list(arg), dtype=self.dtype)
        self.indices = np.arange(len(self.values), dtype=np.int)
        self.size = len(self.values)

    def __internal_indices_of_indices(self, indices):
        return np.where(np.in1d(self.indices, indices))[0]

    def __internal_index_of_index(self, index):
        if index < 0:
            index += self.size
        k = np.where(self.indices == index)[0]
        return k[0] if k.size > 0 else None

    def __internal_index_of_value(self, value):
        k = np.where(self.values == value)[0]
        return k[0] if k.size > 0 else None

    def __eq__(self, other):
        return all(a == b for a, b in zip_longest(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return any(a < b for a, b in zip_longest(self, other))

    def __ge__(self, other):
        return not self.__lt__(other)

    def __mul__(self, multiplier):
        result = []
        for _ in range(multiplier):
            result += self[:]
        return result

    def iter(self):
        """
        Return a sparse iterator that will not densify. Very slow. VERY.
        """
        i = 0
        while i < self.size:
            yield self[i]
            i += 1

    def densify(self):
        """
        Return a dense representation of this vector, as a `numpy.ndarray` of
        shape `(size,)`. This might blow up your RAM when `size` is big.
        """
        dense = np.full(self.size, fill_value=self.default, dtype=self.dtype)
        dense[self.indices] = self.values
        return dense

    def append(self, element):
        """
        Append element, increasing size by exactly one.
        """
        self.values = np.append(self.values, [element])
        self.indices = np.append(self.indices, [self.size])
        self.size += 1

    push = append

    def count(self, value):
        """
        Return the number of occurrences of `value` in this vector.
        Counts the default values too if `value` is equal to the default value.
        """
        return np.where(self.values == value)[0].size + (
            self.size - len(self.values) if value == self.default else 0
        )

    def extend(self, iterable):
        """
        Extend sparse_list by appending elements from the iterable.
        """
        self.__iadd__(iterable)

    def index(self, value):
        """
        Return the first found index of `value`.
        Raises ValueError when the value is not present.
        """
        if value == self.default:
            i = 0
            while i < self.size:
                if self[i] == value:
                    return i
                i += 1
        else:
            i = self.__internal_index_of_value(value)
            if i is not None:
                return self.indices[i]
        raise ValueError('{} not in SparseVector'.format(value))

    def pop(self):
        """
        Remove and return the value at the end of this vector.
        Warn: MAY NOT WORK AS EXPECTED as we don't sort our indices internally.
        Raises IndexError when the vector is empty.
        """
        if self.size < 1:
            raise IndexError('pop from empty SparseVector')
        value = self[-1]
        del self[-1]
        self.size -= 1
        return value

    def remove(self, value):
        """
        Remove the first found occurrence of `value`.
        If you try to remove the default value, nothing will happen.
        Raises ValueError when the `value` is absent from the vector.
        """
        if value == self.default:
            return
        i = self.__internal_index_of_value(value)
        if i is not None:
            self.indices = np.delete(self.indices, i)
            self.values = np.delete(self.values, i)
        else:
            raise ValueError('{} not in SparseVector'.format(value))
