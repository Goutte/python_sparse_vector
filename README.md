Sparse Vector
=============

Available on [https://pypi.python.org/pypi/sparse_vector].

Inspired by [`sparse_list`](https://pypi.python.org/pypi/sparse_vector),
a dictionary-of-keys implementation of a sparse list in python.

A _sparse vector_ is a 1D numerical list where most (say, more than 95% of)
values will be `0` (or some other default) and for reasons of memory efficiency
you don't wish to store these.
(cf. [Sparse array](http://en.wikipedia.org/wiki/Sparse_array))

This implementation has a similar interface to Python's 1D `numpy.ndarray`
but stores the values and indices in linked lists to preserve memory.

sparse_vector is for numerical data, if you want any type of data, have a look
at [`sparse_list`](https://pypi.python.org/pypi/sparse_list)

If you need 2D matrices, have a look at `scipy.sparse`, they also have a
linked lists implementation, `lil_matrix`.


Installation
------------

Installation is simply:

``` bash
$ pip install sparse_vector
```

Usage
-----

See the [unit-tests](https://github.com/Goutte/python_sparse_vector/blob/master/test_sparse_vector.py)!


Contributing
------------

1. Fork it
2. Create your feature branch (``git checkout -b my-new-feature``)
3. Commit your changes (``git commit -am 'Add some feature'``)
4. Ensure the tests pass for all Pythons in
   `.travis.yml <https://github.com/johnsyweb/python_sparse_vector/blob/master/.travis.yml>`__
5. Push to the branch (``git push origin my-new-feature``)
6. Create new Pull Request


Thanks
------

- [johnsyweb](http://johnsy.com/about/) for the original `sparse_list`.