import benchmark
import numpy as np

from sparse_list import SparseList
from sparse_vector import SparseVector


class BenchmarkAbstract(benchmark.Benchmark):

    each = 10            # number of runs
    full_size = 1000000  # total size of sparse lists and vectors
    data_size = 10000    # actual data size of sparse lists and vectors

    def setUp(self):
        # Can also specify tearDown, eachSetUp, and eachTearDown
        self.a = np.random.rand(self.data_size)
        self.k = np.arange(self.data_size)

        self.sl = SparseList(dict(zip(self.k, self.a)))
        self.sl.size = self.full_size

        self.sv = SparseVector((self.k, self.a), size=self.full_size)


class BenchmarkInit(BenchmarkAbstract):

    def test_list_init(self):
        sl = SparseList(self.full_size)

    def test_vector_init(self):
        sv = SparseVector(self.full_size)

    def test_list_init_with_values(self):
        sl = SparseList(dict(zip(self.k, self.a)))
        sl.size = self.full_size

    def test_vector_init_with_values(self):
        sv = SparseVector((self.k, self.a), size=self.full_size)


class BenchmarkGet(BenchmarkAbstract):

    def test_list_get_with_iterable_in_slice(self):
        dummy = self.sl[self.k]

    def test_vector_get_with_iterable_in_slice(self):
        dummy = self.sv[self.k]


class BenchmarkSet(BenchmarkAbstract):

    def test_list_set_with_iterables_in_slice_absent(self):
        sl = SparseList(self.full_size)
        sl[self.k] = self.a

    def test_list_set_with_iterables_in_slice_present(self):
        self.sl[self.k] = self.a

    def test_vector_set_with_iterables_in_slice_absent(self):
        sv = SparseVector(self.full_size)
        sv[self.k] = self.a

    def test_vector_set_with_iterables_in_slice_present(self):
        self.sv[self.k] = self.a


class BenchmarkDensify(BenchmarkAbstract):

    def test_list_densify_with_list(self):
        list(self.sl)

    def test_vector_densify_with_list(self):
        list(self.sv)

    def test_vector_densify_with_numpy(self):
        np.array(self.sv)

    def test_vector_densify_with_densify(self):
        self.sv.densify()


class BenchmarkIterate(BenchmarkAbstract):

    def test_list_iterate(self):
        for k, v in enumerate(self.sl):
            pass

    def test_vector_iterate(self):
        for k, v in enumerate(self.sv):
            pass

    def test_vector_iterate_sparsely(self):
        for k, v in enumerate(self.sv.iter()):
            pass


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")
