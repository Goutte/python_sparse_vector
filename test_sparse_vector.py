#!/usr/bin/env python

import unittest
import numpy
from future.builtins import range
from sparse_vector import SparseVector


class TestSparseVector(unittest.TestCase):

    def test_initialization_zero(self):
        sv = SparseVector(0)
        self.assertEquals(0, len(sv))

    def test_initialization_non_zero(self):
        sv = SparseVector(10)
        self.assertEquals(10, len(sv))

    def test_initialization_no_default(self):
        sv = SparseVector(1)
        self.assertEquals(0, sv.default)

    def test_initialization_with_default(self):
        sv = SparseVector(1, default_value='test')
        self.assertEquals('test', sv.default)

    def test_initialization_tuple_of_indices_and_values(self):
        sv = SparseVector(([1, 3], [7, 2]))
        self.assertEquals([0, 7, 0, 2], list(sv))

    def test_initialisation_by_dict(self):
        sv = SparseVector({
            4: 6,
            3: 5,
        })
        self.assertEquals([0, 0, 0, 5, 6], sv)

    def test_initialisation_by_dict_with_non_numeric_key(self):
        self.assertRaises(ValueError, SparseVector, {'a': 5})

    def test_initialisation_by_list(self):
        sv = SparseVector([0, 1, 2, 4])
        self.assertEquals([0, 1, 2, 4], sv)

    def test_initialisation_by_generator(self):
        gen = (x for x in (1, 2, 3))
        sv = SparseVector(gen)
        self.assertEquals([1, 2, 3], sv)

    def test_initialisation_with_dtype(self):
        sv = SparseVector([1, 2, 3], dtype=float)
        self.assertEquals([1.0, 2.0, 3.0], sv)

    def test_initialization_with_ndarrays(self):
        keys = numpy.arange(1e5, dtype=numpy.int)
        vals = numpy.arange(1e5)
        sv = SparseVector((keys, vals), size=1e6, default_value=0)
        dense_expected = numpy.zeros(int(1e6))
        dense_expected[keys] = vals
        self.assertEquals(list(dense_expected), list(sv))
        numpy.testing.assert_array_almost_equal(dense_expected, sv)

    def test_random_access_write(self):
        sv = SparseVector(1)
        sv[0] = 'alice'
        self.assertEquals('alice', sv[0])

    def test_random_access_read_present(self):
        sv = SparseVector(2)
        sv[0] = 'brent'
        self.assertEquals('brent', sv[0])

    def test_random_access_read_absent(self):
        sv = SparseVector(2, 'absent')
        sv[1] = 'clint'
        self.assertEquals('absent', sv[0])

    def test_iteration_empty(self):
        sv = SparseVector(3)
        self.assertEquals([0, 0, 0], list(sv))

    def test_iteration_populated(self):
        import numpy
        sv = SparseVector(5)
        sv[1], sv[3] = 1, 2
        self.assertEquals([0, 1, 0, 2, 0], list(sv))

    def test_membership_absent(self):
        sv = SparseVector(5)
        sv[2], sv[3], = 1, 2
        self.assertEquals(False, 3 in sv)

    def test_membership_present(self):
        sv = SparseVector(5)
        sv[2], sv[3], = 1, 2
        self.assertEquals(True, 2 in sv)

    def test_string_representations_float_by_default(self):
        sv = SparseVector(5, 0)
        sv[3], sv[4] = 5, 6
        self.assertEquals('[0.0, 0.0, 0.0, 5.0, 6.0]', repr(sv))
        self.assertEquals('[0.0, 0.0, 0.0, 5.0, 6.0]', str(sv))

    def test_string_representations_int(self):
        sv = SparseVector(5, 0, dtype=int)
        sv[3], sv[4] = 5, 6
        self.assertEquals('[0, 0, 0, 5, 6]', repr(sv))
        self.assertEquals('[0, 0, 0, 5, 6]', str(sv))

    def test_access_with_negative_index(self):
        sv = SparseVector([0, 1, 2, 4])
        self.assertEqual(4, sv[-1])

    def test_access_with_negative_index_with_no_value(self):
        sv = SparseVector(5, 0)
        self.assertEqual(0, sv[-1])

    def test_slice(self):
        sv = SparseVector([0, 1, 2, 4], 10)
        self.assertEqual([1, 2], sv[1:3])

    def test_extended_slice(self):
        sv = SparseVector([0, 1, 2, 3, 4, 5, 6, ])
        self.assertEqual([1, 3, 5], sv[1:6:2])

    def test_extended_slice_with_negative_stop(self):
        sv = SparseVector([0, 1, 2, 3, 4, 5, 6, ])
        self.assertEqual([1, 3, 5], sv[1:-1:2])

    def test_slice_reversal_full(self):
        sv = SparseVector([1, 2, 3])
        self.assertEqual([3, 2, 1], sv[::-1])

    def test_slice_reversal_empty(self):
        sv = SparseVector(4)
        self.assertEqual([0, 0, 0, 0], sv[::-1])

    def test_slice_with_list_read(self):
        sv = SparseVector([1, 2, 3, 4, 5])
        ip = [0, 2, 4]
        self.assertEqual([1, 3, 5], sv[ip])

    def test_slice_with_list_write(self):
        sv = SparseVector([1, 2, 3, 4, 5])
        ip = [0, 2, 4]
        sv[ip] = [6, 7, 8]
        self.assertEqual([6, 2, 7, 4, 8], list(sv))
        self.assertEqual([6, 2, 7, 4, 8], sv)
        numpy.testing.assert_array_almost_equal([6, 2, 7, 4, 8], sv)

    def test_slice_with_list_write_2(self):
        sv = SparseVector(([1, 2, 4, 5], [1, 2, 3, 4]))
        ip = [0, 3, 2, 4]
        sv[ip] = [6, 7, 9, 8]
        self.assertEqual([6, 1, 9, 7, 8, 4], list(sv))

    def test_reversed(self):
        sv = SparseVector([1, 2, 3])
        self.assertEqual([3, 2, 1], list(reversed(sv)))

    def test_sorted(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        self.assertEquals([0, 0, 0, 1, 1], list(sorted(sv)))

    def test_get_out_of_bounds(self):
        sv = SparseVector(1)
        self.assertEquals(0, sv[1])

    def test_set_out_of_bounds(self):
        sv = SparseVector(1)
        sv[100] = 1
        self.assertEquals(101, len(sv))

    def test_present_item_removal(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        del sv[0]
        self.assertEquals([0, 0, 0, 0, 1], sv)

    def test_missing_item_removal(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        del sv[1]
        self.assertEquals([1, 0, 0, 0, 1], sv)

    def test_slice_removal(self):
        sv = SparseVector(range(10), 0)
        del sv[3:5]
        self.assertEquals([0, 1, 2, 0, 0, 5, 6, 7, 8, 9], sv)

    def test_append(self):
        sv = SparseVector(1, 0)
        sv.append(1)
        self.assertEquals([0, 1], sv)

    def test_clone(self):
        a = SparseVector([1, 2, 3])
        b = a[:]
        b.append(4)
        self.assertEquals([1, 2, 3], a)
        self.assertEquals([1, 2, 3, 4], b)

    def test_concatenation(self):
        a = SparseVector([1, 2, 3])
        b = SparseVector([4, 5, 6])
        c = a + b
        self.assertEquals([1, 2, 3], a)
        self.assertEquals([4, 5, 6], b)
        self.assertEquals([1, 2, 3, 4, 5, 6], c)

    def test_in_place_concatenation(self):
        a = SparseVector([1, 2, 3])
        b = SparseVector([4, 5, 6])
        a += b
        self.assertEquals([1, 2, 3, 4, 5, 6], a)
        self.assertEquals([4, 5, 6], b)

    def test_equality(self):
        a = SparseVector([1, 2, 3])
        b = SparseVector([1, 2, 3])
        self.assertTrue(a == b)
        self.assertTrue(not a != b)
        self.assertEquals(a, b)
        self.assertTrue(b == a)
        self.assertTrue(not b != a)
        self.assertEquals(b, a)

    def test_inequality_same_length(self):
        a = SparseVector([1, 2, 3])
        b = SparseVector([1, 0, 3])
        self.assertTrue(a != b)
        self.assertTrue(not a == b)
        self.assertNotEqual(a, b)
        self.assertTrue(b != a)
        self.assertTrue(not b == a)
        self.assertNotEqual(b, a)

    def test_inequality_left_longer(self):
        a = SparseVector([1, 2, 3, 4])
        b = SparseVector([1, 2, 3])
        self.assertTrue(a != b)
        self.assertTrue(not (a == b))
        self.assertNotEqual(a, b)
        self.assertTrue(b != a)
        self.assertTrue(not (b == a))
        self.assertNotEqual(b, a)

    def test_less_than(self):
        a = SparseVector([1, 2, 3, 0])
        b = SparseVector([1, 2, 4, 5])
        self.assertTrue(a < b)
        self.assertFalse(a == b)
        self.assertFalse(a >= b)
        self.assertFalse(a > b)

    def test_greater_than(self):
        a = SparseVector([1, 2, 3, 0])
        b = SparseVector([1, 2, 4, 5])
        self.assertTrue(b > a)
        self.assertFalse(b == a)
        self.assertFalse(b <= a)
        self.assertFalse(b < a)

    def test_multiply(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        sv4 = sv * 4
        self.assertEquals([1, 0, 0, 0, 1], sv)
        self.assertEquals(
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1], sv4)
        self.assertEquals(len(sv) * 4, len(sv4))

    def test_multiply_in_place(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        sv *= 4
        self.assertEquals(
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1], sv)

    def test_count_value(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        self.assertEquals(2, sv.count(1))

    def test_count_default_value(self):
        sv = SparseVector(100, 1)
        sv[5] = 1
        self.assertEquals(100, sv.count(1))

    def test_extend(self):
        sv = SparseVector([1, 2, 3])
        sv.extend((4, 5, 6))
        self.assertEquals([1, 2, 3, 4, 5, 6], sv)

    def test_index_value(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        self.assertEquals(0, sv.index(1))

    def test_index_default_value(self):
        sv = SparseVector({0: 1, 4: 1}, 0)
        self.assertEquals(1, sv.index(0))

    def test_index_absent_default_value(self):
        sv = SparseVector([1, 2, 3], 0)
        self.assertRaises(ValueError, sv.index, 0)

    def test_index_absent_value(self):
        sv = SparseVector(1, 0)
        self.assertRaises(ValueError, sv.index, 2)

    def test_pop_no_value(self):
        sv = SparseVector(4)
        self.assertEquals(0, sv.pop())

    def test_pop_empty(self):
        sv = SparseVector(0)
        self.assertRaises(IndexError, sv.pop)

    def test_pop_value(self):
        sv = SparseVector([1, 2, 3])
        popped = sv.pop()
        self.assertEquals(3, popped)
        self.assertEquals(2, len(sv))
        self.assertEquals([1, 2], sv)

    def test_push_value(self):
        sv = SparseVector([1, 2, 3])
        sv.push(4)
        self.assertEquals(4, len(sv))
        self.assertEquals([1, 2, 3, 4], sv)

    def test_remove_value(self):
        sv = SparseVector([1, 2, 3])
        sv.remove(2)
        self.assertEquals(3, len(sv))
        self.assertEquals([1, 0, 3], sv)

    def test_remove_only_first_value(self):
        sv = SparseVector([2, 2, 3])
        sv.remove(2)
        self.assertEquals(3, len(sv))
        self.assertEquals([0, 2, 3], sv)

    def test_remove_non_value(self):
        sv = SparseVector([1, 2, 3])
        self.assertRaises(ValueError, sv.remove, 4)

    def test_remove_default_value_does_nothing(self):
        sv = SparseVector(4, default_value=1)
        sv.remove(1)
        self.assertEquals([1, 1, 1, 1], sv)

if __name__ == '__main__':
    unittest.main()
