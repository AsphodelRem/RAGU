"""
Tests for BatchGenerator utility class.
"""
from ragu.common.batch_generator import BatchGenerator


class TestBatchGenerator:
    def test_get_batches_basic(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        batch_size = 3
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 4  # 10 items / 3 per batch = 4 batches
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7, 8, 9]
        assert batches[3] == [10]  # Last batch has remainder

    def test_get_batches_exact_division(self):
        data = list(range(1, 13))  # 12 items
        batch_size = 4
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 3
        assert all(len(batch) == 4 for batch in batches)
        assert batches[0] == [1, 2, 3, 4]
        assert batches[1] == [5, 6, 7, 8]
        assert batches[2] == [9, 10, 11, 12]

    def test_get_batches_single_batch(self):
        data = [1, 2, 3]
        batch_size = 10
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_get_batches_batch_size_one(self):
        data = [1, 2, 3, 4, 5]
        batch_size = 1
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 5
        assert batches[0] == [1]
        assert batches[1] == [2]
        assert batches[4] == [5]

    def test_get_batches_empty_data(self):
        data = []
        batch_size = 5
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 0

    def test_len_method(self):
        data = list(range(1, 26))  # 25 items
        batch_size = 7
        gen = BatchGenerator(data, batch_size)

        assert len(gen) == 4

    def test_len_method_exact_division(self):
        data = list(range(1, 21))  # 20 items
        batch_size = 5
        gen = BatchGenerator(data, batch_size)

        assert len(gen) == 4

    def test_len_method_single_batch(self):
        data = [1, 2, 3]
        batch_size = 10
        gen = BatchGenerator(data, batch_size)

        assert len(gen) == 1

    def test_len_method_empty_data(self):
        data = []
        batch_size = 5
        gen = BatchGenerator(data, batch_size)

        assert len(gen) == 0

    def test_mixed_type_data(self):
        data = [1, "two", 3.0, None, True, {"key": "value"}]
        batch_size = 2
        gen = BatchGenerator(data, batch_size)

        batches = list(gen.get_batches())

        assert len(batches) == 3
        assert batches[0] == [1, "two"]
        assert batches[1] == [3.0, None]
        assert batches[2] == [True, {"key": "value"}]

    def test_generator_reusability(self):
        data = [1, 2, 3, 4, 5]
        batch_size = 2
        gen = BatchGenerator(data, batch_size)

        batches1 = list(gen.get_batches())
        batches2 = list(gen.get_batches())

        assert batches1 == batches2
        assert len(batches1) == 3
