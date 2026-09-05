import pytest
import torch

from . import base, consts

# Cover 1-D, multi-D and large flatten spans. Mirrors `test_put_.py`.
PUT_BENCH_SHAPES = [
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]


class PutBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = PUT_BENCH_SHAPES
        return None


class PutOutBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        self.shapes = PUT_BENCH_SHAPES
        return None


def put_input_fn(accumulate):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        numel = inp.numel()
        # Write into roughly half of the positions, capped so the core-level
        # 1G shape does not allocate ~512M indices and dominate the run.
        count = max(1, min(numel // 2, 2**20))
        # Allow duplicate indices only when accumulating; keep them unique
        # otherwise so the overwrite path is deterministic.
        if accumulate:
            index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
        else:
            index = torch.randperm(numel, device=device)[:count].to(torch.int64)
        source = torch.randn(count, dtype=dtype, device=device, requires_grad=False)
        yield inp, index, source, accumulate

    return inner


def put_out_input_fn(accumulate):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
        numel = inp.numel()
        count = max(1, min(numel // 2, 2**20))
        if accumulate:
            index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
        else:
            index = torch.randperm(numel, device=device)[:count].to(torch.int64)
        source = torch.randn(count, dtype=dtype, device=device, requires_grad=False)
        out = torch.empty_like(inp)
        # `put.out` takes (self, index, source, accumulate, *, out)
        yield inp, index, source, accumulate, {"out": out}

    return inner


@pytest.mark.put
def test_put():
    bench = PutBenchmark(
        op_name="put",
        torch_op=torch.ops.aten.put,
        input_fn=put_input_fn(accumulate=False),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.put
def test_put_accumulate():
    bench = PutBenchmark(
        op_name="put",
        torch_op=torch.ops.aten.put,
        input_fn=put_input_fn(accumulate=True),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.put_out
def test_put_out():
    bench = PutOutBenchmark(
        op_name="put_out",
        torch_op=torch.ops.aten.put.out,
        input_fn=put_out_input_fn(accumulate=False),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.put
def test_put_out_accumulate():
    bench = PutOutBenchmark(
        op_name="put",
        torch_op=torch.ops.aten.put.out,
        input_fn=put_out_input_fn(accumulate=True),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
