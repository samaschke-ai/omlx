import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_masked_row_matches_gather_qmm(dtype):
    if not glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_masked_row"):
        pytest.skip("masked-row native kernel is unavailable")

    mx.random.seed(7)
    experts, tokens, topk = 8, 6, 6
    input_dims, output_dims = 512, 256
    routes = tokens * topk

    x = mx.random.normal((tokens, 1, input_dims)).astype(dtype)
    weight = mx.random.normal((experts, output_dims, input_dims)).astype(dtype)
    qweight, scales = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
    indices = (mx.arange(routes, dtype=mx.uint32) * 5) % experts
    route_mask = (mx.arange(routes) % 3) != 0

    expanded = x[mx.arange(routes) // topk]
    reference = mx.gather_qmm(
        expanded,
        qweight,
        scales,
        rhs_indices=indices,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    reference = mx.where(route_mask[:, None, None], reference, 0)
    actual = glm_fast.deepseek_mxfp4_gather_qmm_masked_row(
        x,
        qweight,
        scales,
        indices,
        route_mask,
        0,
    )
    mx.eval(reference, actual)

    assert actual.shape == (routes, 1, output_dims)
    mask3 = route_mask[:, None, None]
    assert mx.all(mx.where(mask3, 0, actual) == 0).item()
    assert mx.allclose(actual, reference, rtol=2e-2, atol=2e-2).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_masked_pair_matches_two_gather_qmm_calls(dtype):
    if not glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_masked_pair"):
        pytest.skip("masked-pair native kernel is unavailable")

    mx.random.seed(11)
    experts, tokens, topk = 8, 6, 6
    input_dims, output_dims = 512, 256
    routes = tokens * topk
    x = mx.random.normal((tokens, 1, input_dims)).astype(dtype)
    weights = [
        mx.random.normal((experts, output_dims, input_dims)).astype(dtype)
        for _ in range(2)
    ]
    quantized = [
        mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
        for weight in weights
    ]
    indices = (mx.arange(routes, dtype=mx.uint32) * 5) % experts
    route_mask = (mx.arange(routes) % 3) != 0
    mask3 = route_mask[:, None, None]
    expanded = x[mx.arange(routes) // topk]
    references = [
        mx.where(
            mask3,
            mx.gather_qmm(
                expanded,
                qweight,
                scales,
                rhs_indices=indices,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ),
            0,
        )
        for qweight, scales in quantized
    ]
    reference = mx.concatenate(references, axis=-1)
    actual = glm_fast.deepseek_mxfp4_gather_qmm_masked_pair(
        x,
        quantized[0][0],
        quantized[0][1],
        quantized[1][0],
        quantized[1][1],
        indices,
        route_mask,
        0,
    )
    mx.eval(reference, actual)

    assert actual.shape == (routes, 1, 2 * output_dims)
    assert mx.all(mx.where(mask3, 0, actual) == 0).item()
    assert mx.allclose(actual, reference, rtol=2e-2, atol=2e-2).item()

    mismatched_weight = mx.random.normal(
        (experts - 1, output_dims, input_dims)
    ).astype(dtype)
    mismatched_qweight, mismatched_scales = mx.quantize(
        mismatched_weight, group_size=32, bits=4, mode="mxfp4"
    )
    with pytest.raises(ValueError, match="unsupported shape"):
        glm_fast.deepseek_mxfp4_gather_qmm_masked_pair(
            x,
            quantized[0][0],
            quantized[0][1],
            mismatched_qweight,
            mismatched_scales,
            indices,
            route_mask,
            0,
        )
