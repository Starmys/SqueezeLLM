import time
import torch
import triton
import triton.language as tl
from scipy.sparse import csc_array
import quant_cuda


batch_size = 1
in_features = 4096
out_features = 4096
bits = 4
sparse_rate = 0.0045
top_x = 10

pit_block_size = 32

sparse_nnz = round(in_features * out_features * sparse_rate)


def profile(func):
    for _ in range(200):
        func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        func()
    torch.cuda.synchronize()
    end = time.time()
    return end - start


@triton.jit
def triton_quant_4bit_mv_kernel(
    inputs, weight, lut, outputs,
    batch, K, N,
    stride_inputs_0, stride_inputs_1,
    stride_weight_0, stride_weight_1,
    stride_lut_0, stride_lut_1,
    stride_outputs_0, stride_outputs_1,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    block_id_n = tl.program_id(axis=0)
    block_id_k = tl.program_id(axis=1)

    # pid = tl.program_id(axis=0)
    # GROUP_SIZE_K = 8
    # num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_K * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_k = group_id * GROUP_SIZE_K
    # group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    # block_id_k = first_pid_k + (pid % group_size_k)
    # block_id_n = (pid % num_pid_in_group) // group_size_k

    off_n = block_id_n * BLOCK_SIZE_N
    off_k = block_id_k * BLOCK_SIZE_K

    offs_q = off_k // 8 + tl.arange(0, BLOCK_SIZE_K // 8)
    offs_k = off_k + tl.arange(0, BLOCK_SIZE_K)
    offs_n = off_n + tl.arange(0, BLOCK_SIZE_N)
    offs_l = tl.arange(0, 16)

    l = tl.load(lut + offs_n[:, None] * stride_lut_0 + offs_l * stride_lut_1)  # [BN, 16]
    b = tl.load(weight + offs_n[:, None] * stride_weight_0 + offs_q[None, :])  # [BN, BK // 8]
    l[:, b]

    # for i in range(batch):
    #     a = tl.load(inputs + i * stride_inputs_0 + off_k + offs_k)  # [BK]
    #     c = tl.sum(a[None, :] * b, 1)  # [BN]
    #     tl.atomic_add(outputs + i * stride_outputs_0 + row_idx, c, mask=mask_n)

    a = tl.load(inputs + off_k + offs_k)  # [BK]
    c = tl.sum(a[None, :] * b, 1)  # [BN]
    tl.atomic_add(outputs + row_idx, c, mask=mask_n)


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': pit_block_size}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': pit_block_size}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': pit_block_size}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': pit_block_size}, num_stages=4, num_warps=4),
#     ],
#     key=['N', 'K'],
# )
@triton.jit
def triton_pit_mv_kernel(
    inputs, weight_col_ptr, weight_row_idx, weight_val, outputs,
    batch, K, N,
    stride_inputs_0, stride_inputs_1,
    stride_col_ptr,
    stride_row_idx,
    stride_weight_0, stride_weight_1,
    stride_outputs_0, stride_outputs_1,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    block_id_n = tl.program_id(axis=0)
    block_id_k = tl.program_id(axis=1)

    # pid = tl.program_id(axis=0)
    # GROUP_SIZE_K = 8
    # num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_K * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_k = group_id * GROUP_SIZE_K
    # group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    # block_id_k = first_pid_k + (pid % group_size_k)
    # block_id_n = (pid % num_pid_in_group) // group_size_k

    off_n = block_id_n * BLOCK_SIZE_N
    off_k = block_id_k * BLOCK_SIZE_K

    n_start = tl.load(weight_col_ptr + block_id_k)
    n_end = tl.load(weight_col_ptr + block_id_k + 1)

    if off_n >= n_end - n_start:
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = off_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < n_end - n_start
    row_idx = tl.load(weight_row_idx + n_start + offs_n, mask=mask_n)
    # mask = (offs_k // 8) % 4 == 0
    b = tl.load(weight_val + (n_start + offs_n[:, None]) * stride_weight_0 + offs_k[None, :])  # [BN, BK]

    # for i in range(batch):
    #     a = tl.load(inputs + i * stride_inputs_0 + off_k + offs_k)  # [BK]
    #     c = tl.sum(a[None, :] * b, 1)  # [BN]
    #     tl.atomic_add(outputs + i * stride_outputs_0 + row_idx, c, mask=mask_n)

    a = tl.load(inputs + off_k + offs_k)  # [BK]
    c = tl.sum(a[None, :] * b, 1)  # [BN]
    tl.atomic_add(outputs + row_idx, c, mask=mask_n)


@triton.jit
def triton_pit_mv_kernel_2(
    inputs, weight_col_ptr, weight_row_idx, weight_val, outputs,
    batch, K, N,
    stride_inputs_0, stride_inputs_1,
    stride_col_ptr,
    stride_row_idx,
    stride_weight_0, stride_weight_1,
    stride_outputs_0, stride_outputs_1,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    block_id_k = tl.program_id(axis=0)
    off_k = block_id_k * BLOCK_SIZE_K
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(inputs + off_k + offs_k)  # [BK]

    n_start = tl.load(weight_col_ptr + block_id_k)
    n_end = tl.load(weight_col_ptr + block_id_k + 1)

    rows = tl.cdiv(n_end - n_start, BLOCK_SIZE_N)
    for block_id_n in range(rows):

        off_n = block_id_n * BLOCK_SIZE_N

        offs_n = off_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < n_end - n_start
        row_idx = tl.load(weight_row_idx + n_start + offs_n, mask=mask_n)
        b = tl.load(weight_val + (n_start + offs_n[:, None]) * stride_weight_0 + offs_k[None, :])  # [BN, BK]

        # for i in range(batch):
        #     a = tl.load(inputs + i * stride_inputs_0 + off_k + offs_k)  # [BK]
        #     c = tl.sum(a[None, :] * b, 1)  # [BN]
        #     tl.atomic_add(outputs + i * stride_outputs_0 + row_idx, c, mask=mask_n)

        c = tl.sum(a[None, :] * b, 1)  # [BN]
        tl.atomic_add(outputs + row_idx, c, mask=mask_n)

def triton_pit_mv(
    inputs: torch.Tensor,  # [batch, K]
    weight_col_ptr: torch.Tensor,  # [K / pit_block_size]
    weight_row_idx: torch.Tensor,  # [block_nnz]
    weight_val: torch.Tensor,  # [block_nnz, pit_block_size]
    max_rows: int,
    outputs: torch.Tensor,  # [batch, N]
):
    batch, K = inputs.shape
    batch, N = outputs.shape

    grid = lambda META: (
        triton.cdiv(max_rows, META['BLOCK_SIZE_N']),
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        # triton.cdiv(max_rows, META['BLOCK_SIZE_N']) * triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    triton_pit_mv_kernel[grid](
        inputs, weight_col_ptr, weight_row_idx, weight_val, outputs,
        batch, K, max_rows,
        inputs.stride(0), inputs.stride(1),
        weight_col_ptr.stride(0),
        weight_row_idx.stride(0),
        weight_val.stride(0), weight_val.stride(1),
        outputs.stride(0), outputs.stride(1),
        BLOCK_SIZE_N=128, BLOCK_SIZE_K=pit_block_size, num_stages=3, num_warps=8
    )
    return outputs


torch.manual_seed(2023)

activations = torch.randn((batch_size, in_features), dtype=torch.float32, device='cpu')

quant_weight = torch.randint(
    -0x80000000, 0x80000000,
    (in_features // 32 * bits, out_features),
    dtype=torch.int32, device='cpu',
)
quant_lut = torch.randn((out_features, 2 ** bits), dtype=torch.float32, device='cpu')

sparse_weight = torch.randn((sparse_nnz, ), dtype=torch.float32).numpy()
sparse_rows = torch.randint(0, in_features, (sparse_nnz, ), dtype=torch.int32).numpy()
sparse_cols = torch.randint(0, out_features, (sparse_nnz, ), dtype=torch.int32).numpy()
sparse_arr = csc_array((sparse_weight, (sparse_rows, sparse_cols)), shape=(in_features, out_features))

hybrid_sparse_weight = torch.tensor(sparse_arr.toarray(), dtype=torch.float32, device='cpu')

sparse_weight = torch.tensor(sparse_arr.data, dtype=torch.float32, device='cpu')
sparse_rows = torch.tensor(sparse_arr.indptr[:in_features + 1], dtype=torch.int32, device='cpu')
sparse_cols = torch.tensor(sparse_arr.indices, dtype=torch.int32, device='cpu')

full_cols = torch.randn((in_features, top_x), dtype=torch.float32, device='cpu')
full_col_indices = torch.randint(0, out_features, (top_x, ), dtype=torch.int32, device='cpu')

for i, col in enumerate(full_col_indices):
    hybrid_sparse_weight[:, col] += full_cols[:, i]

hybrid_sparse_weight = hybrid_sparse_weight.reshape((in_features // pit_block_size, pit_block_size, out_features))

mask = hybrid_sparse_weight != 0
print(f'      Sparsity = {(mask.sum() / mask.numel()):.6f}')
block_mask = mask.any(dim=1)
print(f'Block Sparsity = {(block_mask.sum() / block_mask.numel()):.6f}')

bcsc_col_ptr = [0]
bcsc_row_idx = []
bcsc_val = []
max_rows = 0
for col in range(in_features // pit_block_size):
    rows = torch.arange(out_features, device='cpu')[block_mask[col, :]]
    # rows = rows[torch.randperm(rows.numel())]
    bcsc_row_idx.append(rows)
    for row in rows:
        bcsc_val.append(hybrid_sparse_weight[col, :, row])
    bcsc_col_ptr.append(bcsc_col_ptr[-1] + rows.numel())
    max_rows = max(max_rows, rows.numel())
bcsc_col_ptr = torch.tensor(bcsc_col_ptr, device='cpu')
bcsc_row_idx = torch.concat(bcsc_row_idx)
bcsc_val = torch.stack(bcsc_val)
print(f'     Block NNZ = {bcsc_row_idx.numel()}')
print(f'      Max Rows = {max_rows}')


# func_name = 'vecquant4matmul'
# func_name += '_spmv_hybrid'
# func_name += '_nuq_perchannel'
# if batch_size > 1:
#     func_name += '_batched'
# func = getattr(quant_cuda, func_name)

activations = activations.cuda()
quant_weight = quant_weight.cuda()
quant_lut = quant_lut.cuda()
sparse_rows = sparse_rows.cuda()
sparse_cols = sparse_cols.cuda()
sparse_weight = sparse_weight.cuda()
full_cols = full_cols.cuda()
full_col_indices = full_col_indices.cuda()
bcsc_col_ptr = bcsc_col_ptr.cuda()
bcsc_row_idx = bcsc_row_idx.cuda()
bcsc_val = bcsc_val.cuda()


def get_sqllm_func(hybrid: bool):
    func_name = 'vecquant4matmul'
    if hybrid:
        func_name += '_spmv_hybrid'
    func_name += '_nuq_perchannel'
    if batch_size > 1:
        func_name += '_batched'
    return getattr(quant_cuda, func_name)


q_func = get_sqllm_func(False)
h_func = get_sqllm_func(True)


def sqllm_quant():
    outputs = torch.zeros((batch_size, out_features), dtype=torch.float32, device='cuda')
    q_func(
        activations,
        quant_weight,
        outputs,
        quant_lut,
    )
    return outputs


def sqllm_hybrid():
    outputs = torch.zeros((batch_size, out_features), dtype=torch.float32, device='cuda')
    h_func(
        sparse_rows,
        sparse_cols,
        sparse_weight,
        activations,
        full_cols,
        full_col_indices,
        outputs,
        out_features,
        quant_weight,
        quant_lut,
    )
    return outputs


def triton_hybrid():
    outputs = sqllm_quant()
    triton_pit_mv(
        activations,
        bcsc_col_ptr,
        bcsc_row_idx,
        bcsc_val,
        max_rows,
        outputs,
    )
    return outputs


ref_hybrid = sqllm_hybrid()
triton_hybrid()
out_hybrid = triton_hybrid()

# torch.save(outputs, f'ref-{batch_size}.pt')
# ref = torch.load(f'ref-{batch_size}.pt')
torch.testing.assert_close(ref_hybrid, out_hybrid, atol=1e-4, rtol=1e-4)

print(f'[batch = {batch_size}]  SQLLM-Quant  latency: {profile(sqllm_quant):.6f} ms')
print(f'[batch = {batch_size}]  SQLLM-Hybrid latency: {profile(sqllm_hybrid):.6f} ms')
print(f'[batch = {batch_size}] Triton-Hybrid latency: {profile(triton_hybrid):.6f} ms')
