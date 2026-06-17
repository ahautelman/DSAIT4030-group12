import torch
import torch_dct as dct


def split_into_blocks(x, block_size=8):
    """
    Split image into non-overlapping blocks.

    (B, C, H, W) -> (B, C, N_blocks, block_size, block_size)

    """
    B, C, H, W = x.shape
    assert H % block_size == 0 and W % block_size == 0, \
        f"H and W must be divisible by block_size={block_size}"

    x = x.unfold(2, block_size, block_size)   # (B, C, H/b, W, b)
    x = x.unfold(3, block_size, block_size)   # (B, C, H/b, W/b, b, b)
    x = x.contiguous().view(B, C, -1, block_size, block_size)  # (B, C, N, b, b)

    return x


def combine_blocks(blocks, height, width, block_size=8):
    """
    Reassemble blocks back into full image.

    (B, C, N_blocks, block_size, block_size) -> (B, C, H, W)

    """
    B, C, N, b, _ = blocks.shape
    blocks_per_row = width // block_size
    blocks_per_col = height // block_size

    x = blocks.view(B, C, blocks_per_col, blocks_per_row, b, b)
    x = x.permute(0, 1, 2, 4, 3, 5)          # (B, C, H/b, b, W/b, b)
    x = x.contiguous().view(B, C, height, width)

    return x


def dct_2d(x):
    """
    Apply 2D DCT to each block.
    """
    x = dct.dct(x, norm="ortho")                        # DCT along last dim
    x = dct.dct(x.transpose(-2, -1), norm="ortho")      # DCT along second-last dim
    return x.transpose(-2, -1)


def idct_2d(x):
    """
    Apply 2D inverse DCT to each block.

    """
    x = dct.idct(x, norm="ortho")                       # IDCT along last dim
    x = dct.idct(x.transpose(-2, -1), norm="ortho")     # IDCT along second-last dim
    return x.transpose(-2, -1)

def create_triangular_mask(block_size=8, n=8, device=None):
    """
    Creates a triangular high-frequency mask in DCT space.
    
    Based on JPEG zigzag ordering — low frequencies are in top-left,
    high frequencies are in bottom-right.
    
    True = zero out (high frequency)
    False = keep (low frequency)
    """
    max_sum = 2 * (block_size - 1)
    thresh = max_sum - (n - 1)

    u = torch.arange(block_size, device=device).view(block_size, 1)
    v = torch.arange(block_size, device=device).view(1, block_size)
    
    return (u + v) >= thresh  

def apply_dsm_mask(x, z, n, block_size=8):
    if n==0:
        return x, z
    
    B, C, H, W = x.shape
    _, _, h, w = z.shape

    x_blocks = split_into_blocks(x, block_size)
    z_blocks = split_into_blocks(z, block_size)

    x_dct = dct_2d(x_blocks)
    z_dct = dct_2d(z_blocks)

    mask = create_triangular_mask(block_size, n, device=x.device)

    x_dct[..., mask] = 0.0
    z_dct[..., mask] = 0.0

    x_M = combine_blocks(idct_2d(x_dct), H, W, block_size)
    z_M = combine_blocks(idct_2d(z_dct), h, w, block_size)

    return x_M, z_M

if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)

    blocks = split_into_blocks(x, block_size=8)
    print("blocks shape:", blocks.shape)       # (2, 3, 1024, 8, 8)

    restored = combine_blocks(blocks, 256, 256, block_size=8)
    print("restored shape:", restored.shape)   # (2, 3, 256, 256)
    print("reconstruction error:", (x - restored).abs().max().item()) 

    blocks_dct = dct_2d(blocks)
    blocks_restored = idct_2d(blocks_dct)
    print("DCT reconstruction error:", (blocks - blocks_restored).abs().max().item()) 

