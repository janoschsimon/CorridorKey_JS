# MLX Port Debug Log

## Problem
MLX backend produces drastically worse quality than Torch backend.

## Investigation

### Weight Conversion ✅
- Compared all 367 Torch keys vs 365 MLX keys (2 dropped: `num_batches_tracked`)
- Conv weights correctly transposed (O,I,H,W) → (O,H,W,I)
- Refiner stem keys correctly remapped (Sequential indices → named attrs)
- All values match within float32 tolerance

### Backbone (Hiera) ✅
- Patch embed conv output: correlation=1.0, max_diff < 1e-5
- Pos embed interpolation: identical (both bicubic)
- All 4 stage feature maps: correlation=1.0, max_diff < 0.001
- Unroll/reroll: verified matches timm's flat 1D approach
- Attention (MaskUnitAttention): permutation order matches timm exactly
- HieraBlock residual pooling: flat max pool matches timm

### Decoder ❌ — BUG FOUND
- Individual MLP projections (linear_c1..c4): all correlation=1.0
- Upsample to c1 resolution: all match
- **Concatenation order is reversed:**
  - Torch: `cat([c4, c3, c2, c1], dim=1)`
  - MLX: `cat([c1, c2, c3, c4], axis=-1)`
- The `linear_fuse` Conv2d was trained expecting c4|c3|c2|c1 channel order
- MLX feeds c1|c2|c3|c4 → completely wrong feature mixing

### Root Cause
In `corridorkey-mlx/src/corridorkey_mlx/model/decoder.py`, the decoder iterates over features in order `[c1, c2, c3, c4]` and appends projections in that order. The Torch decoder processes them in the same order but concatenates as `[c4, c3, c2, c1]`.

### Fix
Reverse the projected list before concatenation, or iterate in c4→c1 order.
