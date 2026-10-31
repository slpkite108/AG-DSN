# AG-DSN temporal implementation

This revision computes temporal anticipation before current-frame classification and gives recent and older history independent capacities. It is an equation-aligned reference revision, not a recovered implementation of the original experiments. Synthetic checks do not establish the published results or checkpoint provenance. The previous architecture remains in Git history.

`AGDSN` accepts pooled frame descriptors of shape `[B, T, D]`. The optional frozen backbone retains its pooled descriptor output. This revision does not implement spatial patch-token aggregation. The existing Transformer encoder layer configuration, optimizer, and focal and Dice losses are retained.

The current-time computation maps to these methods in `model.py`.

| Paper operation | Implementation |
| --- | --- |
| `Z = E_obs(X)` | `_encode_observations` encodes each frame independently. |
| `S = concat(previous H Z)` | `TemporalState.frame_memory` holds pre-injection representations. |
| `Y = E_tcs(P(S), Z)` | `_present` decodes ordered `F` prediction queries over `concat(P(S), Z)`. |
| `C = E_ctx(L, Y)` | `_context` uses older history as queries and `Y` as memory. |
| `A = D_tcs(Y, C)` | `_anticipate` preserves prediction order. |
| `Zhat = D_obs(Z, A)` | `_forward_current` injects the same call's attached `A` before classification. |

Current logits have shape `[B, 1, K]`. Future logits have shape `[B, F, K]` for offsets `0` through `F-1`, including the current frame. Current loss can backpropagate through present injection, context, and anticipation even without future loss. Only returned history tensors detach.

The independent defaults are `frame_memory_len=32`, `clip_memory_len=64`, and `prediction_len=32`. The recent FIFO appends pre-injection `Z`. Its oldest eviction enters the bounded older FIFO. The manuscript does not fix the older capacity or exact update rule, so this FIFO rule is this revision's implementation convention. Initial state contains zeros. Supplied short histories pad on the left and long histories retain their newest entries. A zero capacity stays `[B, 0, D]` in public state. Empty context attention uses a private zero query.

For streaming, keep one state per video and pass each current frame once.

```python
from model import AGDSN

model = AGDSN(use_backbone=False).eval()
state = None
for current in video_features.split(1, dim=1):
	output = model(current, history_state=state)
	state = output['state']
```

Reset `state = None` for each new video. `model.initial_state(batch_size)` provides an explicit zero state. The model holds no per-video state internally. `forward` accepts one current frame and rejects the old full-sequence call.

`forward_window` accepts exactly `H+1` observable descriptors. Its first `H` frames become recent history and its last frame is current. Optional `long_history` contains older pre-injection `Z` tensors, not raw descriptors. It defaults to zeros for standalone sampled windows. Streaming and window evaluation agree when both receive the same recent observations and older history.

The default training boundary uses 64 labelled frames. Features at indices `0:32` are history and index `32` is current. Only `features[:, :33]` reach `forward_window`; future visual frames never enter the model. Both epoch functions apply this slice and retain the explicit target arguments.

```python
from train import compute_training_loss, slice_subclip_targets

current_targets, future_targets = slice_subclip_targets(labels64)
output = model.forward_window(features64[:, :33])
losses = compute_training_loss(output, current_targets, future_targets)
losses['total'].backward()
```

`slice_subclip_targets` returns `labels[:, 32:33]` and `labels[:, 32:64]` by default. It also accepts independent history and prediction lengths. There is no reversal of prediction outputs or labels between window and streaming calls.

`clip_context_len`, `frame_context`, and `echo_memory` belong to the previous architecture and are removed. Use `frame_memory_len` for recent `S` and `clip_memory_len` for older `L`. The architecture and state API have changed, so old checkpoints are incompatible. Keep the default strict `load_state_dict` behavior; loading old weights with `strict=False` does not reproduce the original model or its reported results.

Run the CPU synthetic checks without a dataset, GPU, or pretrained download.

```text
python -m unittest discover -s tests -v
python test.py
```

The unit tests check memory independence, FIFO updates, same-step current-loss gradients, detached state, reset isolation, current-inclusive target order, streaming and window consistency, and exclusion of future visual slots. `test.py` checks the default model's forward and backward passes. These are software checks, not training or evaluation experiments.
