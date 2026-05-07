# PI-CATS: Physics-Informed Contact-Aware Transformer for ARCTIC Hand-Object Dynamics

This package is a research-grade implementation for building, training, evaluating, and visualizing a physics-informed hand-object interaction model on ARCTIC raw sequences.

The target hypothesis is:

> object motion must be explained by hand contact, local hand motion, and contact dynamics, rather than being predicted as an independent trajectory.

## Design rule: one geometry chain only

All MANO decoding, MANO mesh/joint/endpoints extraction, object pose transformation, nearest object point search, and contact label computation are centralized in:

```text
src/pinn_hoi/common/geometry_engine.py
```

Training, evaluation, and visualization must not create alternative MANO-to-endpoint or object-pose-to-point-cloud logic. This prevents the common bug where the training target, evaluation metrics, and visualization use different coordinate or unit conventions.

## What is different from the previous minimal PINNHSSModel?

The previous package used a simple MLP/diagonal state-space style model. This package uses a stronger top-conference-style architecture:

1. endpoint/object tokenization with geometry-aware embeddings;
2. bidirectional hand-object cross-attention for contact graph construction;
3. temporal transformer encoder over interaction tokens;
4. explicit contact/impulse/friction latent heads;
5. staged PINN losses to balance supervised convergence and physics residual convergence;
6. AMP, gradient clipping, cosine scheduling, warmup, JSONL metrics, and checkpointing.

The model is intentionally not tiny, but it is still trainable on sequence windows because object tokens are downsampled, the number of hand endpoint tokens is small, and physics losses are ramped in gradually.

## Expected ARCTIC layout

After following the official ARCTIC download/unzip instructions, use the `arctic_data/data` directory as `--arctic-root`:

```text
<arctic-root>/raw_seqs/s01/*.mano.npy
<arctic-root>/raw_seqs/s01/*.object.npy
<arctic-root>/raw_seqs/s01/*.egocam.dist.npy
<arctic-root>/meta/object_vtemplates/<object>/mesh.obj
<arctic-root>/splits_json/*.json
```

MANO model directory should contain MANO_RIGHT/MANO_LEFT models readable by `smplx.create(..., model_type='mano')`.

## Install

```bash
cd pihss_arctic_topconf
pip install -r requirements.txt
```

## 0. Build processed sequences from ARCTIC raw data

```bash
python scripts/00_build_arctic_picats_dataset.py \
  --arctic-root /public/home/jiaozixun/arctic_hss_pipeline/mini_data/data \
  --mano-root /public/home/jiaozixun/arctic_hss_pipeline/thirty_part/mano_v1_2/models \
  --out-dir ./outputs/picats_arctic/sequences \
  --split-out ./outputs/picats_arctic/splits \
  --num-object-points 2048 \
  --contact-thresh-m 0.015 \
  --device cuda:0
```

For a quick pipeline check:

```bash
python scripts/00_build_arctic_picats_dataset.py \
  --arctic-root /public/home/jiaozixun/arctic_hss_pipeline/mini_data/data \
  --mano-root /public/home/jiaozixun/arctic_hss_pipeline/thirty_part/mano_v1_2/models \
  --out-dir ./outputs/picats_arctic/sequences \
  --split-out ./outputs/picats_arctic/splits \
  --num-object-points 1024 \
  --contact-thresh-m 0.015 \
  --device cuda:0 \
  --max-seqs 3
```

## 1. Train

```bash
python scripts/01_train_picats.py --config configs/arctic_picats_topconf.yaml
```

Small-data overfit sanity check:

```bash
python scripts/01_train_picats.py --config configs/arctic_picats_topconf.yaml --overfit-batches 4 --epochs 80
```

## 2. Evaluate

```bash
python scripts/02_eval_picats.py \
  --config configs/arctic_picats_topconf.yaml \
  --ckpt ./outputs/picats_arctic/exp_topconf/best.pt \
  --split val
```

## 3. Visualize

```bash
python scripts/03_visualize_picats.py \
  --config configs/arctic_picats_topconf.yaml \
  --ckpt ./outputs/picats_arctic/exp_topconf/best.pt \
  --seq ./outputs/picats_arctic/sequences/s01/capsulemachine_use_01.npz \
  --out ./outputs/picats_arctic/vis/capsulemachine_use_01.mp4
```

## Main outputs in each processed `.npz`

```text
hand_vertices          [T,2,778,3]
hand_joints            [T,2,21,3]
hand_endpoints         [T,2,21,3]
obj_pose7              [T,7]        # [articulation, axis-angle rotation, translation_m]
obj_points_canonical   [N,3]
obj_points_world       [T,N,3]
contact_label          [T,2,21]
endpoint_obj_min_dist  [T,2,21]
endpoint_nearest_obj_idx [T,2,21]
```

`hand_endpoints`, `obj_points_world`, and `contact_label` are built once by `UnifiedGeometryEngine`, then reused by all training/evaluation/visualization code.

## Practical training notes

1. Start with object pose supervision and contact BCE only for a few epochs.
2. Gradually ramp contact velocity, impulse dynamics, and friction losses.
3. If `obj_trans_err_m` drops but `contact_motion_cos` stays low, the model is still predicting object motion independently.
4. If `contact_f1` is high but `no_contact_drift_m` is high, the model learned contact masks but not causal object dynamics.
5. If physics losses explode early, increase `physics_warmup_epochs` and lower `impulse_dyn`, `angular_dyn`, and `friction_cone` weights.
