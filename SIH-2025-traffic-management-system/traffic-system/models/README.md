# Model checkpoints

`shared.npz` is a small (~24KB), lightly-trained starter checkpoint (60
episodes, NumPy backend, weight-shared across all 9 intersections in
`city_map/sample_city_graph.json`) committed to the repo so the stack has a
working, trained brain immediately after `git clone && docker compose up`,
without requiring a training run first. It is enough to demonstrate the
system behaving sensibly, not a production-quality policy.

`brain_service.py` looks for `models/{intersection_id}.npz` first, falling
back to `models/shared.npz`, falling back to an untrained (random-ish, pure
exploration) agent if neither exists.

## Training your own

```bash
# Quick (numpy backend, no torch needed, ~1-2 minutes)
python scripts/train_dqn.py --episodes 150 --backend numpy

# Full training run (torch backend, GPU if available, much better policy)
pip install torch
python scripts/train_dqn.py --episodes 2000 --backend torch --steps-per-episode 360

# Independent (non-shared) agents, if you have the compute budget to
# specialize per intersection instead of one shared policy:
python scripts/train_dqn.py --episodes 2000 --backend torch --no-share-weights
```

Checkpoints land in `models/` (or `--output-dir`), plus a
`training_history.json` with per-episode reward/loss for plotting.
Per-intersection and non-`shared.npz` checkpoints are gitignored (see
`.gitignore`) -- treat `models/` as a local/deployment artifact directory,
not something you hand-edit in version control. For a real production
deployment, push checkpoints to object storage (S3/GCS) or a model
registry and have the brain container pull them at startup instead of a
bind-mounted volume.
