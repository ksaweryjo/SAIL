# Text free speech to image Training with SpokenCOCO Dataset

## Data

Two things are needed:

1. **SpokenCOCO**: spoken captions (wav files) and JSON metadata (`SpokenCOCO.tar.gz` from the authors, MIT).
2. **MS-COCO 2014 images** (`train2014`, `val2014`): downloaded separately, not included in SpokenCOCO.

## TODO

- [x] Verify sample rate and path existence on a subset of files
- [x] Choose `max_sec` from the audio length distribution
- [x] Match image normalization to the chosen image encoder
- [x] Check input requirements of the chosen audio encoder
- [ ] Config wandb
- [ ] Implement Visual Encoder (idk)
- [x] Implement Audio Encoder (HuBERT)
- [ ] Training with wandb
- [ ] Experiments
- [ ] Visalisation
