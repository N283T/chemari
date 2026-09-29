# Reference results

`chemeleon_lightgbm.csv`: LightGBM on CheMeleon fingerprints, next to the notebook's baselines.
`chemeleon_test_predictions.csv`: the CheMeleon model's test predictions; the notebook's Model lab
reads it to list this run on its scoreboard.

- Same train (4,139) / test (513) split, the notebook's LightGBM settings, one run (seed 0).
- CheMeleon ([Burns et al.](https://github.com/JacksonBurns/chemeleon), the Chemprop message
  passing network pretrained to predict Mordred descriptors) is used frozen as a fingerprint:
  each molecule's atom vectors are averaged into one 2048-dimensional vector
  (chemprop 2.3.1, checkpoint `chemeleon_mp.pt` from Zenodo record 15460715).
- The first three rows match the notebook's Model lab scoreboard. The test pEC50 SD is 1.0.
