# Nearest training neighbour of each test compound (ECFP4, 2048 bit)

* test compounds: 513; nearest-neighbour Tanimoto median 0.52
* test CI width: median 0.49, share > 1.5: 4.7%
* random train–test pair: mean |Δ pEC50| 1.21

| neighbour Tanimoto | n | mean \|Δ pEC50\| (1-NN error) | Spearman(1-NN, true) | same, both CI < 1 (n / ρ) |
|---|---|---|---|---|
| all | 513 | 1.23 | 0.05 | 428 / 0.09 |
| 0.3–0.4 | 9 | 0.86 | 0.64 | 5 / 0.05 |
| 0.4–0.5 | 149 | 1.29 | 0.03 | 128 / 0.08 |
| 0.5–0.6 | 264 | 1.41 | -0.01 | 208 / 0.03 |
| 0.6–0.7 | 78 | 0.71 | 0.32 | 74 / 0.41 |
| 0.7–1.0 | 13 | 0.40 | 0.05 | 13 / 0.05 |

## What Tanimoto 0.4–0.6 is made of

* pairs: 413; bits on in the test compound (median) 46, shared 32, differing 30
* shared bits that are set in ≥ 20% of train (generic pieces): 39% on average
* one methyl on the benzene (3b pair): 10 of 29 bits change, Tanimoto 0.71

## Are the pEC50-correlated bits stand-ins for logP?

* logP vs pEC50: Spearman 0.45 (train)
* top 20 bits by |Spearman with pEC50| (set in 2–98% of train): mean |ρ| 0.18
* same sign with logP: 85%; mean |ρ with logP| 0.13
* vs pEC50 after removing a linear logP trend: mean |ρ| 0.16
