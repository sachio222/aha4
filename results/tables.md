## Results, default settings

| Stage scored | What it measures | This repo | Paper |
|---|---|---|---|
| Raw pixels | Floor for comparison | 44.2% | not reported |
| VC(EC) alone | Vision network only, the paper's baseline | 52.2% | 71.6% |
| PR(EC-CA3) | AHA, the paper's headline number | **64.7% ± 1.2** | **86.4%** |
| PC(CA3) | AHA after the Hopfield memory settles | 61.6% ± 0.9 | "no better than" the baseline |
| PM(CA1) | AHA's recalled image | 61.9% ± 0.7 | not reported |

## Closing the gap

| Setting | Matching | VC(EC) | PR(EC-CA3) | PC(CA3) | PM(CA1) |
|---|---|---|---|---|---|
| Default: study-to-test, raw scale | study-to-test | 52.2% | 64.7% ± 1.2 | 61.6% ± 0.9 | 61.9% ± 0.7 |
| Test-to-study matching, the repo's first version | test-to-study | 56.0% | 61.3% ± 0.9 | 62.5% ± 0.8 | 61.5% ± 0.9 |
| Normalised matching, the best configuration found | study-to-test, normalised | 63.0% | 70.6% ± 1.4 | 61.5% ± 0.7 | 62.5% ± 1.1 |
| Normalised matching, test-to-study | test-to-study, normalised | 67.3% | 63.2% ± 1.2 | 62.5% ± 0.8 | 61.7% ± 0.8 |
| VC output normalised before DG and PR | study-to-test | 63.0% | 66.0% ± 1.1 | 61.5% ± 1.4 | 61.3% ± 0.9 |
| VC output normalised before DG and PR, normalised matching | study-to-test, normalised | 63.0% | 70.3% ± 1.5 | 61.3% ± 1.4 | 61.9% ± 1.0 |
| Default, VC pretrained with seeds 0, 1 and 2 | study-to-test | 54.4% ± 1.9 | 67.1% ± 2.0 | 63.0% ± 1.3 | 63.0% ± 1.0 |
| MSE pretraining | study-to-test | 54.0% | 59.3% ± 1.7 | 53.3% ± 1.9 | 54.4% ± 1.4 |
| MSE pretraining, normalised matching | study-to-test, normalised | 59.0% | 63.8% ± 1.2 | 53.2% ± 1.7 | 54.5% ± 1.3 |
| MSE pretraining, seeds 0, 1 and 2 | study-to-test | 54.6% ± 1.7 | 60.2% ± 1.2 | 53.6% ± 1.6 | 54.6% ± 1.9 |
| MSE pretraining, no interest filter | study-to-test | 57.7% | 57.1% ± 1.6 | 23.2% ± 2.6 | 30.0% ± 1.6 |
| MSE pretraining, VC output normalised before DG and PR, normalised matching | study-to-test, normalised | 59.0% | 63.9% ± 1.5 | 53.2% ± 2.1 | 54.3% ± 1.6 |
| Normalised matching, VC pretrained with seeds 0, 1 and 2 | study-to-test, normalised | 63.2% ± 0.1 | 72.0% ± 1.2 | 62.9% ± 1.3 | 63.6% ± 1.0 |

## Ablations

| Setting | Matching | VC(EC) | PR(EC-CA3) | PC(CA3) | PM(CA1) |
|---|---|---|---|---|---|
| No interest filter | study-to-test | 53.8% | 52.5% ± 1.0 | 18.3% ± 1.6 | 26.1% ± 1.7 |
| No centring of drawings | study-to-test | 29.8% | 41.3% ± 1.5 | 38.1% ± 1.1 | 37.6% ± 0.9 |
| Hebbian learning in CA3 instead of pseudo-inverse | study-to-test | 52.2% | 64.7% ± 1.2 | 5.8% ± 0.5 | 5.8% ± 0.6 |

## Devices

| Setting | Matching | VC(EC) | PR(EC-CA3) | PC(CA3) | PM(CA1) |
|---|---|---|---|---|---|
| CPU, seed 0 | study-to-test | 52.2% | 66.0% | 62.0% | 62.2% |
