# Communication Complexity of Universal Features

## About
How many bits must a task-agnostic representation keep so that every task in a family can still be answered accurately?
Surprisingly, the answer is `Theta(d* log(1/eps))` bits for a family with dual VC dimension `d*`.

These experiments:

1. Run the maximum-error variant of Algorithm 1 on the threshold product `F_{d*,N}`.
2. Train top-k SAEs for reconstruction on powerset, parity, and coordinate families, select the smallest representation size that meets each error target, and compare against `B_0` and `B_sh`.
3. Train the SAEs on noiseless coordinates of `[1024]^r` and measure the worst threshold error against the bit budget per dimension.

## Setup

Python 3.13 or newer is required.

```sh
bash scripts/setup.sh
```

## How to run

```sh
scripts/run.sh                                           # run all three experiments
scripts/run.sh configs/analysis.json                     # analysis of the saved outputs
```

## Results

<details>
<summary><strong>Greedy construction on the threshold product</strong></summary>

![Representation size of the greedy construction against d* log(1/eps)](figures/greedy_scaling.png)

Representation size of the maximum-error greedy construction on threshold products, plotted against `d* log(1/eps)`.
The curves follow `d* ceil(log2(1 / (2 eps)))`; the dashed line is `y = x`.

</details>

<details>
<summary><strong>Finite-family capacity saturation</strong></summary>

![Minimum bit budget passing validation at each error target](figures/capacity_target.png)

Minimum tested bit budget passing validation at each error target, shown as the median over seeds with a configuration.
Dotted lines indicate the lossless reference `B_0`.
Open triangles indicate instances for which no configuration passes validation.

![Bit budgets against the shattered-set bound](figures/bit_budget_shattered.png)

Bit budgets against the shattered-set bound `B_sh(eps)`, one point per selected seed with offsets for visibility.
Filled dots were certified on test; the dashed line is `B_bits = B_sh(eps)`.

</details>

<details>
<summary><strong>Accuracy-bit tradeoffs on threshold products</strong></summary>

![Worst-case test error against bits per dimension](figures/threshold_product.png)

Worst-case error against the bit budget per dimension for reconstruction-trained SAEs, one dot per random seed (three per dimension) for each `d*` in `{1, 2, 4, 8, 10}`.
× are the exact error `2^(-l - 1)` of equal-width quantization using `l` bits per coordinate.
Prediction error decreases as the representation budget increases.

</details>

## License 
[MIT License](LICENSE)
