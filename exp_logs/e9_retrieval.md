# E9 - Bridge/encoder retrieval probe: results (processed_zuco2)

## Baselines (shared across checkpoints)

| source | top1 | top10 | median rank |
|---|---|---|---|
| raw EEG (E4) | 0.065 | 0.395 | 16 |
| *chance* | 0.015 | 0.149 | 34 |
| random_bridge (untrained) | 0.069 | 0.404 | 15 |
| text encoder (REBEL, upper bound) | 0.639 | 0.930 | 1 |

## Trained checkpoints

| label | val_f1 | bridge top10 | encoder top10 | bridge top1 | encoder top1 |
|---|---|---|---|---|---|
| checkpoints_e5_s0t0aPL | 0.0014908684308609765 | 0.339 | 0.225 | 0.051 | 0.025 |
| checkpoints_e5_s0t0aPW | 0.0 | 0.348 | 0.326 | 0.049 | 0.048 |
| checkpoints_e5_s0t1aPL | 0.0 | 0.166 | 0.202 | 0.021 | 0.026 |
| checkpoints_e5_s0t1aPW | 0.0 | 0.157 | 0.218 | 0.015 | 0.024 |
| checkpoints_e5_s1t0aPL | 0.001429081814933905 | 0.326 | 0.246 | 0.044 | 0.035 |
| checkpoints_e5_s1t0aPW | 0.0 | 0.343 | 0.357 | 0.047 | 0.057 |
| checkpoints_e5_s1t1aPL | 0.0 | 0.182 | 0.226 | 0.018 | 0.029 |
| checkpoints_e5_s1t1aPW | 0.0 | 0.204 | 0.230 | 0.026 | 0.026 |
| checkpoints_e6_f010 | 0.0006449532408900355 | 0.366 | 0.177 | 0.062 | 0.019 |
| checkpoints_e6_f025 | 0.0 | 0.334 | 0.222 | 0.052 | 0.029 |
| checkpoints_e6_f050 | 0.0 | 0.320 | 0.230 | 0.059 | 0.028 |
| checkpoints_e6_f100 | 0.001429081814933905 | 0.326 | 0.246 | 0.044 | 0.035 |
| checkpoints_e8_c03 | 0.002352941176470588 | 0.311 | 0.186 | 0.055 | 0.024 |
| checkpoints_e8_c06 | 0.0 | 0.323 | 0.210 | 0.052 | 0.027 |
| checkpoints_e8_c09 | 0.001248829222603809 | 0.307 | 0.179 | 0.051 | 0.024 |
| checkpoints_e8_k0 | 0.0014908684308609765 | 0.339 | 0.225 | 0.051 | 0.025 |
| checkpoints_e8_x03 | 0.000570287995437696 | 0.312 | 0.189 | 0.048 | 0.019 |
| checkpoints_e8_x06 | 0.0012861736334405143 | 0.323 | 0.221 | 0.051 | 0.033 |
| checkpoints_e8_x09 | 0.0 | 0.310 | 0.168 | 0.053 | 0.017 |
