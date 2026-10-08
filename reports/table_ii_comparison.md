# Table II Reproduction Comparison

Source paper values are copied from Table II of `paper/paper-riskbench.pdf`.
Reproduced values use the official `ROI_tool.py` on the common 1,632-scene
manifest: 515 interactive, 420 collision, 303 obstacle, and 394
non-interactive scenes. `Delta` means reproduced minus paper, in percentage
points for P/R/FA/F1 and the native PIC units for PIC.

| Method / split | P paper | P ours | dP | R paper | R ours | dR | PIC paper | PIC ours | dPIC | FA paper | FA ours | dFA | All F1 paper | All F1 ours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random / interactive | 9.5 | 13.16 | +3.66 | 50.0 | 15.34 | -34.66 | 16.7 | 10.8 | -5.9 | - | - | - | 15.1 | 14.58 |
| Random / collision | 19.2 | 34.23 | +15.03 | 49.8 | 15.23 | -34.57 | 15.7 | 20.4 | +4.7 | - | - | - | 15.1 | 14.58 |
| Random / obstacle | 7.2 | 12.73 | +5.53 | 50.3 | 12.86 | -37.44 | 11.4 | 19.9 | +8.5 | - | - | - | 15.1 | 14.58 |
| Random / non-interactive | - | - | - | - | - | - | - | - | - | 80.8 | 31.64 | -49.16 | 15.1 | 14.58 |
| Range (10m) / interactive | 43.6 | 46.72 | +3.12 | 69.1 | 65.82 | -3.28 | 7.1 | 3.4 | -3.7 | - | - | - | 53.6 | 56.14 |
| Range (10m) / collision | 69.3 | 77.04 | +7.74 | 62.4 | 52.90 | -9.50 | 3.8 | 3.2 | -0.6 | - | - | - | 53.6 | 56.14 |
| Range (10m) / obstacle | 33.3 | 49.05 | +15.75 | 88.5 | 76.61 | -11.89 | 0.8 | 4.5 | +3.7 | - | - | - | 53.6 | 56.14 |
| Range (10m) / non-interactive | - | - | - | - | - | - | - | - | - | 15.2 | 13.74 | -1.46 | 53.6 | 56.14 |
| Kalman filter / interactive | 37.0 | 27.31 | -9.69 | 54.0 | 8.74 | -45.26 | 15.4 | 9.0 | -6.4 | - | - | - | 46.7 | 27.51 |
| Kalman filter / collision | 59.9 | 75.46 | +15.56 | 58.1 | 31.48 | -26.62 | 4.2 | 6.1 | +1.9 | - | - | - | 46.7 | 27.51 |
| Kalman filter / obstacle | 35.3 | 21.73 | -13.57 | 61.2 | 3.56 | -57.64 | 10.2 | 22.6 | +12.4 | - | - | - | 46.7 | 27.51 |
| Kalman filter / non-interactive | - | - | - | - | - | - | - | - | - | 18.8 | 6.80 | -12.00 | 46.7 | 27.51 |

The paper's test-set count is 1,689 (521 interactive, 375 collision, 322
obstacle, 471 non-interactive), while the released files used here provide a
different common manifest. Therefore the all-scenario F1 values are useful for
scale comparison, but they are not an apples-to-apples statistical rerun.
