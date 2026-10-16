# A Deep Learning Scheme of Electromagnetic Scattering From Scatterers With Incomplete Profiles

Research code accompanying the paper by Ji-Yuan Wang, Xin-Yue Lou, Liang Zhang, Yun-Chuan Wang, and Xiao-Min Pan, published in *IEEE Transactions on Antennas and Propagation*, vol. 74, no. 10, pp. 9125-9135, 2026.

**Paper:** [https://doi.org/10.1109/TAP.2025.3635042](https://doi.org/10.1109/TAP.2025.3635042)

This release provides data generation, training, and evaluation for the four input-output pairs in the MNIST validation experiment.

## Abstract

A deep learning scheme is proposed to solve the electromagnetic (EM) scattering problems where the profile of the dielectric scatterer of interest is incomplete. As compensation for the incompletion of the profile, a limited amount of scattering/prescribed data is provided, which, in principle, contains sufficient information associated with the missing part of the profile. The existing solvers can hardly solve this type of problem. On one hand, the well-developed forward solvers have no mechanism to accept the scattering data to recover the unknown part of the profile. On the other hand, the existing solvers for inverse problems cannot retrieve the complete profile with an acceptable accuracy from the limited amount of scattering data, even when the available part of the profile can be fed into the solvers. To overcome the difficulty, the EM forward scattering from an incompletely known dielectric scatterer is derived in this work. A deep learning scheme is then developed where the forward and inverse scattering problems can be solved simultaneously. Numerical experiments are conducted to demonstrate the performance of the proposed DL-based scheme in terms of 2-D EM scattering problems.

## Repository structure

```text
.
|-- forward_data/
|   |-- label.py                     # Dataset download and forward simulation
|   `-- Forward_computation.py       # Domain, incident fields, Green's Functions, and VIE solver
|-- _config.py                       # Numerical precision
|-- _network_module.py               # Models and dataset classes
|-- main_train_2d_pair_I.py
|-- main_train_2d_pair_II.py
|-- main_train_2d_pair_III.py
|-- main_train_2d_pair_iv.py
|-- main_test_2d_pair_I.py
|-- main_test_2d_pair_II.py
|-- main_test_2d_pair_III.py
|-- main_test_2d_pair_iv.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation


```bash
python -m pip install -r requirements.txt
```

## Generate the datasets

```bash
cd forward_data
python label.py
cd ..
```

## Train the models

Run the training scripts from the repository root:

```bash
python main_train_2d_pair_I.py
python main_train_2d_pair_II.py
python main_train_2d_pair_III.py
python main_train_2d_pair_iv.py
```

Before training Pair IV, run Pair I or Pair III to generate the required training and validation scattered-field data.

## Evaluate

Training saves checkpoints in the repository root, and all four test scripts load them directly from that directory:

| Pair | Checkpoint location |
| --- | --- |
| I | `unet_rv_num15000_64rx_pair1.pkl` |
| II | `unet_rv_num15000_64rx_pair2.pkl` |
| III | `unet_rv_num15000_64rx_pair3.pkl` |
| IV | `unet_rv_num15000_64rx_pair4.pkl` |

Run evaluation from the repository root:

```bash
python main_test_2d_pair_I.py
python main_test_2d_pair_II.py
python main_test_2d_pair_III.py
python main_test_2d_pair_iv.py
```

Pair IV reads `forward_data/Es_test_data_1_1_num1000_64rx.npy`, which Pair I or Pair III evaluation generates. Evaluate one of those pairs before Pair IV.


## Errata

In the manuscript, the REs in Section IV were incorrectly described as contrast ($\chi$) errors. They were actually evaluated using relative permittivity ($\varepsilon_r = \chi + 1$). This clarification does not alter the reported conclusions. The authors regret the inaccurate description.

## Citation

```bibtex
@article{wang2026incomplete,
  author = {Wang, Ji-Yuan and Lou, Xin-Yue and Zhang, Liang and Wang, Yun-Chuan and Pan, Xiao-Min},
  title = {A Deep Learning Scheme of Electromagnetic Scattering From Scatterers With Incomplete Profiles},
  journal = {IEEE Transactions on Antennas and Propagation},
  year = {2026},
  volume = {74},
  number = {10},
  pages = {9125--9135},
  doi = {10.1109/TAP.2025.3635042}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions, please contact Ji-Yuan Wang at [jiyuanwang14@gmail.com](mailto:jiyuanwang14@gmail.com).
