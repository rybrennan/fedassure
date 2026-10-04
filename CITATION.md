# Citation and provenance

## This harness

Anchor 163 LLC, *telltale: a measurement harness for federated aggregation integrity*, 2026.
https://github.com/rybrennan/telltale. MIT licence (see `LICENSE`).

## Methods this builds on

- McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B. A. y. (2017). Communication-efficient
  learning of deep networks from decentralized data. *AISTATS*. Used for FedAvg, implemented directly in
  `telltale/fedavg.py`.
- Hsu, T.-M. H., Qi, H., Brown, M. (2019). Measuring the effects of non-identical data distribution
  for federated visual classification. arXiv:1909.06335. Used for Dirichlet non-IID partitioning.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1–2). Used for CUSUM.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Trans. Inf. Theory* 37(1).
  Used for Jensen–Shannon divergence.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
  *JASA* 22(158). Used for the interval on detection probability.

## Datasets

Neither dataset is redistributed here; `data/` is ignored by git and populated by the loaders.

- **Fashion-MNIST.** Xiao, H., Rasul, K., Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset
  for benchmarking machine learning algorithms. arXiv:1708.07747. MIT licence. Downloaded by
  torchvision on first use. 60,000 training / 10,000 test images, 10 classes.
- **DeepShip.** Irfan, M., Jiangbin, Z., Ali, S., Iqbal, M., Masood, Z., Hamid, U. (2021). DeepShip:
  an underwater acoustic benchmark dataset and a separable convolution based autoencoder for
  classification. *Expert Systems with Applications* 183. Public portion cloned from
  https://github.com/irfankamboh/DeepShip (63 recordings, four vessel classes, about ninety
  minutes); the repository states no explicit licence and is used here for research measurement
  only. The full dataset is available from the authors by request.

## Software

PyTorch (BSD-3), torchvision (BSD-3), NumPy (BSD-3), SciPy (BSD-3), Matplotlib (PSF-based),
pytest (MIT). Exact versions in `requirements-lock.txt`. No Flower, no Ray, no other framework:
the reasons are in the README.
