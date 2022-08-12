Continuous Doubly Constrained Batch Reinforcement Learning
=============================================
Quantile regression is a fundamental problem in statistical learning motivated by the need to quantify uncertainty in predictions, or to model a diverse population without being overly reductive. For instance, epidemiological forecasts, cost estimates, and revenue predictions all benefit from being able to quantify the range of possible values accurately. As such, many models have been developed for this problem over many years of research in econometrics, statistics, and machine learning.

Rather than proposing yet another (new) algorithm for quantile regression we adopt a meta viewpoint: we investigate methods for aggregating any number of conditional quantile models, in order to improve accuracy and robustness. We consider weighted ensembles where weights may vary over not only individual
models, but also over quantile levels, and feature values. All of the models we consider in this paper can be fit using modern deep learning toolkits, and hence are widely accessible (from an implementation point of view) and scalable. 

To improve the accuracy of the predicted quantiles (or equivalently, prediction intervals), we develop tools for ensuring that quantiles remain monotonically ordered, and apply conformal calibration methods. These can be used without any modification of the original library of base models. We also review some basic theory surrounding quantile aggregation and related scoring rules, and contribute a few new results to this literature (for example, the fact that post sorting or post isotonic regression can only improve the weighted interval score). Finally, we provide an extensive suite of empirical comparisons across 34 data sets from two different benchmark repositories. 

Thisrepository provides the implementation of [Flexible Model Aggregation for Quantile Regression](https://arxiv.org/abs/2103.00083). If you use this code please cite the paper using the following bibtex:

```
@article{fakoor2022quantile,
  title={Flexible Model Aggregation for Quantile Regression},
  author={Rasool Fakoor, Taesup Kim, Jonas Mueller, Alexander J. Smola, Ryan J. Tibshirani},
  journal={arXiv preprint arXiv:2103.00083},
  year={2021},
}


```
## Getting Started
```
Run the following commands in the specified order: 

1) python -u nested_base_quantile_models.py --DATA_PATH ~/mydata/ --data_loc ~/rawdata/ --task-id yacht -seed 1

2) python -u merge_nested_dara.py --DATA_PATH ~/mydata/ --task-id yacht -seed 1

3) python -u nested_aggr_quantile_models.py --DATA_PATH ~/mydata/ --task-id yacht -seed 1 --RESULT_PATH ~/myresult/

```
The code works on both GPU and CPU machines.

In order to run this code, you will need to install pytorch, lightgbm, numpy, openml, scikit_learn,scipy, autogluon, statsmodels, etc.

## License
This project is licensed under the Apache-2.0 License.

# Contact

Please open an issue on [issues tracker](https://github.com/amazon-research/quantile-aggregation) to report problems or to ask questions or send an email to me, [Rasool Fakoor](https://github.com/rasoolfa).
