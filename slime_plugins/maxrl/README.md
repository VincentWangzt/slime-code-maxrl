# Rollout MaxRL regression hooks

Use the first-class estimator with the boxed-number reward and metric hooks:

```text
--advantage-estimator maxrl
--rollout-shuffle
--n-samples-per-prompt <N>
--global-batch-size <rollout-batch-size-times-N>
--custom-rm-path slime_plugins.maxrl.regression.boxed_gaussian_reward
--reward-key maxrl_log_likelihood
--eval-reward-key maxrl_score
--custom-rollout-log-function-path slime_plugins.maxrl.regression.log_train_regression_metrics
--custom-eval-rollout-log-function-path slime_plugins.maxrl.regression.log_eval_regression_metrics
```

Each CDSS row must provide a finite numeric label and a non-empty
`metadata.language`. Evaluation configuration must name the dataset `CDSS`.
The reward extracts the last complete `\boxed{NUMBER}` and accepts only signed
integers, decimals with digits on both sides of the point, or `e`/`E`
scientific notation.

MaxRL defaults to degree `N`, leave-one-out baseline subtraction, log-likelihood
supremum `0`, Gaussian score standard deviation `1`, and 65 evaluation samples
per prompt. The sample counts and estimator settings remain configurable
through their corresponding CLI options.
