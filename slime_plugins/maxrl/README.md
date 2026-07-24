# Rollout MaxRL regression hooks

Use the first-class estimator with the boxed-number reward and metric hooks:

```text
--advantage-estimator maxrl
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
The reward reuses Slime's rightmost-box extraction, strips the extracted
content, and converts it with `float`. This accepts finite signed values,
including `.5`, `1.`, and scientific notation. A missing, malformed, or
non-finite rightmost box is an extraction failure; earlier boxes are not used
as fallbacks.

MaxRL consumes already normalized log-likelihoods: every value must be finite
or `-inf` and no value may be materially greater than `0`. The bundled reward
computes
`maxrl_log_likelihood = -0.5 * ((prediction - target) / maxrl_score_std)^2`
and `maxrl_score = exp(maxrl_log_likelihood)`.

MaxRL requires the exact `--custom-rm-path` and `--reward-key` shown above and
per-sample reward mode (do not set `--group-rm`). It defaults to degree `N`,
leave-one-out baseline subtraction, Gaussian score standard deviation `1`,
and one evaluation sample per prompt. For a 65-sample CDSS median evaluation,
set `--n-samples-per-eval-prompt 65` explicitly (globally or in the CDSS eval
dataset configuration). The logging hooks and `--eval-reward-key maxrl_score`
shown above are recommended, but are not validation requirements.

The launcher uses Slime's existing `--message-processor` interface to load
`prompts/code_regression.yaml`, prefix-truncate code to 2,048 model tokens,
and apply the model chat template. If `--rollout-max-prompt-len` is set, the
data source fails when a rendered prompt exceeds that limit. The top-level
`prompts/` directory is cached in the Modal image.
