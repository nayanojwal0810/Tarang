# TARANG decision support summary

We apply the duration and energy models to the 59,520 sessions in the prepared dataset. We then calculate projected unplug times and summarize historical active charging time.

The prediction summary describes model outputs across the prepared dataset. It does not measure model accuracy. The duration and energy evaluation reports provide the test results.

## Prediction outputs

| Target | Total sessions | Valid predictions | Invalid predictions | Mean prediction | Median prediction |
|---|---:|---:|---:|---:|---:|
| Connection duration (hours) | 59,520 | 59,518 | 2 | 10.05 | 11.47 |
| Energy consumption (kWh) | 59,520 | 59,520 | 0 | 10.23 | 7.66 |

TARANG marks a prediction as valid when it is finite and non-negative. It keeps the model's original numeric output. When the duration prediction is invalid, TARANG does not calculate a projected unplug time.

## Historical active charging ratio

We calculate the share of a session spent actively charging:

    Active charging ratio = (ChargingDuration / 60) / connection duration in hours

We classify sessions using local UK dates in Europe/London.

| Session type | Total sessions | Valid ratios | Ratios from 0 to 1 | Mean ratio | Median ratio |
|---|---:|---:|---:|---:|---:|
| Same-day | 21,700 | 15,947 | 15,947 | 0.689 | 0.780 |
| Overnight | 37,820 | 29,349 | 29,348 | 0.188 | 0.143 |

The mean and median use all valid ratios. The histogram count includes only ratios from 0 to 1. One valid overnight ratio is above 1, so it does not appear in the histogram count.

This ratio describes recorded charging activity. It does not show that the remaining connection time or energy can be shifted, and it does not guarantee that a charger or grid can use that time.

## Batch advisory fields

For each input session, TARANG adds:

- predicted_Duration
- predicted_ConsumedkWh
- projected_unplug_time_utc
- duration_prediction_valid
- energy_prediction_valid

## Scope

TARANG provides statistical estimates for review. It does not schedule vehicles, control charger hardware, change charging power, optimise tariffs, or dispatch energy.

## Supporting results

- [Prediction output summary](prediction_summary.csv)
- [Historical charging ratio summary](flexibility_summary.csv)
- [Decision support contract](../../docs/decision_support/decision_support_contract.md)
- [Inference contract](../../docs/inference/inference_contract.md)
