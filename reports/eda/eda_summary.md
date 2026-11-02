# Charging data exploration

This report summarizes when participants charged, how long vehicles stayed connected, and how much energy they used.

The processed dataset contains 59,520 sessions from 267 participants.

## Participant activity

| Measure | Mean | Median | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| Observation period (days) | 374.88 | 379.00 | 141.95 | 22 | 624 |
| Sessions per participant | 222.92 | 186.00 | 161.54 | 14 | 1,013 |
| Sessions per week | 4.20 | 3.80 | 2.44 | 1.00 | 13.72 |
| Mean connection time per participant (hours) | 13.04 | 11.71 | 8.22 | 1.64 | 85.35 |

The average participant was observed for about 375 days and recorded 4.2 sessions per week.

## Arrival time, connection time, and energy use

The most common arrival hour was 18:00 in UK local time.

| Session measure | Mean | Median | 75th percentile | 95th percentile |
|---|---:|---:|---:|---:|
| Connection time (hours) | 12.14 | 11.45 | 14.92 | 25.77 |
| Energy use (kWh) | 10.44 | 7.16 | 13.05 | 29.48 |

Connection time measures how long a vehicle stays plugged in. It differs from active charging time.

## Historical charging flexibility ratio

We calculate the ratio of active charging time to total connection time. We classify sessions using local UK dates.

| Session type | Total sessions | Valid ratios | Ratios in 0–1 histogram | Mean ratio | Median ratio |
|---|---:|---:|---:|---:|---:|
| Same-day | 21,700 | 15,947 | 15,947 | 0.689 | 0.780 |
| Overnight | 37,820 | 29,349 | 29,348 | 0.188 | 0.143 |

The mean and median use all valid ratios. The histogram includes only values from 0 to 1. One valid overnight ratio is above 1, which explains the difference between the valid and histogram counts.

The lower overnight ratio shows that active charging took up a smaller share of the total connection period in these records. This is a description of past sessions; it does not confirm that the remaining time or energy can be shifted.

## Figures

- [Participant availability](data_availability.png)
- [Connection time by arrival hour](duration_by_start_hour.png)
- [Energy use by arrival hour](energy_by_start_hour.png)
- [Charging flexibility distributions](charging_flexibility.png)
