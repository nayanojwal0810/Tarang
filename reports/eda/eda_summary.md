# Charging data exploration

We explore when participants connect their vehicles, how long they stay connected, how much energy they use, and how active charging compares with total connection time.

The prepared dataset contains 59,520 sessions from 267 participants.

## Participant activity

| Measure | Mean | Median |
|---|---:|---:|
| Observation period per participant (days) | 374.88 | 379.00 |
| Sessions per participant | 222.92 | 186.00 |
| Sessions per week | 4.20 | 3.80 |
| Mean connection time per participant (hours) | 13.04 | 11.71 |

We observe each participant for about 375 days on average. Participants record about 4.2 sessions per week.

## Arrival time, connection time, and energy use

The most common arrival hour is 18:00 in local UK time.

| Session measure | Mean | Median | 75th percentile | 95th percentile |
|---|---:|---:|---:|---:|
| Connection time (hours) | 12.14 | 11.45 | 14.92 | 25.77 |
| Energy use (kWh) | 10.44 | 7.16 | 13.05 | 29.48 |

Connection time measures the full period between plug-in and unplug. It includes active charging and any time the vehicle remains connected after charging ends.

## Active charging within the connection period

We calculate the ratio of active charging time to total connection time:

    Active charging ratio = (ChargingDuration / 60) / connection duration in hours

We classify sessions by local UK dates in Europe/London.

| Session type | Total sessions | Valid ratios | Ratios from 0 to 1 | Mean ratio | Median ratio |
|---|---:|---:|---:|---:|---:|
| Same-day | 21,700 | 15,947 | 15,947 | 0.689 | 0.780 |
| Overnight | 37,820 | 29,349 | 29,348 | 0.188 | 0.143 |

The mean and median use all valid ratios. The histogram includes only ratios from 0 to 1. One valid overnight ratio is above 1, so it does not appear in the histogram count.

Overnight sessions use a smaller share of their connection time for active charging in these records. This describes past sessions. It does not confirm that the remaining time or energy can be shifted.

## Figures

- [Participant data availability](data_availability.png)
- [Connection time by arrival hour](duration_by_start_hour.png)
- [Energy use by arrival hour](energy_by_start_hour.png)
- [Active charging ratios](charging_flexibility.png)
