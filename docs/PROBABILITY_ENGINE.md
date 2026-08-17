# EMPIRICAL PROBABILITY & EXPECTED VALUE ENGINE — NSE SWING AI

## Overview
Calculates empirical win rates, sample sizes ($n \ge 30$), confidence intervals, and Net Expected Value (Net EV).

## Net Expected Value Formula
$$\text{Net EV} = P(\text{Win}) \times \text{Target}_1\% - (1 - P(\text{Win})) \times \text{StopLoss}\% - \text{Friction}\%$$

Friction includes brokerage, STT, exchange fees, GST, stamp duty, bid-ask spread, and ADTV market impact slippage.

## Calibration Status
- `CALIBRATED_PROBABILITY`: Backed by $\ge 30$ historical observations.
- `UNAVAILABLE`: Sample size $< 30$. Setups with $n < 30$ are automatically disqualified.
