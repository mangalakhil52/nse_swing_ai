# POINT-IN-TIME SAFETY & LEAKAGE GUARD — NSE SWING AI

## Overview
Point-In-Time (PIT) safety guarantees that decisions made at time $T$ consume ONLY information published/available on or before $T$.

## Central PIT Enforcer
Implemented in `PointInTimeFilter` (`src/data/point_in_time.py`):
- **Market Data**: Bars timestamped $\le \text{as\_of\_date}$.
- **Fundamentals**: Quarterly/annual results filed on or before $\text{as\_of\_date}$.
- **News**: Articles published $\le \text{as\_of\_date}$.
- **Corporate Events**: Announcements published $\le \text{as\_of\_date}$.

`available_at <= decision_timestamp` is enforced centrally at the data access layer.
