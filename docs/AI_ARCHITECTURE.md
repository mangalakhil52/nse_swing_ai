# MULTI-AGENT AI ARCHITECTURE — NSE SWING AI

## Overview
Nine domain specialist agents evaluate candidates concurrently under `CIOOrchestrator` (`src/agents/cio_orchestrator.py`).

## Specialist Agent Desks
1. `TechnicalAnalysisAgent`: Pattern recognition, trend, EMA alignment, RSI, VCP waves.
2. `RelativeStrengthAgent`: Mansfield Relative Strength vs Nifty 50.
3. `FundamentalAnalysisAgent`: QoQ & YoY earnings acceleration, FCF/PAT conversion, ROE, ROCE.
4. `SectorRotationAgent`: Sector rank momentum and theme rotation.
5. `InstitutionalFlowAgent`: Bulk deals, block deals, delivery volume surges.
6. `NewsIntelligenceAgent`: Materiality index, earnings surprise %, unpriced catalysts.
7. `CatalystAgent`: Regulatory approvals, order wins, capex expansions.
8. `ForensicAnalysisAgent`: Promoter pledging, audit qualifications, cashflow divergence.
9. `ThesisKillerAgent`: Independent adversarial review (SURVIVES, WEAKENED, KILLED).

## Gatekeeper & Geometry Desks (Zero Alpha Score)
- `RiskManagementAgent`: Risk gatekeeper (score = 0.0).
- `TradeConstructionAgent`: Structural levels & sizing (score = 0.0).
