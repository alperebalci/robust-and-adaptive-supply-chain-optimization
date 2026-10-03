# System Dynamics for Operations

A discrete stock-flow model for operational feedback, supply-line delay and the bullwhip effect.

The implementation represents:

- inventory and backlog as stocks;
- customer demand, shipments, receipts and replenishment orders as flows;
- an explicit delayed supply line;
- exponential demand forecasting;
- inventory-position feedback;
- order smoothing;
- capacity limits;
- bullwhip measurement `Var(orders) / Var(demand)`;
- policy sweeps trading inventory, backlog and order volatility.

The constant-demand test initializes both inventory and the supply line at equilibrium. Step-demand tests then show how aggressive forecasting/replenishment feedback can amplify order variance, while smoother feedback can reduce oscillation.

Run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

This is a transparent discrete-time system-dynamics benchmark, not a calibrated claim about a specific supply chain. Multi-echelon feedback, endogenous lead times, capacity expansion and empirical calibration are natural extensions.
