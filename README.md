# EconoCausal: Dynamic Pricing & Budget Allocation via Double Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Optimization](https://img.shields.io/badge/solver-SciPy%20HiGHS-orange.svg)](https://scipy.org/)
[![Test Suite](https://img.shields.io/badge/tests-120%2F120%20passing-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-multi--stage%20ready-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Enterprise prescriptive pricing engine integrating causal inference (Double Machine Learning / Individual Treatment Effect estimation), mathematical linear programming (SciPy HiGHS), and statistical distribution drift monitoring. EconoCausal solves budget-constrained promotional allocation by identifying customer-level incremental conversion lift rather than correlational purchase propensities.

---

## 1. Core Problem & Causal Framework

### Correlation vs. Causation in Promotional Campaigns
Standard predictive models score customer purchase probability $P(Y=1 \mid X)$. Allocating discounts to customers with high conversion probabilities wastes budget on buyers who would purchase anyway ("Sure Things") and risks disturbing customers who react negatively to promotional messaging ("Sleeping Dogs").

EconoCausal measures the **Individual Treatment Effect (ITE)**, also termed causal uplift:
$$\tau_i = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i]$$
where $Y_i(1)$ is the potential outcome under discount and $Y_i(0)$ is the baseline outcome without discount.

```
                      Causal Quadrant Decision Matrix
     High
      ▲
      │    SURE THINGS                  PERSUADABLES
      │    Baseline Conversion ≥ 0.70   ITE > 0  &  (ITE × Value) > Cost
      │    Organic purchasers           Target with Promotional Discount
Baseline│    ACTION: Do Not Discount      ACTION: Allocate Optimal Tier
Prob  │──────────────────────────────────────────────────────────
 (Y₀) │    LOST CAUSES                  SLEEPING DOGS
      │    Baseline Conversion < 0.20   ITE < 0
      │    Non-responsive               Offer Triggers Negative Reaction
      │    ACTION: Do Not Discount      ACTION: Strictly Do Not Discount
      ▼
      └──────────────────────────────────────────────────────────►
                                 Causal Uplift (ITE: Y₁ - Y₀)
```

Targeting decisions based purely on correlation disproportionately allocate budget to Sure Things and Lost Causes. EconoCausal restricts promotional capital strictly to **Persuadables**, maximizing net incremental profit.

---

## 2. Architecture & Module Ownership

The repository is modularized into four distinct layers:

| Layer / Directory | Milestone / Module | Lead Engineer | Primary Responsibilities |
|:---|:---|:---|:---|
| `src/data/` | **M1: Ingestion & Feature Store** | Data Engineering | Customer profile ingestion, RFM feature engineering, covariate validation. |
| `src/causal/` | **M2: Causal ML Engine** | Causal ML Team | Double Machine Learning (DML), Causal Forests, ITE scoring, DoWhy DAGs, refutation tests. |
| `src/optimization/` | **M3: Prescriptive Solvers** | **Aniket Kapgate** | SciPy HiGHS LP solvers (1D & 2D MCKP), in-memory LRU cache, PSI/KS drift engine, Qini/AUUC uplift metrics, Knee-Point budget analyzer, Monte Carlo simulator. |
| `src/api/` | **M3: Production API & CLI** | **Aniket Kapgate** | FastAPI service, 10+ REST endpoints, Pydantic v2 data contracts, OpenAPI 3.1 exporter, offline batch allocation CLI. |
| `frontend/` | **M4: Interactive Dashboard** | Frontend Team | React, Vite, Plotly interactive frontier charts, policy comparison previews. |

---

## 3. Mathematical Optimization Engine

### 1D Continuous Knapsack Relaxation & Deterministic Integer Rounding
For single-tier allocations, each customer $i \in \{1, \dots, N\}$ has decision variable $x_i \in [0, 1]$, promotional cost $c_i$, customer value $v_i$, and estimated uplift $\tau_i$.

$$\max_{x} \sum_{i=1}^N \Big(\tau_i \cdot v_i - c_i\Big) x_i$$
$$\text{subject to} \quad \sum_{i=1}^N c_i x_i \le B, \quad 0 \le x_i \le 1 \quad \forall i$$

1. **Candidate Pre-filtering:** Subsets eligible candidates $\mathcal{E} = \{i : \tau_i > 0 \text{ and } \tau_i v_i - c_i > 0\}$. Customers with $\tau_i \le 0$ (Sleeping Dogs) or negative expected net profit are pruned upfront ($x_i = 0$).
2. **SciPy HiGHS LP Solve:** Solves continuous relaxation via `scipy.optimize.linprog(method='highs')`.
3. **Deterministic Rounding:** If integer rounding is enabled, fractional boundary variables are mapped to $\{0, 1\}$ using greedy ROI sorting without violating total budget $B$.

### 2D Multiple-Choice Knapsack Problem (MCKP)
When multiple discount tiers $k \in \{1, \dots, K\}$ are available (e.g., $10 vs. $20 vs. $30), decision variables $x_{i, k} \in \{0, 1\}$ assign at most one tier per customer:

$$\max_{x} \sum_{i=1}^N \sum_{k=1}^K \Big(\tau_{i, k} \cdot v_i - c_{i, k}\Big) x_{i, k}$$
$$\text{subject to} \quad \sum_{i=1}^N \sum_{k=1}^K c_{i, k} x_{i, k} \le B$$
$$\sum_{k=1}^K x_{i, k} \le 1.0 \quad \forall i \in \{1, \dots, N\} \quad \text{(Mutual Exclusivity)}$$
$$0 \le x_{i, k} \le 1.0 \quad \forall i, k$$

### Statistical Drift Engine (PSI & Kolmogorov-Smirnov)
Detects distribution shifts between training baseline ITEs ($Q$) and production scoring ITEs ($P$):

$$\text{PSI} = \sum_{b=1}^B \big(P_b - Q_b\big) \times \ln\left(\frac{P_b}{Q_b}\right)$$

- $\text{PSI} < 0.10$: Stable distribution (`none`).
- $0.10 \le \text{PSI} < 0.25$: Moderate distribution drift (`moderate`).
- $\text{PSI} \ge 0.25$: Significant distribution shift (`critical`).

Two-sample Kolmogorov-Smirnov test measures empirical CDF distance:
$$D = \sup_{u} |F_P(u) - F_Q(u)|$$

### Knee-Point Budget Detection (Kneedle / Chord Distance)
Evaluates allocation outcomes across discrete budgets $B \in [B_{\min}, B_{\max}]$. Coordinates $(x_i, y_i) = (\text{Budget}_i, \text{Profit}_i)$ are normalized to $[0, 1]$:
$$d_i = \frac{|\Delta x (y_{i, \text{norm}} - y_{0, \text{norm}}) - \Delta y (x_{i, \text{norm}} - x_{0, \text{norm}})|}{\sqrt{\Delta x^2 + \Delta y^2}}$$
The optimal recommendation $k = \operatorname{argmax}_i d_i$ isolates the elbow where marginal return degrades:
$$\text{marginal\_roi}_{\text{post}} = \frac{y_{k+1} - y_k}{x_{k+1} - x_k}, \quad \text{ratio} = \frac{\text{marginal\_roi}_{\text{post}}}{\text{marginal\_roi}_{\text{pre}}}$$

---

## 4. REST API Reference

The FastAPI service exposes 12 endpoints organized into Monitoring, Scoring, and Optimization categories:

| HTTP Method | Endpoint Path | Category | Description |
|:---|:---|:---|:---|
| `GET` | `/health` | Monitoring | Liveness probe verifying service health and environment metadata. |
| `GET` | `/ready` | Monitoring | Readiness probe confirming solver, configuration, and logging subsystems. |
| `GET` | `/api/v1/mock-customers` | Scoring & Previews | Generates synthetic customer cohort with realistic ITE distributions for UI testing. |
| `POST` | `/api/v1/score-summary` | Scoring & Previews | Computes aggregate cohort uplift statistics, mean ITE, and quadrant counts. |
| `POST` | `/api/v1/allocate` | Optimization | Runs 1D budget-constrained LP optimization returning customer assignments. |
| `POST` | `/api/v1/allocate-multi-tier` | Optimization | Solves 2D MCKP multi-tier discount optimization with mutual exclusivity. |
| `POST` | `/api/v1/compare-policies` | Optimization | Benchmarks causal SciPy optimization against naive blanket discounting. |
| `POST` | `/api/v1/drift-check` | Optimization | Evaluates PSI and two-sample Kolmogorov-Smirnov drift between ITE distributions. |
| `POST` | `/api/v1/executive-report` | Optimization | Produces executive summary with quadrant breakdown and ROI figures. |
| `POST` | `/api/v1/recommend-budget` | Optimization | Sweeps budget frontier and returns knee-point budget recommendation. |
| `POST` | `/api/v1/simulate-confidence` | Optimization | Executes Monte Carlo perturbation to evaluate policy robustness and risk. |
| `POST` | `/api/v1/uplift-curve` | Optimization | Computes cumulative uplift, Qini curve deciles, and AUUC metrics. |

Interactive Swagger documentation is available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

---

## 5. Offline Batch CLI

The built-in CLI enables high-throughput offline batch allocation for CSV and JSON datasets without requiring the HTTP server.

```bash
# Optimize customer discount allocation from CSV
python -m src.cli \
  --input data/customer_scores.csv \
  --budget 2500.0 \
  --objective maximize_profit \
  --output data/allocations_output.csv
```

### Supported CLI Arguments
- `-i, --input` *(required)*: Path to input CSV or JSON containing customer records (`customer_id`, `predicted_ite`, `treatment_cost`, `customer_value`).
- `-b, --budget` *(required)*: Total promotional campaign budget ceiling ($B > 0$).
- `-o, --output` *(default: `allocations_output.csv`)*: Destination path for allocation output file.
- `--objective` *(default: `maximize_profit`)*: Optimization goal (`maximize_profit`, `maximize_revenue`, or `maximize_conversions`).
- `--min-spend` *(default: `0.0`)*: Minimum expenditure threshold per targeted customer.

---

## 6. Production Docker Deployment

A multi-stage Docker build provides a minimal runtime image (~180MB) with an active container healthcheck.

```bash
# 1. Build the production image
docker build -t econocausal:v1 .

# 2. Run container with port forwarding
docker run -d \
  --name econocausal-engine \
  -p 8000:8000 \
  --restart unless-stopped \
  econocausal:v1

# 3. Check container health status
docker inspect --format='{{json .State.Health.Status}}' econocausal-engine
```

Container specification details:
- **Base:** `python:3.11-slim`
- **Port:** `8000`
- **Healthcheck:** `curl -f http://localhost:8000/health || exit 1` (Interval: 30s, Timeout: 5s, Retries: 3)
- **Execution:** `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

---

## 7. Verification & Test Suite

The test suite covers unit logic, REST endpoints, linear programming constraints, numerical boundary conditions, and randomized fuzzing.

```bash
# Run complete test suite via Python unittest
python -m unittest discover tests -v
```

### Test Suite Structure (120 Tests)
- `tests/test_api/test_config.py`: Environment settings and configuration overrides.
- `tests/test_api/test_deployment.py`: Dockerfile and `.dockerignore` artifact structural validation.
- `tests/test_api/test_drift_and_stress.py`: Drift endpoints and high-volume stress tests (1,000 customers).
- `tests/test_api/test_health.py`: Liveness (`/health`) and readiness (`/ready`) probe verification.
- `tests/test_api/test_openapi_export.py`: OpenAPI 3.1 JSON export and Markdown summary generation.
- `tests/test_api/test_optimization_routes.py`: Core REST endpoint integration and validation error handling.
- `tests/test_api/test_scoring.py`: Synthetic customer data generation and score aggregation.
- `tests/test_cli.py`: Batch CLI file I/O, argument parsing, and output formatting.
- `tests/test_optimization/test_baseline.py`: Blanket discount and greedy heuristic solvers.
- `tests/test_optimization/test_cache.py`: In-memory thread-safe LRU allocation result caching.
- `tests/test_optimization/test_drift.py`: PSI calculations, KS test statistics, and drift categorization.
- `tests/test_optimization/test_knee_point.py`: Geometric chord distance knee detector and marginal ROI analyzer.
- `tests/test_optimization/test_reporting.py`: Causal quadrant classifier and executive campaign report generator.
- `tests/test_optimization/test_simulation.py`: Monte Carlo policy confidence and risk bounds sampling.
- `tests/test_optimization/test_solver.py`: 1D and 2D SciPy HiGHS LP solves, mutual exclusivity, and budget constraints.
- `tests/test_optimization/test_uplift_metrics.py`: Cumulative uplift curves, AUUC, and Qini coefficient calculations.
- `tests/test_optimization/test_fuzz_bounds.py`: Randomized numerical fuzzing, micro/massive budget scaling, and degenerate cohorts.

```text
Ran 120 tests in 4.151s
OK
```

