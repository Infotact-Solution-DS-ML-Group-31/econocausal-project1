# EconoCausal API

**Version:** 0.1.0  
**Description:** Prescriptive AI Dynamic Pricing & Discount Allocation Engine

## Summary of Endpoints

| HTTP Method | Endpoint Path | Tag / Category | Description |
|:---|:---|:---|:---|
| `GET` | `/api/v1/mock-customers` | Scoring & Previews | Generate synthetic customer ITE dataset for UI testing |
| `POST` | `/api/v1/score-summary` | Scoring & Previews | Calculate aggregate uplift statistics across a customer cohort |
| `POST` | `/api/v1/allocate` | Optimization & Allocation | Compute optimal discount allocation under budget constraints |
| `POST` | `/api/v1/ingest-and-allocate` | Optimization & Allocation | Ingest raw causal ML records from M2 and compute optimal discount allocation |
| `POST` | `/api/v1/allocate-multi-tier` | Optimization & Allocation | Compute optimal multi-tier discount allocation under budget constraints |
| `POST` | `/api/v1/compare-policies` | Optimization & Allocation | Benchmark causal optimization against naive blanket discount baseline |
| `POST` | `/api/v1/drift-check` | Optimization & Allocation | Evaluate distribution drift between reference and current ITE distributions |
| `POST` | `/api/v1/uplift-curve` | Optimization & Allocation | Generate cumulative uplift curve and Qini metrics for campaign evaluation |
| `POST` | `/api/v1/executive-report` | Optimization & Allocation | Generate executive summary report with causal quadrant segmentation |
| `POST` | `/api/v1/simulate-confidence` | Optimization & Allocation | Run Monte Carlo simulation to quantify policy risk and confidence bounds |
| `GET` | `/` | General | Root service metadata and online status |
| `GET` | `/health` | Monitoring | Liveness probe for health monitoring |
| `GET` | `/ready` | Monitoring | Readiness probe verifying subsystem operational readiness |
