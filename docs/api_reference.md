# EconoCausal API

**Version:** 0.1.0  
**Description:** Prescriptive AI Dynamic Pricing & Discount Allocation Engine

## Summary of Endpoints

| HTTP Method | Endpoint | Request Model | Description |
|:---|:---|:---|:---|
| `GET` | `/api/v1/mock-customers` | `None` | Generate synthetic customer ITE dataset for UI testing |
| `POST` | `/api/v1/score-summary` | `List[CustomerDataPoint]` | Calculate aggregate uplift statistics across a customer cohort |
| `POST` | `/api/v1/allocate` | `OptimizationRequest` | Compute optimal discount allocation under budget constraints |
| `POST` | `/api/v1/ingest-and-allocate` | `IngestAndAllocateRequest` | Ingest raw causal ML records from M2 and compute optimal discount allocation |
| `POST` | `/api/v1/allocate-multi-tier` | `MultiTierOptimizationRequest` | Compute optimal multi-tier discount allocation under budget constraints |
| `POST` | `/api/v1/compare-policies` | `OptimizationRequest` | Benchmark causal optimization against naive blanket discount baseline |
| `POST` | `/api/v1/drift-check` | `DriftCheckRequest` | Evaluate distribution drift between reference and current ITE distributions |
| `POST` | `/api/v1/uplift-curve` | `UpliftCurveRequest` | Generate cumulative uplift curve and Qini metrics for campaign evaluation |
| `POST` | `/api/v1/executive-report` | `OptimizationRequest` | Generate executive summary report with causal quadrant segmentation |
| `POST` | `/api/v1/simulate-confidence` | `MonteCarloSimulationRequest` | Run Monte Carlo simulation to quantify policy risk and confidence bounds |
| `GET` | `/` | `None` | Root |
| `GET` | `/health` | `None` | Health Check |
| `GET` | `/ready` | `None` | Readiness Check |
