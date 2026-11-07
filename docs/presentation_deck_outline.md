# EconoCausal: Final Presentation Deck Outline & Slide Blueprint

**Project Title:** EconoCausal: Prescriptive Dynamic Pricing via Double Machine Learning and Constrained Optimization  
**Author / Presenter:** Aniket Kapgate (Role: M3/M4 Full-Stack Optimization & UI Lead)  
**Evaluation:** Infotact Solutions Group 31 Final Evaluation  
**Target Duration:** 5–7 Minutes (30–45 Seconds per Slide)  
**Artifact Format:** Markdown Slide-by-Slide Blueprint with Visual Layouts, Bullets, and Verbatim Speaker Scripts  

---

## Slide 1: Title & Team Credentials

### Visual Blueprint
- **Layout:** Dark modern hero slide with enterprise blue/indigo gradient accent.
- **Center:** Large bold title: **EconoCausal: Prescriptive Dynamic Pricing Engine**.
- **Subtitle:** *Bridging Causal Machine Learning and Mathematical Optimization for Promotional Revenue Maximization*.
- **Presenter Card (Bottom Left):**
  - **Name:** Aniket Kapgate
  - **Role:** Full-Stack Optimization & UI Engineering Lead (M3 & M4 Owner)
  - **Cohort:** Infotact Solutions Internship – Group 31
- **Tech Stack Badges (Bottom Right):** Python 3.11 | SciPy (HiGHS Dual-Simplex) | FastAPI (Pydantic v2) | React 18 & Vite | Plotly.js | Docker.

### Bullet Content
- **Project Scope:** End-to-end causal inference and constrained optimization platform.
- **Core Mission:** Eliminate promotional revenue waste by isolating incremental buyers from organic conversions.
- **Role Ownership:**
  - **M3:** Mathematical modeling (LP/MCKP knapsack), SciPy HiGHS solvers, statistical drift (PSI/KS), knee-point budget detector, and FastAPI microservice.
  - **M4:** Interactive React 18 single-page dashboard, Plotly data visualizations, offline resilience engine, and automated end-to-end test suite.

### Speaker Notes
> "Good morning, members of the evaluation panel. I am Aniket Kapgate, presenting as the M3 Optimization and M4 UI Lead for Group 31. Today, I am excited to walk you through **EconoCausal**—our production-grade prescriptive dynamic pricing platform. By combining Double Machine Learning causal inference with SciPy's HiGHS linear programming solver and a responsive React dashboard, EconoCausal transforms promotional discounts from an untracked marketing expense into a mathematically provable profit driver. Let us begin with the fundamental problem EconoCausal solves."

---

## Slide 2: Executive Summary & The Problem

### Visual Blueprint
- **Layout:** Split comparison slide (Two Columns).
- **Left Column ("The Industry Flaw - Churn Propensity"):** Red alert card showing high churn risk users receiving discounts; showing $40\%$ budget waste on non-responsive customers.
- **Right Column ("The Causal Counterfactual"):** Green success card showing uplift quadrant isolation.
- **Center Callout Quote:** *"Would this customer purchase IF AND ONLY IF given a promotional discount?"*

### Bullet Content
- **The Pitfall of Predictive Marketing:**
  - Standard predictive models estimate conversion probability: $P(Y = 1 \mid X)$.
  - High propensity scores indicate customers who would buy anyway ("Sure Things").
  - Result: 30% to 40% of campaign budgets are spent subsidizing organic purchases with zero incremental lift.
- **The Counterfactual Causal Shift:**
  - EconoCausal replaces predictive correlation with causal uplift estimation.
  - Evaluates individual customer elasticity to establish incremental purchase velocity.
  - Establishes a strict boundary between organic behavior and treatment-induced conversion.

### Speaker Notes
> "In modern enterprise marketing, over 40% of promotional budgets are squandered on naive targeting. Conventional ML models rank customers by purchase propensity: $P(Y \mid X)$. But high-propensity users are predominantly *Sure Things*—customers who purchase organically regardless of a discount. Discounting them simply cannibalizes baseline margins. EconoCausal pivots from the correlational question 'Who will buy?' to the counterfactual question: 'Who will buy *if and only if* incentivized?' This counterfactual foundation protects profit margins and ensures every marketing dollar drives incremental revenue."

---

## Slide 3: Econometric Foundation (Double Machine Learning)

### Visual Blueprint
- **Layout:** 2x2 Causal Quadrants Decision Grid on the left; Neyman-Rubin potential outcomes formulation on the right.
- **Grid Visualization:**
  - Top-Left: **Sure Things** ($Y_0 \ge 0.70, \tau_i \approx 0$) $\rightarrow$ *Action: Do Not Discount (Protect Margin)*.
  - Top-Right: **Persuadables** ($\tau_i > 0, \tau_i \cdot v_i > c_i$) $\rightarrow$ *Action: Target with Optimal Tier*.
  - Bottom-Left: **Lost Causes** ($Y_0 < 0.20, \tau_i \approx 0$) $\rightarrow$ *Action: Do Not Discount (Zero Yield)*.
  - Bottom-Right: **Sleeping Dogs** ($\tau_i < 0$) $\rightarrow$ *Action: Strictly Quarantined (Harm Prevention)*.

### Bullet Content
- **Neyman-Rubin Potential Outcomes:**
  - Treatment outcome $Y_i(1)$ vs. Control outcome $Y_i(0)$.
  - Individual Treatment Effect (ITE): $\tau_i = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i]$.
  - Identified under Conditional Independence Assumption (CIA / Unconfoundedness) with Common Support.
- **The Four Behavioral Quadrants:**
  - **Persuadables:** The only cohort generating positive incremental return over cost.
  - **Sleeping Dogs:** Subsidizing them triggers churn, unsubscribe, or negative brand reactions.
  - **Sure Things & Lost Causes:** Zero incremental uplift; spend yields 0% marginal ROI.

### Speaker Notes
> "To answer this counterfactual question, Module 2 utilizes Double Machine Learning under the Neyman-Rubin Potential Outcomes framework to estimate customer-level Individual Treatment Effects ($\tau_i$). This decomposes our entire customer base into four distinct behavioral quadrants. As shown on the matrix, *Persuadables* are the only quadrant entitled to promotional investment. *Sure Things* and *Lost Causes* provide no incremental lift, while *Sleeping Dogs* actually react negatively to promotional messaging. My role in M3 is to translate these econometric uplift signals into mathematically optimal allocation decisions."

---

## Slide 4: Mathematical Optimization Engine (SciPy HiGHS)

### Visual Blueprint
- **Layout:** Top formula block showing the Continuous Knapsack LP formulation; bottom diagram contrasting 1D Single-Tier with 2D Multiple-Choice Knapsack (MCKP).
- **Visuals:** Equation callouts with color-coded variables ($\tau_i \cdot v_i - c_i$ highlighted as net objective coefficient).
- **Simplex Flowchart:** Input Cohort $\rightarrow$ LP Continuous Solve via HiGHS $\rightarrow$ Deterministic Greedy Rounding $\rightarrow$ Feasible Binary Allocation.

### Bullet Content
- **1D Single-Tier Formulation (Knapsack LP Relaxation):**
  $$\max_{x} \sum_{i=1}^N \Big(\tau_i \cdot v_i - c_i\Big) x_i \quad \text{s.t.} \quad \sum_{i=1}^N c_i x_i \le B, \quad 0 \le x_i \le 1$$
  - Solved with SciPy's C++ **HiGHS** dual-simplex solver (`scipy.optimize.linprog`).
  - Guaranteed sub-millisecond execution times for cohorts up to $100,000$ users.
- **Deterministic Integer Rounding:**
  - Fractional knapsack boundary variable is resolved via greedy feasibility check.
  - Ensures 100% budget adherence without stochastic rounding error.
- **2D Multiple-Choice Knapsack Problem (MCKP):**
  - Evaluates multi-tier discount offers ($10 vs. $20) under mutual exclusivity:
  $$\sum_{k=1}^K x_{i, k} \le 1.0 \quad \forall i \in \{1, \dots, N\}$$

### Speaker Notes
> "Once uplift is estimated, we must solve a constrained resource allocation problem under hard budget limits. We formulate single-tier allocation as a continuous knapsack linear program, solved via SciPy's high-performance HiGHS dual-simplex engine. Following continuous relaxation, we apply a deterministic greedy rounding policy that guarantees strict budget compliance. For campaigns with multiple promotional tiers—such as $10 versus $20 coupons—we formulate a 2D Multiple-Choice Knapsack Problem. The HiGHS solver enforces mutual exclusivity, ensuring no customer receives more than one offer while maximizing total portfolio net profit."

---

## Slide 5: Domain Pruning & Statistical Drift Detection

### Visual Blueprint
- **Layout:** Two horizontal panels.
- **Left Panel:** Flow diagram of Pre-Solver Domain Pruning reducing matrix dimensions by 62%.
- **Right Panel:** Visual distribution shift chart illustrating PSI decile bin divergence and the Kolmogorov-Smirnov supremum distance $D$.

### Bullet Content
- **Pre-Solver Domain Pruning:**
  - Automatically fixes $x_i = 0$ for all Sleeping Dogs ($\tau_i \le 0$).
  - Drops non-profitable candidates where $(\tau_i \cdot v_i - c_i \le 0)$ under profit maximization.
  - Short-circuits micro-budgets ($B < \min c_i$) in zero iterations.
  - Shrinks active LP constraint matrix by over 60%, drastically cutting peak memory footprint.
- **Real-Time Statistical Drift Detection:**
  - **Population Stability Index (PSI):**
    $$\text{PSI} = \sum_{b=1}^{10} (P_b - Q_b) \ln(P_b / Q_b)$$
    - Thresholds: $<0.10$ Stable, $0.10 - 0.25$ Moderate, $\ge 0.25$ Critical alert.
  - **Two-Sample Kolmogorov-Smirnov Test ($D$):** Assesses continuous cumulative distribution divergence.

### Speaker Notes
> "Real-world enterprise systems demand operational efficiency and data integrity. In M3, we implement two critical features. First, pre-solver domain pruning: by identifying Sleeping Dogs and negative-yield candidates beforehand, we prune over 60% of matrix variables before invoking HiGHS, slashing solve latencies. Second, because customer behavior shifts over time, we embedded a real-time statistical drift engine. By monitoring scoring distributions against calibration baselines via Population Stability Index (PSI) and Kolmogorov-Smirnov tests, the system alerts operators the instant macro market conditions cause model drift."

---

## Slide 6: System Architecture & API Design

### Visual Blueprint
- **Layout:** End-to-end Microservice Architecture Diagram.
- **Flow:** Client UI / External Systems $\rightarrow$ FastAPI Router (12 Endpoints) $\rightarrow$ Pydantic v2 Validation $\rightarrow$ LRU Allocation Cache (SHA-256) $\rightarrow$ Optimization Core (HiGHS / Drift / Knee-Point) $\rightarrow$ JSON Response.
- **Table Summary:** Compact summary table of the 12 production REST endpoints.

### Bullet Content
- **FastAPI Production Microservice:**
  - 12 fully typed, production-ready REST endpoints documented via OpenAPI 3.1.
  - Pydantic v2 schemas enforcing strict validation on customer vectors, costs, and budgets.
  - Clean separation: API routes, solver interfaces, caching layers, and statistical modules.
- **Sub-Millisecond In-Memory LRU Cache (`AllocationCache`):**
  - Thread-safe caching layer keyed on the SHA-256 hash of solver inputs.
  - Sub-millisecond response latency on repeated queries with `cache_hit: true` telemetry.
- **Knee-Point Budget Recommendation:**
  - Implements vector chord distance (Kneedle curvature) to recommend the optimal budget where marginal ROI diminishes.

### Speaker Notes
> "On Slide 6, we see our backend microservice architecture. The entire engine is exposed through an asynchronous FastAPI REST API with 12 production endpoints. Every request is validated by Pydantic v2 schemas. To support high-throughput scoring, we implemented a thread-safe in-memory LRU cache keyed by SHA-256 hashes of input vectors, returning repeated runs in under a millisecond. We also built an automated Knee-Point Budget Detector that computes the Pareto frontier of diminishing returns, recommending the exact capital expenditure that maximizes marginal ROI."

---

## Slide 7: Interactive Dashboard & UI Walkthrough (React + Plotly)

### Visual Blueprint
- **Layout:** Screenshot mock of the EconoCausal interactive dashboard.
- **Callouts:**
  1. *Top Navbar:* Live backend connection status indicator (`Online: 127.0.0.1:8000`).
  2. *Left Sidebar:* Budget slider, objective selector, tier mode toggle, and action buttons.
  3. *Top KPI Cards:* Total Spend, Budget Utilization %, Incremental Revenue, and Net Profit.
  4. *Visual Analytics Grid:* Plotly Qini Cumulative Uplift Curve and Pareto Budget Frontier with Knee-Point pin.
  5. *Allocation Matrix:* Searchable, paginated customer table with causal quadrant color badges.

### Bullet Content
- **Role M4 Deliverable: React 18 & Plotly.js SPA:**
  - Built with React 18, Vite, and Tailwind CSS for rapid enterprise responsiveness.
  - Direct API proxy communication with fallback to synthetic client simulation if backend is offline.
- **Visual Analytics Suite:**
  - **Qini Curve:** Visualizes cumulative incremental conversions vs. random targeting baseline.
  - **Pareto Frontier:** Plots expected net profit across budget steps with automatic knee-point detection.
  - **Allocation Matrix:** Real-time customer search, filtering, and quadrant categorization (Persuadable, Sure Thing, etc.).
- **Standalone Distribution:**
  - Single-file zero-dependency HTML build (`frontend/standalone.html`) for instant client-side demos.

### Speaker Notes
> "As M4 lead, I designed and implemented the interactive frontend dashboard using React 18, Tailwind CSS, and Plotly.js. Business users can dynamically adjust campaign budgets, select optimization objectives, and immediately visualize financial KPIs. The Qini uplift curve proves the model's superiority over random baselines, while the budget frontier plot clearly highlights the knee point of diminishing returns. Below the charts, our customer allocation matrix allows marketers to search and inspect individual customer decisions. Furthermore, the frontend includes an offline greedy solver fallback, ensuring flawless client presentations even without network connectivity."

---

## Slide 8: Empirical Benchmark Results

### Visual Blueprint
- **Layout:** Prominent side-by-side benchmark comparison table with key metrics highlighted in green and gold.
- **Bar Chart Graphic:** Net Profit comparison ($340 Blanket vs. $3,140 Causal LP).

### Bullet Content
- **Empirical Experiment ($N = 1,000$ Customers, Budget Ceiling $B = \$2,500$):**

| Strategy | Total Spend | Incremental Revenue | Net Profit | Campaign ROI | Sleeping Dogs Quarantined |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Blanket Discounting** | $2,500.00 | $2,840.50 | $340.50 | 13.6% | 0 (All Targeted) |
| **Random Rollout Baseline** | $2,500.00 | $2,650.00 | $150.00 | 6.0% | 0 (Random Sample) |
| **Propensity Model Targeting** | $2,500.00 | $3,120.00 | $620.00 | 24.8% | 12 (Partial Pruning) |
| **EconoCausal SciPy LP** | **$1,840.00** | **$4,980.20** | **$3,140.20** | **170.7%** | **184 (100% Quarantined)** |

- **Key Financial Takeaways:**
  - **9.2x Net Profit Surge:** Increased net profit from $340.50 to $3,140.20 on the same customer cohort.
  - **Capital Conservation:** Achieved higher return while spending 26.4% *less* than the maximum budget limit.
  - **100% Harm Prevention:** Quarantined all 184 Sleeping Dogs, preventing brand erosion and customer churn.

### Speaker Notes
> "Here are the empirical benchmark results on a standardized cohort of 1,000 customers. Compared to standard blanket discounting, EconoCausal delivers a **9.2x increase in net profit**—jumping from $340 to over $3,140. Notice that EconoCausal actually conserved capital, spending only $1,840 of the $2,500 budget because the solver recognized that allocating further funds to marginal users would yield negative returns. Most importantly, EconoCausal quarantined 100% of the 184 Sleeping Dogs in the cohort, preventing costly customer churn that blanket campaigns actively provoke."

---

## Slide 9: Quality Assurance & Engineering Rigor

### Visual Blueprint
- **Layout:** Three achievement panels.
- **Panel 1:** Test Suite Badge showing **130 / 130 Tests Passing (100%)** in 1.89 seconds.
- **Panel 2:** Multi-Stage Docker Build Architecture (180MB minimal image footprint).
- **Panel 3:** Continuous Integration & Edge-Case Fuzzing Architecture.

### Bullet Content
- **Automated Verification Suite:**
  - **130 Automated Tests:** 100% pass rate across optimization, API routes, drift, CLI, and frontend.
  - **Sub-2-Second Execution:** Entire test suite discovers and executes in 1.89 seconds.
  - **Boundary & Fuzz Testing:** 30 randomized customer cohorts, micro-budgets ($10^{-6}$), massive budgets ($10^7$), and degenerate zero-uplift profiles.
- **Production Containerization (`Dockerfile`):**
  - Multi-stage build based on `python:3.11-slim` separating compiler toolchains from runtime.
  - Built-in 30-second automated healthcheck probe (`curl -f http://localhost:8000/health`).
  - Optimized `.dockerignore` eliminating test caches, git metadata, and local development artifacts.

### Speaker Notes
> "To guarantee enterprise-grade reliability, we subjected EconoCausal to rigorous engineering quality assurance. Our test suite includes 130 automated unit, integration, deployment, and numerical fuzzing tests—all executing in under two seconds with a 100% pass rate. We tested extreme boundary conditions, including micro-budgets of $10^{-6}$ and multi-million dollar allocations. Finally, the application is packaged in a multi-stage Docker image with built-in Kubernetes healthcheck probes, ready for immediate cloud deployment on AWS ECS, GCP Cloud Run, or Kubernetes."

---

## Slide 10: Conclusion & Business Impact

### Visual Blueprint
- **Layout:** Executive summary closing slide.
- **Key Metric Pillars:**
  - **9.2x** Net Profit Multiple
  - **100%** Sleeping Dog Quarantining
  - **<2.5 ms** Solver Execution Latency
  - **130 / 130** Passing Test Suite
- **Roadmap Horizons Card:** Immediate Production Delivery $\rightarrow$ Continuous Pricing Curves $\rightarrow$ Streaming Kafka Integration.

### Bullet Content
- **Core Value Delivered:**
  - Replaced wasteful propensity heuristics with provably optimal causal decision-making.
  - Unified causal econometrics, mathematical LP solvers, and modern interactive UI into a cohesive production platform.
  - Delivered comprehensive technical documentation, including the 291-line formal methodology dossier.
- **Production Readiness:**
  - Available via REST microservice, CLI batch tool, React web application, and Docker container.
- **Future Roadmap:**
  - Extending discrete tiers to continuous price elasticity curves.
  - Real-time streaming Kafka inference for instantaneous personalized pricing.

### Speaker Notes
> "In conclusion, EconoCausal bridges the critical divide between econometric theory and commercial execution. By substituting predictive churn heuristics with prescriptive causal optimization, we provide businesses with a mathematically verified competitive advantage: higher net profits, reduced promotional waste, and zero customer alienation. The platform is complete, thoroughly tested, fully documented, and production ready. Thank you for your time, and I now welcome any questions from the panel."

---

## Summary of Deck Execution Parameters

- **Total Slides:** 10 Slides
- **Target Presentation Time:** 5 Minutes 30 Seconds
- **Average Time per Slide:** ~33 Seconds
- **Target Audience:** Technical evaluators, econometricians, marketing executives, and software architects.


## Final Evaluation Readiness Checklist (07 Nov 2026)
- Slide Deck Architecture: 10 slides complete with visual blueprints, bullet points, and 30-second speaker scripts.
- Live Demonstration Prepared: Interactive Swagger UI (/docs) and React/Plotly dashboard (/dashboard) verified.
- Code Freeze Confirmed: All 148 automated unit and integration tests passing with 100% pass rate.
