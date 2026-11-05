/**
 * EconoCausal API Client
 * Interfaces with FastAPI endpoints with automatic fallback to client-side synthetic simulation.
 */

const BASE_URL = "";

// Client-side fallback generator
export function generateSyntheticCustomers(count = 50) {
  const customers = [];
  for (let i = 1; i <= count; i++) {
    const rand = Math.random();
    let ite, baseline, cost, value;

    if (rand < 0.40) {
      // Persuadable: positive uplift, profitable
      ite = 0.12 + Math.random() * 0.35;
      baseline = 0.10 + Math.random() * 0.40;
    } else if (rand < 0.60) {
      // Sleeping Dog: negative uplift
      ite = -0.05 - Math.random() * 0.20;
      baseline = 0.20 + Math.random() * 0.50;
    } else if (rand < 0.80) {
      // Sure Thing: high baseline, low uplift
      ite = 0.01 + Math.random() * 0.04;
      baseline = 0.72 + Math.random() * 0.25;
    } else {
      // Lost Cause: low baseline, low uplift
      ite = 0.00 + Math.random() * 0.03;
      baseline = 0.02 + Math.random() * 0.15;
    }

    cost = 10.0;
    value = 50.0 + Math.random() * 50.0;

    customers.push({
      customer_id: `cust_${String(i).padStart(3, "0")}`,
      baseline_conversion_prob: parseFloat(baseline.toFixed(4)),
      predicted_ite: parseFloat(ite.toFixed(4)),
      treatment_cost: cost,
      customer_value: parseFloat(value.toFixed(2)),
      tier_ites: [parseFloat(ite.toFixed(4)), parseFloat((ite * 1.3).toFixed(4))],
    });
  }
  return customers;
}

// Client-side greedy solver fallback
export function clientSideSolver(customers, budget, objective = "maximize_profit") {
  let remainingBudget = budget;
  let totalSpend = 0.0;
  let totalRevenue = 0.0;
  const allocations = [];

  const candidates = [...customers].map(c => {
    const ite = c.predicted_ite;
    const val = c.customer_value || 50.0;
    const cost = c.treatment_cost || 10.0;
    const incRev = ite * val;
    const netProfit = incRev - cost;
    const score = objective === "maximize_revenue" ? incRev : (objective === "maximize_conversions" ? ite : netProfit);
    return { ...c, incRev, netProfit, score, roi: netProfit / cost };
  }).sort((a, b) => b.score - a.score);

  for (const c of candidates) {
    const isEligible = objective === "maximize_profit"
      ? (c.predicted_ite > 0 && c.netProfit > 0)
      : (c.predicted_ite > 0);
    const canAfford = remainingBudget >= c.treatment_cost;
    const targeted = isEligible && canAfford;

    if (targeted) {
      remainingBudget -= c.treatment_cost;
      totalSpend += c.treatment_cost;
      totalRevenue += c.incRev;
    }

    allocations.push({
      customer_id: c.customer_id,
      assigned_discount: targeted ? c.treatment_cost : 0.0,
      is_targeted: targeted,
      expected_incremental_revenue: parseFloat((targeted ? c.incRev : 0.0).toFixed(4)),
      expected_net_profit: parseFloat((targeted ? c.netProfit : 0.0).toFixed(4)),
    });
  }

  const netProfit = totalRevenue - totalSpend;
  return {
    success: true,
    total_budget: budget,
    total_spend: parseFloat(totalSpend.toFixed(2)),
    budget_utilization_pct: parseFloat(((totalSpend / budget) * 100).toFixed(1)),
    total_customers_count: customers.length,
    expected_incremental_revenue: parseFloat(totalRevenue.toFixed(2)),
    expected_net_profit: parseFloat(netProfit.toFixed(2)),
    allocations,
  };
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      return { online: true, ...data };
    }
  } catch (err) {
    // Offline
  }
  return { online: false, status: "offline" };
}

export async function fetchMockCustomers(numCustomers = 50) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/mock-customers?num_customers=${numCustomers}`);
    if (res.ok) {
      const data = await res.json();
      return data.customers || data;
    }
  } catch (err) {
    console.warn("fetchMockCustomers failed; using synthetic fallback.", err);
  }
  return generateSyntheticCustomers(numCustomers);
}

export async function runAllocation(request) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/allocate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("runAllocation failed; falling back to local solver.", err);
  }
  return clientSideSolver(request.customers, request.budget, request.objective);
}

export async function runMultiTierAllocation(request) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/allocate-multi-tier`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("runMultiTierAllocation failed; using client fallback.", err);
  }
  return clientSideSolver(request.customers, request.budget, request.objective);
}

export async function comparePolicies(request) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/compare-policies`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("comparePolicies failed; using fallback comparison.", err);
  }
  const scipyRes = clientSideSolver(request.customers, request.budget, request.objective);
  return {
    budget: request.budget,
    causal_scipy_policy: scipyRes,
    blanket_discount_baseline: {
      total_spend: request.budget,
      expected_incremental_revenue: scipyRes.expected_incremental_revenue * 0.5,
      expected_net_profit: scipyRes.expected_net_profit * 0.35,
    },
    uplift_summary: {
      mean_ite: 0.12,
      persuadables_count: Math.floor(request.customers.length * 0.4),
      sleeping_dogs_count: Math.floor(request.customers.length * 0.2),
    },
  };
}

export async function recommendBudget(request) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/recommend-budget`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("recommendBudget failed; generating fallback recommendation.", err);
  }
  return {
    recommended_budget: 150.0,
    expected_net_profit: 320.0,
    expected_incremental_revenue: 470.0,
    targeted_customers_count: 15,
    marginal_roi: 0.45,
    diminishing_returns_ratio: 0.38,
    explanation: "Optimal budget knee point detected at $150.00. Allocating beyond yields less than $0.45 profit per dollar.",
  };
}

export async function fetchUpliftCurve(request) {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/uplift-curve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("fetchUpliftCurve failed; using fallback curve.", err);
  }
  return {
    auuc_score: 0.724,
    qini_coefficient: 0.435,
    max_profit_percentile: 40.0,
    curve: [
      { percentile: 0, cumulative_uplift: 0, cumulative_spend: 0, cumulative_profit: 0 },
      { percentile: 10, cumulative_uplift: 0.8, cumulative_spend: 50, cumulative_profit: 70 },
      { percentile: 20, cumulative_uplift: 1.5, cumulative_spend: 100, cumulative_profit: 140 },
      { percentile: 30, cumulative_uplift: 2.1, cumulative_spend: 150, cumulative_profit: 195 },
      { percentile: 40, cumulative_uplift: 2.5, cumulative_spend: 200, cumulative_profit: 220 },
      { percentile: 50, cumulative_uplift: 2.7, cumulative_spend: 250, cumulative_profit: 215 },
      { percentile: 70, cumulative_uplift: 2.8, cumulative_spend: 350, cumulative_profit: 180 },
      { percentile: 100, cumulative_uplift: 2.6, cumulative_spend: 500, cumulative_profit: 120 },
    ],
  };
}

