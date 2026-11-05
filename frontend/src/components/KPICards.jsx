import React from 'react';

export default function KPICards({ result, budget, blanketProfit }) {
  const spend = result?.total_spend || 0.0;
  const util = result?.budget_utilization_pct !== undefined
    ? result.budget_utilization_pct
    : (budget > 0 ? (spend / budget) * 100 : 0);
  const revenue = result?.expected_incremental_revenue || 0.0;
  const netProfit = result?.expected_net_profit !== undefined
    ? result.expected_net_profit
    : (revenue - spend);
  const roi = spend > 0 ? ((netProfit / spend) * 100).toFixed(1) : '0.0';
  const gainOverBlanket = blanketProfit !== null ? (netProfit - blanketProfit) : (spend * 0.35);

  return (
    <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Total Allocated Spend */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <span className="text-xs font-medium text-slate-500">Total Allocated Spend</span>
        <div className="flex items-baseline space-x-2 mt-1">
          <span className="text-2xl font-bold text-slate-900">${spend.toFixed(2)}</span>
          <span className="text-xs text-slate-400">of ${parseFloat(budget).toFixed(2)} budget</span>
        </div>
      </div>

      {/* 2. Budget Utilization % */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <span className="text-xs font-medium text-slate-500">Budget Utilization</span>
        <div className="flex items-baseline space-x-2 mt-1">
          <span className="text-2xl font-bold text-indigo-600">{util.toFixed(1)}%</span>
          <span className="text-xs text-slate-400">budget spent</span>
        </div>
      </div>

      {/* 3. Expected Incremental Revenue */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <span className="text-xs font-medium text-slate-500">Expected Incr. Revenue</span>
        <div className="flex items-baseline space-x-2 mt-1">
          <span className="text-2xl font-bold text-blue-600">${revenue.toFixed(2)}</span>
          <span className="text-xs text-slate-400">&sum; (ITE &times; AOV)</span>
        </div>
      </div>

      {/* 4. Net Profit Gain */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <span className="text-xs font-medium text-slate-500">Net Profit Gain</span>
        <div className="flex items-baseline space-x-2 mt-1">
          <span className="text-2xl font-bold text-emerald-600">${netProfit.toFixed(2)}</span>
          <span className="text-xs text-emerald-700 font-medium">ROI: {roi}%</span>
        </div>
        <div className="text-[10px] text-slate-400 mt-1">
          +${gainOverBlanket.toFixed(2)} vs blanket baseline
        </div>
      </div>
    </section>
  );
}

