import React from 'react';

export default function ControlPanel({
  budget,
  setBudget,
  objective,
  setObjective,
  tierMode,
  setTierMode,
  onOptimize,
  onCompare,
  onQuick50,
  isLoading,
}) {
  return (
    <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Campaign Policy & Budget Controls</h2>
          <p className="text-xs text-slate-500">Configure constrained linear programming solvers and tier parameters</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={onQuick50}
            disabled={isLoading}
            className="px-3 py-2 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-300 transition shadow-sm disabled:opacity-50"
          >
            Quick Synthetic 50 Customers
          </button>
          <button
            onClick={onCompare}
            disabled={isLoading}
            className="px-3.5 py-2 text-xs font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 rounded-lg border border-amber-300 transition shadow-sm disabled:opacity-50"
          >
            Compare with Blanket Policy
          </button>
          <button
            onClick={onOptimize}
            disabled={isLoading}
            className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-sm hover:shadow transition flex items-center space-x-1.5 disabled:opacity-50"
          >
            <span>{isLoading ? 'Optimizing...' : 'Run Causal Optimization'}</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-1">
        {/* Budget Slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs">
            <label htmlFor="budget-slider" className="font-medium text-slate-700">Campaign Budget Limit ($B)</label>
            <span className="font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded text-sm">
              ${parseFloat(budget).toFixed(2)}
            </span>
          </div>
          <input
            id="budget-slider"
            type="range"
            min="10"
            max="2000"
            step="10"
            value={budget}
            onChange={(e) => setBudget(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>$10 min</span>
            <span>$500</span>
            <span>$1,000</span>
            <span>$2,000 max</span>
          </div>
        </div>

        {/* Objective Select */}
        <div className="space-y-1.5">
          <label htmlFor="objective-select" className="block text-xs font-medium text-slate-700">Optimization Objective</label>
          <select
            id="objective-select"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="maximize_profit">MAXIMIZE_PROFIT (Net Incremental Profit)</option>
            <option value="maximize_revenue">MAXIMIZE_REVENUE (Gross Incremental Revenue)</option>
            <option value="maximize_conversions">MAXIMIZE_CONVERSIONS (Pure Causal Conversions)</option>
          </select>
          <p className="text-[10px] text-slate-500">Prunes negative net profit users automatically under MAXIMIZE_PROFIT</p>
        </div>

        {/* Discount Tier Mode */}
        <div className="space-y-1.5">
          <label htmlFor="tier-select" className="block text-xs font-medium text-slate-700">Discount Tier Mode</label>
          <select
            id="tier-select"
            value={tierMode}
            onChange={(e) => setTierMode(e.target.value)}
            className="w-full text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="1d_binary">1D Binary Discount ($10 promotional tier)</option>
            <option value="2d_multitier">2D Multi-Tier MCKP ($10 & $20 discrete tiers)</option>
          </select>
          <p className="text-[10px] text-slate-500">Enforces mutual exclusivity &sum; x_{`{i,k}`} &le; 1 across discount tiers</p>
        </div>
      </div>
    </section>
  );
}

