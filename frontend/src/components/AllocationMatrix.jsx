import React, { useState, useMemo } from 'react';

export default function AllocationMatrix({ customers = [], allocations = [] }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [activePersona, setActivePersona] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Enrich allocations with customer baseline profile
  const enrichedRows = useMemo(() => {
    const custMap = new Map(customers.map((c) => [c.customer_id, c]));
    return allocations.map((row) => {
      const cust = custMap.get(row.customer_id) || {};
      const ite = cust.predicted_ite !== undefined ? cust.predicted_ite : (row.predicted_ite || 0.0);
      const baseProb = cust.baseline_conversion_prob !== undefined ? cust.baseline_conversion_prob : 0.10;
      const val = cust.customer_value || 50.0;
      const cost = cust.treatment_cost || 10.0;

      let persona = 'LOST_CAUSES';
      if (ite < 0) {
        persona = 'SLEEPING_DOGS';
      } else if (ite > 0 && ite * val > cost) {
        persona = 'PERSUADABLES';
      } else if (baseProb >= 0.70) {
        persona = 'SURE_THINGS';
      }

      return {
        ...row,
        baseline_conversion_prob: baseProb,
        predicted_ite: ite,
        customer_value: val,
        persona,
      };
    });
  }, [customers, allocations]);

  // Quadrant counts
  const quadrantStats = useMemo(() => {
    let persuadables = 0;
    let sleepingDogs = 0;
    let sureThings = 0;
    let lostCauses = 0;
    const total = enrichedRows.length;

    enrichedRows.forEach((r) => {
      if (r.persona === 'PERSUADABLES') persuadables++;
      else if (r.persona === 'SLEEPING_DOGS') sleepingDogs++;
      else if (r.persona === 'SURE_THINGS') sureThings++;
      else lostCauses++;
    });

    const pct = (c) => (total > 0 ? ((c / total) * 100).toFixed(1) : '0.0');
    return {
      persuadables: { count: persuadables, pct: pct(persuadables) },
      sleepingDogs: { count: sleepingDogs, pct: pct(sleepingDogs) },
      sureThings: { count: sureThings, pct: pct(sureThings) },
      lostCauses: { count: lostCauses, pct: pct(lostCauses) },
    };
  }, [enrichedRows]);

  // Filtering
  const filtered = useMemo(() => {
    return enrichedRows.filter((r) => {
      const matchesSearch = r.customer_id.toLowerCase().includes(searchTerm.toLowerCase().trim());
      const matchesPersona = activePersona === 'ALL' || r.persona === activePersona;
      return matchesSearch && matchesPersona;
    });
  }, [enrichedRows, searchTerm, activePersona]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const displayedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, currentPage]);

  return (
    <div className="space-y-6">
      {/* Persona Quadrant Breakdown Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Persuadables */}
        <div
          onClick={() => { setActivePersona(activePersona === 'PERSUADABLES' ? 'ALL' : 'PERSUADABLES'); setCurrentPage(1); }}
          className={`cursor-pointer bg-white rounded-xl border-l-4 border-emerald-500 border-t border-r border-b p-4 shadow-sm transition hover:shadow ${activePersona === 'PERSUADABLES' ? 'ring-2 ring-emerald-400 bg-emerald-50/20' : 'border-slate-200'}`}
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-900 uppercase">Persuadables</h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800">Target</span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-extrabold text-emerald-700">{quadrantStats.persuadables.count}</span>
            <span className="text-xs text-slate-500">({quadrantStats.persuadables.pct}%)</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">ITE &gt; 0 & profitable; treatment drives conversion.</p>
        </div>

        {/* Sleeping Dogs */}
        <div
          onClick={() => { setActivePersona(activePersona === 'SLEEPING_DOGS' ? 'ALL' : 'SLEEPING_DOGS'); setCurrentPage(1); }}
          className={`cursor-pointer bg-white rounded-xl border-l-4 border-rose-500 border-t border-r border-b p-4 shadow-sm transition hover:shadow ${activePersona === 'SLEEPING_DOGS' ? 'ring-2 ring-rose-400 bg-rose-50/20' : 'border-slate-200'}`}
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-900 uppercase">Sleeping Dogs</h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-100 text-rose-800">Pruned</span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-extrabold text-rose-700">{quadrantStats.sleepingDogs.count}</span>
            <span className="text-xs text-slate-500">({quadrantStats.sleepingDogs.pct}%)</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">ITE &lt; 0; offer triggers negative reaction/churn.</p>
        </div>

        {/* Sure Things */}
        <div
          onClick={() => { setActivePersona(activePersona === 'SURE_THINGS' ? 'ALL' : 'SURE_THINGS'); setCurrentPage(1); }}
          className={`cursor-pointer bg-white rounded-xl border-l-4 border-blue-500 border-t border-r border-b p-4 shadow-sm transition hover:shadow ${activePersona === 'SURE_THINGS' ? 'ring-2 ring-blue-400 bg-blue-50/20' : 'border-slate-200'}`}
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-900 uppercase">Sure Things</h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-100 text-blue-800">Organic</span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-extrabold text-blue-700">{quadrantStats.sureThings.count}</span>
            <span className="text-xs text-slate-500">({quadrantStats.sureThings.pct}%)</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">High baseline conversion; discounts waste margin.</p>
        </div>

        {/* Lost Causes */}
        <div
          onClick={() => { setActivePersona(activePersona === 'LOST_CAUSES' ? 'ALL' : 'LOST_CAUSES'); setCurrentPage(1); }}
          className={`cursor-pointer bg-white rounded-xl border-l-4 border-slate-400 border-t border-r border-b p-4 shadow-sm transition hover:shadow ${activePersona === 'LOST_CAUSES' ? 'ring-2 ring-slate-400 bg-slate-50' : 'border-slate-200'}`}
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-900 uppercase">Lost Causes</h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-700">Holdout</span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-2xl font-extrabold text-slate-600">{quadrantStats.lostCauses.count}</span>
            <span className="text-xs text-slate-500">({quadrantStats.lostCauses.pct}%)</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Low baseline; discounts will not motivate purchase.</p>
        </div>
      </section>

      {/* Allocation Matrix Table */}
      <section className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">Allocation Matrix</h3>
            <p className="text-xs text-slate-500">Prescriptive customer decisions generated by SciPy constrained optimization</p>
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="text"
              placeholder="Search customer ID..."
              value={searchTerm}
              onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
              className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 w-48"
            />
            <span className="text-xs text-slate-400 font-medium whitespace-nowrap">
              {filtered.length} records
            </span>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto border border-slate-200 rounded-lg table-scrollbar">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                <th className="py-2.5 px-3">Customer ID</th>
                <th className="py-2.5 px-3">Baseline Conversion %</th>
                <th className="py-2.5 px-3">Predicted ITE</th>
                <th className="py-2.5 px-3">Assigned Discount</th>
                <th className="py-2.5 px-3">Exp. Net Profit</th>
                <th className="py-2.5 px-3">Targeted Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium">
              {displayedRows.length === 0 ? (
                <tr>
                  <td colSpan="6" className="text-center py-8 text-slate-400">
                    No matching records found.
                  </td>
                </tr>
              ) : (
                displayedRows.map((row) => {
                  const iteVal = parseFloat(row.predicted_ite || 0);
                  const isPositive = iteVal >= 0;
                  return (
                    <tr key={row.customer_id} className="hover:bg-slate-50 transition border-b border-slate-100">
                      <td className="py-2.5 px-3 font-semibold text-slate-800">{row.customer_id}</td>
                      <td className="py-2.5 px-3 text-slate-600">{(row.baseline_conversion_prob * 100).toFixed(1)}%</td>
                      <td className={`py-2.5 px-3 font-semibold ${isPositive ? 'text-indigo-600' : 'text-rose-600'}`}>
                        {iteVal.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-medium text-slate-700">${parseFloat(row.assigned_discount || 0).toFixed(2)}</td>
                      <td className={`py-2.5 px-3 ${row.expected_net_profit >= 0 ? 'text-emerald-600 font-semibold' : 'text-slate-400'}`}>
                        ${parseFloat(row.expected_net_profit || 0).toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3">
                        {row.is_targeted ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 text-emerald-800">
                            Targeted
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-600">
                            Holdout
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between pt-2 text-xs">
          <span className="text-slate-500">
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex items-center space-x-1">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="px-2.5 py-1 rounded border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="px-2.5 py-1 rounded border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

