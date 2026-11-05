import React from 'react';

export default function ExecutiveInsight({ summaryText, bulletInsights = [] }) {
  return (
    <section className="bg-gradient-to-r from-indigo-50 via-blue-50 to-emerald-50 rounded-xl border border-indigo-100 p-4 shadow-sm flex items-start space-x-3">
      <div className="p-2 bg-indigo-600 text-white rounded-lg shadow-sm mt-0.5">
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
        </svg>
      </div>
      <div className="flex-1 space-y-1">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-700">Executive Insight</span>
          <span className="text-[11px] text-slate-400">&bull; Causal Prescription vs Blanket Targeting</span>
        </div>
        <p className="text-sm font-semibold text-slate-800">
          {summaryText || "Causal optimization prunes non-incremental customers to maximize promotional ROI."}
        </p>
        {bulletInsights.length > 0 && (
          <ul className="text-xs text-slate-600 list-disc list-inside space-y-0.5 pt-1">
            {bulletInsights.map((insight, idx) => (
              <li key={idx}>{insight}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

