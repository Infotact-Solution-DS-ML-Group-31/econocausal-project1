import React from 'react';

export default function Navbar({ isOnline }) {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-xl shadow-md">
            E
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 tracking-tight">EconoCausal</h1>
            <p className="text-xs text-slate-500 font-medium">Prescriptive Dynamic Pricing & Uplift Dashboard</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div
            className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-semibold border shadow-sm transition ${
              isOnline
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
              }`}
            ></span>
            <span>{isOnline ? 'Online: 127.0.0.1:8000' : 'Standalone Demo Mode'}</span>
          </div>

          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-medium text-indigo-600 hover:text-indigo-800 transition"
          >
            API Docs &rarr;
          </a>
        </div>
      </div>
    </header>
  );
}

