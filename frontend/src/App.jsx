import { useEffect, useState, lazy, Suspense } from "react";
import axios from "axios";
import "./App.css";
import Papa from "papaparse";

const Plot = lazy(() => import("react-plotly.js"));
const API = "http://127.0.0.1:8000";

const examples = {
  uplift: {
    treatment: [1, 0, 1, 0, 1, 0, 1, 0],
    outcome: [1, 0, 1, 0, 1, 1, 0, 0],
    predicted_uplift: [0.8, 0.2, 0.7, 0.1, 0.6, 0.3, 0.5, 0.15],
  },
  allocation: {
    budget: 100,
    offers: [
      { customer_id: "C001", offer_id: "OFFER_001"
, cost: 20, expected_incremental_value: 45 },
      { customer_id: "C002", offer_id: "OFFER_002", cost: 30, expected_incremental_value: 60 },
      { customer_id: "C003", offer_id: "OFFER_003", cost: 15, expected_incremental_value: 25 },
      { customer_id: "C004", offer_id: "OFFER_004", cost: 25, expected_incremental_value: 55 },
    ],
  },
  drift: {
    reference_records: [
      { age: 25, segment: "A" },
      { age: 30, segment: "B" },
      { age: 35, segment: "A" },
      { age: 40, segment: "B" },
      { age: 45, segment: "A" },
      { age: 32, segment: "B" },
    ],
    current_records: [
      { age: 26, segment: "A" },
      { age: 31, segment: "A" },
      { age: 34, segment: "B" },
      { age: 65, segment: "C" },
      { age: 70, segment: "C" },
      { age: 68, segment: "C" },
    ],
    features: ["age", "segment"],
    bins: 10,
  },
};

function Status({ ok, children }) {
  return (
    <span className={`status ${ok ? "status-ok" : "status-error"}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

function JsonTool({ title, description, initialValue, onRun, result, loading, error }) {
  const [value, setValue] = useState(
    JSON.stringify(initialValue, null, 2)
  );
  const [parseError, setParseError] = useState("");

  async function submit() {
    try {
      const parsed = JSON.parse(value);
      setParseError("");
      await onRun(parsed);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setParseError("Invalid JSON. Check commas, quotes and brackets.");
      }
    }
  }

  return (
    <article className="tool-card">
      <div className="tool-heading">
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>

      <label className="editor-label">Request JSON</label>
      <textarea
        className="json-editor"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        spellCheck="false"
      />

      {parseError && <p className="tool-error">{parseError}</p>}
      {error && <p className="tool-error">{error}</p>}

      <button className="run-button" onClick={submit} disabled={loading}>
        {loading ? "Processing..." : "Run analysis"}
      </button>

      {result && (
        <div className="result-panel">
          <h4>API response</h4>
          {title === "Uplift evaluation" && <UpliftChart result={result} />}
          {title === "Budget allocation" && <AllocationSummary result={result} />}
          {title === "Drift monitoring" && <DriftSummary result={result} />}
          <details>
            <summary>View complete JSON response</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </div>
      )}
    </article>
  );
}


function UpliftChart({ result }) {
  if (!Array.isArray(result.curve)) {
    return <p>Uplift response received. No curve data was returned.</p>;
  }

  const n = result.n_observations || result.curve.length - 1;

  const x = result.curve.map((_, i) =>
    Number(((i / n) * 100).toFixed(1))
  );

  const traces = [
    {
      x,
      y: result.curve,
      type: "scatter",
      mode: "lines+markers",
      name: "Uplift ranking",
    },
  ];

  if (Array.isArray(result.random_curve)) {
    traces.push({
      x: result.random_curve.map((_, i) =>
        Number(((i / n) * 100).toFixed(1))
      ),
      y: result.random_curve,
      type: "scatter",
      mode: "lines",
      name: "Random baseline",
      line: { dash: "dash" },
    });
  }

  return (
    <>
      <div className="result-metrics">
        <div>
          <span>AUUC</span>
          <strong>{result.auuc ?? "—"}</strong>
        </div>
        <div>
          <span>Qini coefficient</span>
          <strong>{result.qini_coefficient ?? "—"}</strong>
        </div>
        <div>
          <span>Observations</span>
          <strong>{result.n_observations ?? "—"}</strong>
        </div>
        <div>
          <span>Treated</span>
          <strong>{result.n_treated ?? "—"}</strong>
        </div>
        <div>
          <span>Control</span>
          <strong>{result.n_control ?? "—"}</strong>
        </div>
      </div>
<Suspense fallback={<p className="result-note">Loading chart...</p>}>
  <Plot
    data={traces}
    layout={{
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cbd5e1" },
      margin: { l: 50, r: 15, t: 20, b: 55 },
      xaxis: {
        title: "Population reached (%)",
        range: [0, 100],
        gridcolor: "#28344b",
      },
      yaxis: {
        title: "Cumulative Qini",
        gridcolor: "#28344b",
      },
      legend: { orientation: "h", y: -0.3 },
    }}
    useResizeHandler
    style={{ width: "100%", height: "340px" }}
    config={{ responsive: true, displayModeBar: false }}
  />
</Suspense>

      <p className="result-note">
        AUUC: {result.auuc} · Random AUUC: {result.random_auuc}
      </p>
      <p className="result-note">
        This evaluates the supplied ranking. It does not train an uplift
        model or establish causal identification.
      </p>

      {Array.isArray(result.warnings) && result.warnings.length > 0 && (
        <details>
          <summary>Evaluation warnings ({result.warnings.length})</summary>
          <ul>
            {result.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

function AllocationSummary({ result }) {
  const allocations = result.allocations;

  return (
    <>
      <div className="result-metrics">
        <div><span>Total cost</span><strong>{result.total_cost ?? "—"}</strong></div>
       <div>
  <span>Expected incremental value</span>
  <strong>{result.total_expected_incremental_value ?? "—"}</strong>
</div>
        <div><span>Selected offers</span><strong>{result.selected_count ?? "—"}</strong></div>
      </div>

      {Array.isArray(allocations) && allocations.length > 0 ? (
        <div className="table-scroll">
          <table className="result-table">
            <thead>
              <tr>
                {Object.keys(allocations[0]).map((key) => (
                  <th key={key}>{key.replaceAll("_", " ")}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {allocations.map((item, i) => (
                <tr key={i}>
                  {Object.values(item).map((value, j) => (
                    <td key={j}>{String(value)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>Allocation response received. See the complete response below.</p>
      )}

      <p className="result-note">
        The optimizer uses supplied expected incremental values. It does not
        estimate treatment effects.
      </p>
    </>
  );
}

function DriftSummary({ result }) {
  return (
    <>
      <div className="result-metrics">
        <div><span>Overall status</span><strong>{result.status ?? "—"}</strong></div>
        <div><span>Features checked</span><strong>{result.features_checked ?? "—"}</strong></div>
        <div><span>Drifted features</span><strong>{result.drifted_features?.length ?? "—"}</strong></div>
      </div>

      {Array.isArray(result.results) && (
        <div className="table-scroll">
          <table className="result-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Type</th>
                <th>PSI</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((item, i) => (
                <tr key={i}>
                  <td>{item.feature}</td>
                  <td>{item.feature_type}</td>
                  <td>{item.psi}</td>
                  <td>{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="result-note">
        PSI is a distribution-change signal. It does not prove causal validity
        or model performance degradation.
      </p>
    </>
  );
}

function App() {
  const [health, setHealth] = useState("checking");
  const [readiness, setReadiness] = useState(null);
  const [error, setError] = useState("");
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState({});
  const [toolErrors, setToolErrors] = useState({});
  const [csvData, setCsvData] = useState([]);
const [csvFile, setCsvFile] = useState("");
const [csvError, setCsvError] = useState("");
const [csvValidation, setCsvValidation] = useState(null);
const [csvLoading, setCsvLoading] = useState(false);
const [largeFileResult, setLargeFileResult] = useState(null);
const [largeFileLoading, setLargeFileLoading] = useState(false);


async function handleCsvUpload(event) {
  const file = event.target.files?.[0];

  setCsvData([]);
  setCsvFile("");
  setCsvError("");
  setCsvValidation(null);
  setLargeFileResult(null);

  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".csv")) {
    setCsvError("Please upload a CSV file.");
    return;
  }

  setCsvFile(file.name);

  // Large files are inspected by FastAPI, not parsed in the browser.
  if (file.size > 10 * 1024 * 1024) {
    setLargeFileLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API}/data/inspect-csv`,
        formData
      );

      setLargeFileResult(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setCsvError(
        typeof detail === "string"
          ? detail
          : JSON.stringify(detail || err.message)
      );
      setCsvFile("");
    } finally {
      setLargeFileLoading(false);
    }

    return;
  }

  // Keep the existing small-file PapaParse workflow.
  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    complete: (result) => {
      if (result.errors.length > 0) {
        setCsvError(result.errors[0].message);
        setCsvFile("");
        return;
      }

      if (!result.meta.fields?.length || result.data.length === 0) {
        setCsvError("The CSV has no usable headers or data rows.");
        setCsvFile("");
        return;
      }

      setCsvData(result.data);
    },
    error: (error) => {
      setCsvError(error.message);
      setCsvFile("");
    },
  });
}


async function validateUploadedCsv() {
  if (!csvData.length) return;

  setCsvLoading(true);
  setCsvError("");
  setCsvValidation(null);

  try {
    // Limit the API validation sample to avoid sending excessively
    // large requests from the browser.
const sample = csvData.slice(0, 5000).map((row) => ({
  ...row,
  treatment:
    row.treatment?.trim() === "0" || row.treatment?.trim() === "1"
      ? Number(row.treatment.trim())
      : row.treatment,
  outcome:
    row.outcome?.trim() === "0" || row.outcome?.trim() === "1"
      ? Number(row.outcome.trim())
      : row.outcome,
}));
    const response = await axios.post(`${API}/causal/validate`, {
      records: sample,
    });

    setCsvValidation({
      ...response.data,
      sample_size: sample.length,
      total_rows: csvData.length,
    });
  } catch (err) {
    const detail = err.response?.data?.detail;
    setCsvError(
      typeof detail === "string"
        ? detail
        : JSON.stringify(detail || err.message)
    );
  } finally {
    setCsvLoading(false);
  }
}

const csvColumns = csvData.length ? Object.keys(csvData[0]) : [];

const missingSummary = csvColumns.map((column) => ({
  column,
  missing: csvData.filter(
    (row) =>
      row[column] === null ||
      row[column] === undefined ||
      String(row[column]).trim() === ""
  ).length,
}));

  useEffect(() => {
    async function loadBackendStatus() {
      try {
        const [readinessResponse] = await Promise.all([
          axios.get(`${API}/health`),
          axios.get(`${API}/causal/readiness`),
        ]);
        setHealth("connected");
        setReadiness(readinessResponse.data);
        setError("");
      } catch (err) {
        setHealth("disconnected");
        setError(err.response?.data?.detail || err.message);
      }
    }
    loadBackendStatus();
  }, []);

  async function runTool(key, endpoint, payload) {
    setLoading((old) => ({ ...old, [key]: true }));
    setToolErrors((old) => ({ ...old, [key]: "" }));
    setResults((old) => ({ ...old, [key]: null }));

    try {
      const response = await axios.post(`${API}${endpoint}`, payload);
      setResults((old) => ({ ...old, [key]: response.data }));
    } catch (err) {
      const detail = err.response?.data?.detail;
      setToolErrors((old) => ({
        ...old,
        [key]: typeof detail === "string"
          ? detail
          : JSON.stringify(detail || err.message),
      }));
    } finally {
      setLoading((old) => ({ ...old, [key]: false }));
    }
  }

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">EC</div>
          <div><h2>EconoCausal</h2><p>Pricing Intelligence</p></div>
        </div>

        <p className="nav-heading">WORKSPACE</p>
        <nav>
          <a className="nav-link active" href="#overview">Overview</a>
          <a className="nav-link" href="#uplift">Uplift Evaluation</a>
          <a className="nav-link" href="#allocation">Budget Allocation</a>
          <a className="nav-link" href="#monitoring">Drift Monitoring</a>
        </nav>

        <div className="sidebar-bottom">
          <p>PROJECT 01</p>
          <strong>Dynamic Pricing</strong>
          <span>Infotact Solutions</span>
        </div>
      </aside>

      <section className="main-content">
        <header className="topbar" id="overview">
          <div>
            <p className="eyebrow">ECONOMICS × MACHINE LEARNING</p>
            <h1>Dashboard Overview</h1>
          </div>
          <Status ok={health === "connected"}>
            {health === "connected" ? "API Connected" : health}
          </Status>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">DECISION INTELLIGENCE PLATFORM</p>
            <h2>Dynamic pricing, guided by causal evidence.</h2>
            <p>
              Evaluate campaign impact, inspect uplift rankings,
              optimize budgets, and monitor data distributions.
            </p>
          </div>
          <div className="welcome-symbol">EC<span>.</span></div>
        </section>

        {error && <div className="error-banner">Backend error: {error}</div>}

        <section className="section">
          <div className="section-heading">
            <div><p className="eyebrow">SYSTEM STATUS</p><h2>Platform health</h2></div>
          </div>

          <div className="metrics-grid">
            <article className="metric-card">
              <span className="metric-label">Backend API</span>
              <strong className="metric-value">{health === "connected" ? "Online" : health}</strong>
              <span className="metric-note">FastAPI service · Port 8000</span>
            </article>
            <article className="metric-card">
              <span className="metric-label">Causal readiness</span>
              <strong className="metric-value">{readiness?.status || "Loading..."}</strong>
              <span className="metric-note">Structural data readiness check</span>
            </article>
            <article className="metric-card">
              <span className="metric-label">Frontend</span>
              <strong className="metric-value">Online</strong>
              <span className="metric-note">React · Vite</span>
            </article>
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <div><p className="eyebrow">CAUSAL VALIDATION</p><h2>Data readiness assessment</h2></div>
            {readiness && <Status ok={readiness.status === "ready"}>{readiness.status}</Status>}
          </div>
          <article className="readiness-card">
            <h3>{readiness?.status === "ready" ? "Structural checks passed" : "Causal identification is not established"}</h3>
            <p>
              Historical retail transactions do not automatically establish
              campaign assignment, exposed non-purchasers, or causal identification.
              A structural readiness response is not proof of causality.
            </p>
            {readiness && (
              <details>
                <summary>View readiness response</summary>
                <pre>{JSON.stringify(readiness, null, 2)}</pre>
              </details>
            )}
          </article>
        </section>

        <section className="section" id="uplift">
          <div className="section-heading">
            <div><p className="eyebrow">CAUSAL ANALYTICS</p><h2>Uplift evaluation</h2></div>
          </div>
          <JsonTool
            title="Uplift evaluation"
            description="Submit treatment, outcome, and predicted uplift arrays to calculate Qini and AUUC."
            initialValue={examples.uplift}
            onRun={(data) => runTool("uplift", "/causal/uplift", data)}
            result={results.uplift}
            loading={loading.uplift}
            error={toolErrors.uplift}
          />
        </section>

        <section className="section" id="allocation">
          <div className="section-heading">
            <div><p className="eyebrow">OPTIMIZATION ENGINE</p><h2>Budget allocation</h2></div>
          </div>
          <JsonTool
            title="Budget allocation"
            description="Provide offer costs and expected incremental values to optimize selection under a budget."
            initialValue={examples.allocation}
            onRun={(data) => runTool("allocation", "/optimization/allocate", data)}
            result={results.allocation}
            loading={loading.allocation}
            error={toolErrors.allocation}
          />
        </section>

        <section className="section" id="monitoring">
          <div className="section-heading">
            <div><p className="eyebrow">MODEL MONITORING</p><h2>Data drift monitoring</h2></div>
          </div>
          <JsonTool
            title="Drift monitoring"
            description="Compare reference and current feature records using Population Stability Index."
            initialValue={examples.drift}
            onRun={(data) => runTool("drift", "/monitoring/drift", data)}
            result={results.drift}
            loading={loading.drift}
            error={toolErrors.drift}
          />
        </section>
<section className="section" id="campaign">
  <div className="section-heading">
    <div>
      <p className="eyebrow">DATA INGESTION</p>
      <h2>Campaign data upload</h2>
    </div>
  </div>

  <article className="tool-card">
    <h3>Upload your CSV dataset</h3>
    <p>
      Upload a customer-level CSV to inspect its structure and
      check whether it contains the required causal fields.
    </p>

    <label className="upload-box">
      <span>Choose CSV file</span>
      <input
        type="file"
        accept=".csv,text/csv"
        onChange={handleCsvUpload}
      />
    </label>

    {csvError && <p className="tool-error">{csvError}</p>}
    
{largeFileLoading && (
  <p className="result-note">
    Uploading and inspecting large CSV file. Please wait...
  </p>
)}

{largeFileResult && (
  <div className="result-panel">
    <h3>Large dataset inspection</h3>

    <div className="result-metrics">
      <div>
        <span>Total rows</span>
        <strong>
          {largeFileResult.total_rows.toLocaleString()}
        </strong>
      </div>
      <div>
        <span>Columns</span>
        <strong>{largeFileResult.total_columns}</strong>
      </div>
      <div>
        <span>Causal schema</span>
        <strong>
          {largeFileResult.causal_schema_present
            ? "Present"
            : "Incomplete"}
        </strong>
      </div>
    </div>

   <h4>Missing causal fields</h4>
<ul className="missing-fields-list">
  {largeFileResult.missing_causal_fields.map((field) => (
    <li key={field}>{field}</li>
  ))}
</ul>

    <h4>Missing-value summary</h4>
    <div className="table-scroll">
      <table className="result-table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Missing count</th>
            <th>Missing %</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(largeFileResult.missing_values).map(
            ([column, stats]) => (
              <tr key={column}>
                <td>{column}</td>
                <td>{stats.missing_count.toLocaleString()}</td>
                <td>{stats.missing_percentage}%</td>
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>

    <p className="result-note">
      {largeFileResult.warning}
    </p>
  </div>
)}


   {csvFile && csvData.length > 0 && (
      <>
        <p className="upload-success">
          Loaded: <strong>{csvFile}</strong>
        </p>

        <div className="result-metrics">
          <div>
            <span>Total rows</span>
            <strong>{csvData.length.toLocaleString()}</strong>
          </div>
          <div>
            <span>Columns</span>
            <strong>{csvColumns.length}</strong>
          </div>
          <div>
            <span>Validation sample</span>
            <strong>{Math.min(csvData.length, 5000).toLocaleString()}</strong>
          </div>
        </div>

        <h4>Column overview</h4>
        <div className="table-scroll">
          <table className="result-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Missing values</th>
                <th>Completeness</th>
              </tr>
            </thead>
            <tbody>
              {missingSummary.map((item) => (
                <tr key={item.column}>
                  <td>{item.column}</td>
                  <td>{item.missing}</td>
                  <td>
                    {(
                      ((csvData.length - item.missing) / csvData.length) *
                      100
                    ).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h4>Data preview — first 10 rows</h4>
        <div className="table-scroll">
          <table className="result-table">
            <thead>
              <tr>
                {csvColumns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {csvData.slice(0, 10).map((row, index) => (
                <tr key={index}>
                  {csvColumns.map((column) => (
                    <td key={column}>
                      {String(row[column] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <button
          className="run-button"
          onClick={validateUploadedCsv}
          disabled={csvLoading}
        >
          {csvLoading ? "Validating..." : "Validate causal data"}
        </button>
      </>
    )}

    {csvValidation && (
      <div className="result-panel">
        <h4>Causal data validation result</h4>

        <Status ok={csvValidation.valid}>
          {csvValidation.valid ? "Structural checks passed" : "Validation failed"}
        </Status>

        <div className="result-metrics">
          <div>
            <span>Rows checked</span>
            <strong>{csvValidation.rows_checked ?? csvValidation.sample_size}</strong>
          </div>
          <div>
            <span>Total uploaded rows</span>
            <strong>{csvValidation.total_rows}</strong>
          </div>
          <div>
            <span>Structural validity</span>
            <strong>{csvValidation.valid ? "Valid" : "Invalid"}</strong>
          </div>
        </div>

        {csvValidation.errors?.length > 0 && (
          <>
            <h4>Validation errors</h4>
            <ul className="validation-errors">
              {csvValidation.errors.map((item, index) => (
                <li key={index}>{String(item)}</li>
              ))}
            </ul>
          </>
        )}

        {csvValidation.warnings?.length > 0 && (
          <>
            <h4>Warnings</h4>
            <ul className="validation-warnings">
              {csvValidation.warnings.map((item, index) => (
                <li key={index}>{String(item)}</li>
              ))}
            </ul>
          </>
        )}

        <p className="result-note">
          Structural validation does not establish causal identification.
          This check uses only the first {csvValidation.sample_size} rows.
        </p>
      </div>
    )}
  </article>
</section>
        <footer>EconoCausal · Infotact Solutions · Project 01</footer>
      </section>
    </main>
  );
}

export default App;
