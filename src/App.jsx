import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''

const MODALITIES = [
  { id: 'optical', label: 'Optical', detail: 'Sentinel-2 / RGB' },
  { id: 'sar', label: 'SAR', detail: 'Radar backscatter' },
]

const EXAMPLES = [
  'What is visible in this satellite image?',
  'What new buildings appeared between these two images?',
  'Use optical and SAR images together to identify water-covered regions.',
]

function App() {
  const [files, setFiles] = useState([])
  const [modalities, setModalities] = useState(['optical'])
  const [query, setQuery] = useState(EXAMPLES[0])
  const [response, setResponse] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  const previews = useMemo(
    () => files.map((file) => ({ file, url: URL.createObjectURL(file) })),
    [files],
  )

  useEffect(() => () => previews.forEach(({ url }) => URL.revokeObjectURL(url)), [previews])

  function acceptFiles(incoming) {
    const valid = [...incoming].filter((file) => file.type.startsWith('image/') || /\.(tif|tiff)$/i.test(file.name))
    setFiles(valid)
    setError(valid.length < incoming.length ? 'Only PNG, JPEG, and TIFF images are supported.' : '')
  }

  async function analyze() {
    if (!query.trim()) {
      setError('Enter a natural-language query to continue.')
      return
    }
    setLoading(true)
    setError('')
    setResponse(null)
    const form = new FormData()
    form.append('query', query.trim())
    form.append('modalities', modalities.join(','))
    files.forEach((file) => form.append('images', file))
    try {
      const result = await fetch(`${API_URL}/analyze`, { method: 'POST', body: form })
      const raw = await result.text()
      let payload = {}
      try {
        payload = raw ? JSON.parse(raw) : {}
      } catch {
        throw new Error('The backend returned an unreadable response.')
      }
      if (!result.ok) throw new Error(payload.detail || 'Analysis request failed.')
      setResponse(payload)
    } catch (requestError) {
      setError(requestError.message || 'Could not connect to the BHUVISION backend.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <Header />
      <div className="workspace-grid">
        <section className="left-column">
          <div className="section-kicker"><span>01</span> REQUEST BUILDER</div>
          <section className="request-card surface">
            <div className="card-intro">
              <div><p className="label">Mission input</p><h2>Give the agent a scene to inspect.</h2></div>
              <span className="live-badge"><i /> LOCAL</span>
            </div>
            <UploadArea files={files} previews={previews} dragActive={dragActive} onFiles={acceptFiles} setDragActive={setDragActive} />
            {files.length > 0 && <div className="file-strip">{files.map((file) => <span key={`${file.name}-${file.size}`}>{file.name}</span>)}</div>}

            <div className="field-block">
              <div className="field-header"><label htmlFor="modalities">Modalities</label><span>request context</span></div>
              <div className="modality-grid" id="modalities">
                {MODALITIES.map((modality) => <button type="button" className={`modality-chip ${modalities.includes(modality.id) ? 'selected' : ''}`} key={modality.id} onClick={() => setModalities((current) => current.includes(modality.id) ? current.filter((item) => item !== modality.id) : [...current, modality.id])}><span className={`modality-icon ${modality.id}`}>{modality.id === 'sar' ? '⌁' : '◌'}</span><span><strong>{modality.label}</strong><small>{modality.detail}</small></span><b>{modalities.includes(modality.id) ? '✓' : '+'}</b></button>)}
              </div>
              <p className="field-note">Selection is passed as request context; the backend reports actual capability availability.</p>
            </div>

            <div className="field-block query-block">
              <div className="field-header"><label htmlFor="query">Natural-language query</label><span>agent instruction</span></div>
              <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} rows="4" placeholder="Ask about the uploaded imagery..." />
              <div className="example-row"><span>Try an example</span>{EXAMPLES.map((example) => <button type="button" key={example} onClick={() => setQuery(example)}>{shortExample(example)}</button>)}</div>
            </div>
            <button className="analyze-button" onClick={analyze} disabled={loading}><span className={loading ? 'button-spinner' : 'button-mark'}>{loading ? '' : '↗'}</span>{loading ? 'Building trace…' : 'Analyze scene'}<span className="button-arrow">→</span></button>
            {error && <div className="alert error-alert"><span>!</span>{error}</div>}
          </section>
          <p className="privacy-note"><span>⌁</span> Local prototype · No model outputs are invented when a capability is unavailable.</p>
        </section>

        <section className="right-column">
          <div className="section-kicker"><span>02</span> INTELLIGENCE OUTPUT</div>
          {loading ? <LoadingState /> : response ? <Result response={response} previews={previews} /> : <WelcomeState />}
        </section>
      </div>
    </main>
  )
}

function Header() {
  return <header className="topbar"><div className="brand"><span className="brand-mark"><i /><i /><i /></span><span><strong>BHUVISION</strong><small>Agentic Remote-Sensing Intelligence</small></span></div><div className="topbar-meta"><span className="topbar-dot" /> DEMO ENVIRONMENT <span className="version">v0.1</span></div></header>
}

function UploadArea({ files, previews, dragActive, onFiles, setDragActive }) {
  function drop(event) {
    event.preventDefault()
    setDragActive(false)
    onFiles(event.dataTransfer.files)
  }
  return <label className={`upload-area ${dragActive ? 'drag-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragActive(true) }} onDragLeave={() => setDragActive(false)} onDrop={drop}>
    <input type="file" multiple accept="image/png,image/jpeg,image/tiff,.tif,.tiff" onChange={(event) => onFiles(event.target.files)} />
    {files.length ? <div className="preview-grid">{previews.slice(0, 4).map(({ file, url }) => <div className="preview-tile" key={`${file.name}-${file.size}`}><img src={url} alt={file.name} /><span>{file.name}</span></div>)}{files.length > 4 && <div className="more-tile">+{files.length - 4}</div>}</div> : <><span className="upload-glyph">＋</span><strong>Drop satellite imagery here</strong><span>or browse from your device</span><small>PNG · JPEG · TIFF</small></>}
  </label>
}

function WelcomeState() {
  return <section className="welcome-card surface"><div className="radar-graphic"><span /><span /><span /><b>◎</b></div><p className="label">Awaiting mission input</p><h2>Turn imagery into an auditable lead.</h2><p className="welcome-copy">The agent will show what it can determine, what it cannot, and the evidence behind every step.</p><div className="welcome-path"><span>QUERY</span><i>→</i><span>PLAN</span><i>→</i><span>EVIDENCE</span></div></section>
}

function LoadingState() {
  return <section className="loading-card surface"><div className="loading-orbit"><span /><span /><span /></div><p className="label">Processing request</p><h2>Following the analysis trace…</h2><p>Planning the query and checking available capabilities.</p><div className="loading-lines"><i /><i /><i /></div></section>
}

function Result({ response, previews }) {
  const plan = response.plan || {}
  const routing = response.routing || {}
  const result = response.result || {}
  const audit = response.audit_trace || {}
  const confidence = response.confidence_report || { signals: [], aggregate_value: null, notes: [] }
  const unavailable = routing.requested_capability && !routing.available
  const semanticAnswer = false
  const images = result.image_analysis || []
  return <div className="result-stack">
    <section className="answer-card surface">
      <div className="answer-top"><p className="label">Final answer</p><span className={`outcome-badge ${unavailable ? 'unavailable' : 'observed'}`}><i />{unavailable ? 'CAPABILITY UNAVAILABLE' : 'NO SEMANTIC ANSWER'}</span></div>
      <h2>{semanticAnswer ? 'Semantic finding available.' : 'No semantic finding was produced.'}</h2>
      <p className="answer-explanation">{unavailable ? routing.reason : result.summary || 'The current pipeline returned no semantic interpretation.'}</p>
      <div className="answer-foot"><span>Honest output boundary</span><strong>{semanticAnswer ? 'Model-backed result' : 'Metadata and trace only'}</strong></div>
    </section>

    <div className="summary-grid">
      <InfoCard label="Analysis type" value={formatLabel(plan.task || 'unknown')} sub={`${plan.image_count || 0} image${plan.image_count === 1 ? '' : 's'}`} icon="◈" />
      <InfoCard label="Selected capability" value={routing.requested_capability || 'Deterministic only'} sub={routing.available ? 'available' : 'unavailable'} icon="⌁" tone={unavailable ? 'amber' : 'green'} />
    </div>

    <ConfidenceCard report={confidence} />
    <EvidenceCard response={response} images={images} previews={previews} />

    <details className="trace-card surface"><summary><span><b>04</b><strong>Analysis trace</strong></span><em>Expand record <i>⌄</i></em></summary><div className="trace-content"><TraceRow label="Execution" value={formatLabel(audit.execution_status || routing.execution || 'not recorded')} /><TraceRow label="Plan" value={`${formatLabel(plan.task || 'unknown')} · ${plan.modalities?.join(' + ') || 'modalities not declared'}`} /><TraceRow label="Requested capability" value={routing.requested_capability || 'none'} /><TraceRow label="Analysis ID" value={audit.analysis_id || 'not recorded'} /><details className="json-trace"><summary>View serialized trace</summary><pre>{JSON.stringify({ plan, routing, audit, confidence }, null, 2)}</pre></details></div></details>
  </div>
}

function InfoCard({ label, value, sub, icon, tone = 'green' }) { return <section className="info-card surface"><div className={`info-icon ${tone}`}>{icon}</div><div><p className="label">{label}</p><strong>{value}</strong><small className={tone === 'amber' ? 'amber-text' : ''}>{sub}</small></div></section> }

function ConfidenceCard({ report }) {
  const scalar = report.aggregate_value
  return <section className="confidence-card surface"><div className="card-title-row"><div><p className="label">Confidence / reliability</p><h3>{scalar === null || scalar === undefined ? 'Structured signals, no aggregate' : `${Math.round(scalar * 100)} / 100 derived`}</h3></div><span className="reliability-mark">◎</span></div><div className="signal-list">{(report.signals || []).map((signal) => <div className="signal-row" key={signal.name}><span className={`signal-status ${signal.status}`} /><div><strong>{formatLabel(signal.name)}</strong><small>{signal.explanation}</small></div><b>{signal.value === null || signal.value === undefined ? (signal.status === 'available' ? 'metrics only' : 'unavailable') : `${Math.round(signal.value * 100)}%`}</b></div>)}</div><p className="confidence-note">{(report.notes || []).join(' ') || 'No confidence summary was returned.'}</p></section>
}

function EvidenceCard({ response, images, previews }) {
  const evidence = response.evidence || []
  return <section className="evidence-card surface"><div className="card-title-row"><div><p className="label">Visual evidence</p><h3>What the system actually observed</h3></div><span className="evidence-count">{evidence.length} item{evidence.length === 1 ? '' : 's'}</span></div><div className="evidence-layout"><div className="evidence-preview-strip">{previews.length ? previews.map(({ file, url }) => <img key={`${file.name}-${file.size}`} src={url} alt={file.name} />) : <div className="no-preview">No image preview</div>}</div><div className="evidence-details">{images.length ? images.map((image) => <div className="evidence-item" key={image.filename}><span className="evidence-icon">▧</span><div><strong>{image.filename}</strong><small>{image.format} · {image.width} × {image.height} · {image.channels} channel{image.channels === 1 ? '' : 's'}</small></div><em>{image.quality?.warnings?.length ? image.quality.warnings.join(', ') : 'quality checks passed'}</em></div>) : <p className="empty-copy">No supported image metadata was returned.</p>}</div></div><p className="evidence-disclaimer">Visual evidence is limited to uploaded-image metadata in this prototype. No semantic overlay or object label is being claimed.</p></section>
}

function TraceRow({ label, value }) { return <div className="trace-row"><span>{label}</span><strong>{value}</strong></div> }
function shortExample(example) { return example.length > 34 ? `${example.slice(0, 31)}…` : example }
function formatLabel(value) { return String(value).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) }

export default App
