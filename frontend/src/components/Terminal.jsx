import React from 'react'

export default function Terminal({ logs }) {
  return (
    <div className="terminal">
      {logs.length===0 ? (
        <div className="term-dim">— awaiting ingestion — drop image to begin forensic trace.</div>
      ) : logs.map((l,i)=>(
        <div key={i} className="term-line">
          <span className="term-ts">[{l.ts}]</span>
          <span className={l.cls || ''}>{l.msg}</span>
        </div>
      ))}
      {logs.length>0 && logs[logs.length-1]?.live && <span className="blink" style={{color:'#00e5cc'}}>▌</span>}
    </div>
  )
}

export function buildLogsFromResponse(file, response) {
  const ts = () => new Date().toISOString().slice(11,19)
  const lines = []
  const push = (msg, cls='') => lines.push({ ts: ts(), msg, cls })
  // try to extract info from enriched detection if available via /api/detect/{id} shape
  const sha = response.sha256 || response.sha || '—'
  const dims = response.dimensions || '—'
  const filename = response.filename || file?.name || 'upload'
  const ftype = response.file_type || (filename.split('.').pop()||'').toUpperCase()
  const tUtc = response.timestamp_utc || new Date().toISOString()
  push(`ingest » ${filename}  [${ftype}]`, 'term-dim')
  if (sha && sha!=='—') push(`sha256 » ${String(sha).slice(0,16)}…${String(sha).slice(-8)}`, 'term-dim')
  if (dims) push(`dimensions » ${dims}`, 'term-dim')
  push(`dispatch » 7 models [NPR, UFD, IAPL, SDXL, DIFFUSION_RECON, CAPCHECK, NONESCAPE]`, 'term-dim')
  const breakdown = response.model_breakdown || []
  breakdown.forEach(b=>{
    const voteUpper = b.vote === 'fake' ? 'SYNTHETIC' : b.vote === 'real' ? 'AUTHENTIC' : 'PASS'
    const scoreStr = b.score!=null ? (b.display_type==='variance' ? `${(b.score).toFixed(4)} var (thr ${b.threshold ?? 0.5})` : `${(b.score*100).toFixed(2)}%`) : '—'
    const cls = b.vote==='fake' ? 'term-err' : b.vote==='real' ? 'term-ok' : 'term-warn'
    push(`${b.model} (${b.internal_name}) → ${voteUpper}  ${scoreStr}`, cls)
  })
  const receipt = response.receipt
  if (receipt) {
    const verdict = receipt.verdict || response.verdict || '—'
    push(`consensus » ${verdict} — ${receipt.confidence_label} · ${(receipt.confidence*100).toFixed(2)}%`, receipt.verdict?.includes('SYNTHETIC') ? 'term-err' : 'term-ok')
    push(`verification_id » ${response.verification_id}`, 'term-dim')
  }
  push(`audit » logged @ ${tUtc}`, 'term-dim')
  return lines
}

export function buildLogsDetecting(file) {
  const ts = () => new Date().toISOString().slice(11,19)
  return [
    { ts: ts(), msg: `ingest » ${file.name} — ${(file.size/1024).toFixed(1)} KB`, cls:'term-dim' },
    { ts: ts(), msg: `hashing » sha256 …`, cls:'term-dim', live:true },
    { ts: ts(), msg: `dispatch » 7 models — awaiting inference…`, cls:'term-dim', live:true },
  ]
}
