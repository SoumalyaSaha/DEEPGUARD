import React, { useEffect, useState, useCallback } from 'react'
import { getHistory, getDetection } from '../api.js'
import Receipt from './Receipt.jsx'

function formatTs(ts){
  try{
    const d = new Date(ts)
    return d.toLocaleString('en-GB', { hour12:false }) + ' UTC'
  }catch{ return ts }
}

export default function AuditLog(){
  const [filter,setFilter]=useState('all')
  const [search,setSearch]=useState('')
  const [debounced,setDebounced]=useState('')
  const [data,setData]=useState({ total_attested:0, total_filtered:0, results:[] })
  const [loading,setLoading]=useState(false)
  const [err,setErr]=useState(null)
  const [offset,setOffset]=useState(0)
  const limit=20
  const [inspect,setInspect]=useState(null) // full record
  const [inspectLoading,setInspectLoading]=useState(false)

  useEffect(()=>{
    const id=setTimeout(()=> setDebounced(search), 300)
    return ()=> clearTimeout(id)
  },[search])
  useEffect(()=> setOffset(0), [filter, debounced])

  const fetchPage = useCallback(async ()=>{
    setLoading(true); setErr(null)
    try{
      const res = await getHistory({ filter, search: debounced, limit, offset })
      setData(res)
    }catch(e){ setErr(e.message)}
    finally{ setLoading(false)}
  },[filter, debounced, offset])

  useEffect(()=>{ fetchPage() },[fetchPage])

  const handleInspect = async (verification_id)=>{
    setInspectLoading(true)
    try{
      const rec = await getDetection(verification_id)
      setInspect(rec)
    }catch(e){ alert(e.message)}
    finally{ setInspectLoading(false)}
  }

  const verdictBadge = (v)=>{
    if (String(v).includes('SYNTHETIC')) return <span className="badge-verdict badge-synth">SYNTHETIC</span>
    if (String(v).includes('AUTHENTIC')) return <span className="badge-verdict badge-auth">AUTHENTIC</span>
    return <span className="badge-verdict badge-incon">INCONCLUSIVE</span>
  }

  return (
    <div>
      <div className="panel" style={{marginBottom:16}}>
        <div className="panel-header">
          <span>Prediction Audit Log</span>
          <span style={{color:'#7a8e9e',fontSize:11}}>TOTAL ATTESTED: <strong style={{color:'#00e5cc'}}>{data.total_attested}</strong> &nbsp;|&nbsp; FILTERED: {data.total_filtered}</span>
        </div>
        <div className="panel-body" style={{display:'flex',gap:10,flexWrap:'wrap',alignItems:'center'}}>
          <div className="filters">
            {['all','synthetic','authentic','disputed'].map(f=>(
              <button key={f} className={`chip ${filter===f?'active':''}`} onClick={()=>setFilter(f)}>{f}</button>
            ))}
          </div>
          <div className="search">
            <span style={{color:'#7a8e9e'}}>⌕</span>
            <input placeholder="search filename / sha256 / verification_id" value={search} onChange={e=>setSearch(e.target.value)} />
          </div>
          {loading && <span className="small muted">loading…</span>}
        </div>
      </div>

      {err && <div className="error-box" style={{marginBottom:12}}>{err}</div>}

      <div className="panel">
        <div style={{overflowX:'auto'}}>
          <table className="history-table">
            <thead>
              <tr style={{textAlign:'left',fontSize:10,letterSpacing:'0.08em',color:'#7a8e9e',borderBottom:'1px solid rgba(0,229,204,0.15)'}}>
                <th style={{padding:'8px'}}>THUMB</th>
                <th style={{padding:'8px'}}>FILE</th>
                <th style={{padding:'8px'}}>ID</th>
                <th style={{padding:'8px'}}>TIMESTAMP</th>
                <th style={{padding:'8px'}}>VERDICT</th>
                <th style={{padding:'8px'}}>CONF</th>
                <th style={{padding:'8px'}}>FLAG</th>
                <th style={{padding:'8px'}}></th>
              </tr>
            </thead>
            <tbody>
              {data.results.map(r=>(
                <tr key={r.verification_id} className="history-row">
                  <td>
                    {r.thumbnail_url ? <img src={r.thumbnail_url} alt="" className="thumb" onError={e=>{e.currentTarget.style.display='none'; e.currentTarget.nextSibling.style.display='grid'}} /> : null}
                    <div className="thumb-ph" style={{display: r.thumbnail_url?'none':'grid'}}>NO IMG</div>
                  </td>
                  <td>
                    <div style={{fontWeight:600, color:'#dff6f0'}}>{r.filename || '—'}</div>
                    <div className="small muted">{r.file_type} · {r.dimensions}</div>
                  </td>
                  <td style={{fontFamily:'Share Tech Mono',fontSize:10,color:'#a8c0b8'}}>{r.verification_id}</td>
                  <td className="small muted">{formatTs(r.timestamp_utc)}</td>
                  <td>{verdictBadge(r.verdict)}</td>
                  <td style={{color:'#a8c0b8'}}>{(r.confidence*100).toFixed(2)}%</td>
                  <td>{r.disputed ? <span style={{color:'#ff6b76',fontSize:10}}>● DISPUTED</span> : <span className="small muted">—</span>}</td>
                  <td><button className="btn btn-ghost" style={{padding:'4px 10px',fontSize:10}} disabled={inspectLoading} onClick={()=>handleInspect(r.verification_id)}>INSPECT</button></td>
                </tr>
              ))}
              {data.results.length===0 && !loading && (
                <tr><td colSpan={8} style={{padding:20,textAlign:'center'}} className="muted small">No records match.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="panel-body" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div className="small muted">Showing {data.results.length} of {data.total_filtered} — offset {offset}</div>
          <div className="pagination">
            <button className="btn btn-ghost" disabled={offset===0} onClick={()=>setOffset(o=>Math.max(0,o-limit))}>‹ Prev</button>
            <button className="btn btn-ghost" disabled={offset+limit >= data.total_filtered} onClick={()=>setOffset(o=>o+limit)}>Next ›</button>
          </div>
        </div>
      </div>

      {inspect && (
        <div className="modal-overlay" onClick={()=>setInspect(null)}>
          <div className="modal" onClick={e=>e.stopPropagation()}>
            <div style={{padding:'10px 14px',display:'flex',justifyContent:'space-between',alignItems:'center',borderBottom:'1px solid rgba(0,229,204,0.15)'}}>
              <span style={{fontFamily:'Share Tech Mono',letterSpacing:'0.08em',color:'#00e5cc',fontSize:11}}>INSPECT — {inspect.verification_id}</span>
              <button className="btn btn-ghost" onClick={()=>setInspect(null)}>✕</button>
            </div>
            <div style={{padding:14}}>
              <Receipt data={inspect} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
