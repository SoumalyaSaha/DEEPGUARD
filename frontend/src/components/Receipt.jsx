import React, { useMemo, useState } from 'react'
import { sendFeedback } from '../api.js'

function seededBars(seed, count=48) {
  // deterministic visual barcode from verification_id
  let h=0
  for(let i=0;i<seed.length;i++) h = (h*31 + seed.charCodeAt(i)) >>>0
  const bars=[]
  for(let i=0;i<count;i++){
    h = (h*1664525 + 1013904223) >>>0
    const thin = (h % 3)===0
    const height = 12 + (h % 32)
    bars.push({ thin, height })
  }
  return bars
}

export default function Receipt({ data, fileMeta }) {
  const [fbStatus, setFbStatus] = useState(null)
  const [fbLoading, setFbLoading] = useState(false)

  if (!data) {
    return (
      <div className="receipt">
        <div className="receipt-header"><span>Forensic Audit Thermal Receipt</span><span style={{color:'#7a8e9e'}}>—</span></div>
        <div className="receipt-body muted small">No verification yet.</div>
      </div>
    )
  }

  // data can be either /api/detect POST response (has receipt, verification_id, model_breakdown, thumbnail_url)
  // or /api/detect/{id} GET response (full audit record with sha256, dimensions, etc)
  const isFull = !!data.sha256
  const verification_id = data.verification_id
  const feedback_token = data.feedback_token
  const timestamp_utc = data.timestamp_utc || new Date().toISOString()
  const filename = data.filename || fileMeta?.name || '—'
  const sha256 = data.sha256 || '—'
  const dimensions = data.dimensions || fileMeta?.dimensions || '—'
  const file_type = data.file_type || (filename.split('.').pop()||'').toUpperCase()
  const verdict = data.verdict || data.receipt?.verdict || '—'
  const confidence = data.confidence ?? data.receipt?.confidence ?? 0
  const breakdown = data.model_breakdown || []
  const thumbnail_url = data.thumbnail_url || null

  const bars = useMemo(()=> seededBars(verification_id || 'VA-XXXX-XXXX'), [verification_id])
  const isSynthetic = String(verdict).includes('SYNTHETIC')

  const handleFeedback = async (user_verdict) => {
    if (!feedback_token) { setFbStatus({type:'error', msg:'No feedback_token — re-upload to attest.'}); return }
    setFbLoading(true); setFbStatus(null)
    try{
      await sendFeedback(feedback_token, user_verdict)
      setFbStatus({type:'success', msg: user_verdict==='correct' ? 'Attested as ground truth — logged.' : 'Dispute logged — marked as contested.'})
    }catch(e){ setFbStatus({type:'error', msg:e.message}) }
    finally{ setFbLoading(false) }
  }

  const handleDownloadCSV = () => {
    const rows = [
      ['verification_id', verification_id],
      ['timestamp_utc', timestamp_utc],
      ['filename', filename],
      ['sha256', sha256],
      ['dimensions', dimensions],
      ['file_type', file_type],
      ['verdict', verdict],
      ['confidence', confidence],
      ['confidence_label', data.receipt?.confidence_label || ''],
      ['models_responded', data.receipt?.models_responded || breakdown.filter(b=>b.vote!=='error').length],
      ['models_total', data.receipt?.models_total || breakdown.length],
      ['feedback_token', feedback_token || ''],
      ['thumbnail_url', thumbnail_url || ''],
    ]
    breakdown.forEach(b=>{
      rows.push([`model_${b.model}_score`, b.score ?? ''])
      rows.push([`model_${b.model}_vote`, b.vote])
      rows.push([`model_${b.model}_internal`, b.internal_name])
      rows.push([`model_${b.model}_display_type`, b.display_type])
      if (b.threshold!=null) rows.push([`model_${b.model}_threshold`, b.threshold])
    })
    const csv = rows.map(r=> r.map(v=> `"${String(v).replaceAll('"','""')}"`).join(',')).join('\n')
    const blob = new Blob([csv], {type:'text/csv'})
    const url = URL.createObjectURL(blob)
    const a=document.createElement('a'); a.href=url; a.download=`${verification_id}_audit.csv`; a.click(); URL.revokeObjectURL(url)
  }

  const handleShare = async () => {
    const url = `${window.location.origin}/verify/${verification_id}`
    try{ await navigator.clipboard.writeText(url); setFbStatus({type:'success', msg:`Share link copied: ${url}`}) }
    catch{ setFbStatus({type:'error', msg:url}) }
  }

  return (
    <div className="receipt">
      <div className="receipt-header">
        <span>Forensic Audit Thermal Receipt</span>
        <span style={{color:'#7a8e9e',fontSize:10}}>{timestamp_utc}</span>
      </div>
      <div className="receipt-body">
        {thumbnail_url && <img src={thumbnail_url} alt="thumb" style={{width:'100%',height:120,objectFit:'cover',borderRadius:6,border:'1px solid rgba(0,229,204,0.15)',marginBottom:10}} onError={e=>e.currentTarget.style.display='none'} />}
        <div className={`verdict-line`} style={{borderColor: isSynthetic? 'rgba(255,59,74,0.3)':'rgba(0,255,136,0.3)', background: isSynthetic? 'rgba(255,59,74,0.06)':'rgba(0,255,136,0.06)'}}>
          <span className={`verdict-big ${isSynthetic?'verdict-synth':'verdict-auth'}`}>{verdict}</span>
          <span style={{fontSize:12,color:'#a8c0b8'}}>{(confidence*100).toFixed(2)}%</span>
        </div>

        <div className="kv"><span>verification_id</span><span style={{fontFamily:'Share Tech Mono'}}>{verification_id}</span></div>
        <div className="kv"><span>timestamp_utc</span><span>{timestamp_utc}</span></div>
        <div className="kv"><span>filename</span><span>{filename}</span></div>
        <div className="kv"><span>sha256</span><span style={{fontSize:10}}>{sha256}</span></div>
        <div className="kv"><span>dimensions</span><span>{dimensions}</span></div>
        <div className="kv"><span>file_type</span><span>{file_type}</span></div>
        <div className="kv"><span>confidence</span><span>{(confidence*100).toFixed(2)}% — {data.receipt?.confidence_label || `${breakdown.filter(b=>b.vote==='fake').length} flagged`}</span></div>
        <div className="kv"><span>feedback_token</span><span style={{fontSize:9,opacity:0.7}}>{feedback_token ? `${feedback_token.slice(0,8)}…` : '—'}</span></div>

        <div className="weights">
          <div style={{color:'#00e5cc',letterSpacing:'0.06em',marginBottom:4}}>MODEL WEIGHTS</div>
          <div>npr:1.00 · ufd:1.00 · iapl:0.844 · sdxl:1.00 · umm:1.00 · capcheck:1.00 · nonescape:1.00</div>
          {breakdown.map((b,i)=>(
            <div key={i} style={{display:'flex',justifyContent:'space-between'}}>
              <span>{b.model} ({b.internal_name})</span><span>{b.score!=null? (b.score*100).toFixed(2)+'%':'—'} · {b.vote}</span>
            </div>
          ))}
        </div>

        <div style={{marginTop:10}}>
          <div style={{fontSize:9,letterSpacing:'0.08em',color:'#7a8e9e'}}>VERIFICATION BARCODE</div>
          <div className="barcode">
            {bars.map((b,i)=> <div key={i} className={`bar ${b.thin?'thin':'thick'}`} style={{height: b.height}} />)}
          </div>
          <div style={{textAlign:'center',fontFamily:'Share Tech Mono',fontSize:10,letterSpacing:'0.12em',marginTop:4,color:'#a8c0b8'}}>{verification_id}</div>
        </div>

        <div style={{marginTop:14, display:'flex',gap:8,flexWrap:'wrap'}}>
          <button className="btn btn-green" disabled={fbLoading} onClick={()=>handleFeedback('correct')}>YES GROUND TRUTH</button>
          <button className="btn btn-red" disabled={fbLoading} onClick={()=>handleFeedback('incorrect')}>NO DISPUTED</button>
          <button className="btn btn-ghost" onClick={handleDownloadCSV}>DOWNLOAD CSV AUDIT</button>
          <button className="btn btn-ghost" onClick={handleShare}>SHARE LINK</button>
        </div>
        {fbStatus && <div className={fbStatus.type==='error'?'error-box':'success-box'}>{fbStatus.msg}</div>}
      </div>
    </div>
  )
}
