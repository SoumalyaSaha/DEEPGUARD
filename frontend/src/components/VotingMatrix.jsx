import React from 'react'

function badgeForVote(vote) {
  if (vote === 'fake') return { text: 'FLAGGED', cls: 'badge-flagged' }
  if (vote === 'real') return { text: 'PASS', cls: 'badge-pass' }
  return { text: 'PASS', cls: 'badge-error' } // error maps to PASS per spec
}

export default function VotingMatrix({ breakdown, receipt }) {
  if (!breakdown || breakdown.length===0) {
    return (
      <div className="panel">
        <div className="panel-header">7-Model Voting Matrix</div>
        <div className="panel-body muted small">No inference yet — upload an image to populate the ensemble.</div>
      </div>
    )
  }
  const isSynthetic = receipt?.verdict?.includes('SYNTHETIC')
  const bannerCls = isSynthetic ? 'synthetic' : (receipt?.verdict?.includes('AUTHENTIC') ? 'authentic' : 'incon')
  return (
    <div className="panel">
      <div className="panel-header">
        <span>7-Model Voting Matrix</span>
        <span style={{color:'#7a8e9e',fontSize:10}}>{breakdown.length} MODELS</span>
      </div>
      <div className="panel-body">
        <div className="matrix-grid">
          {breakdown.map((b,i)=>{
            const badge = badgeForVote(b.vote)
            const pct = b.score!=null ? Math.max(0, Math.min(100, b.score*100)) : 0
            const fillCls = b.vote==='fake' ? 'fake' : 'real'
            return (
              <div key={i} className="model-card">
                <div className="model-card-head">
                  <div style={{minWidth:0}}>
                    <div className="model-title" title={b.model}>{b.model}</div>
                    <div className="model-sub">{b.label}</div>
                    <div className="small muted" style={{fontSize:9}}>{b.internal_name}</div>
                  </div>
                  <span className={`badge ${badge.cls}`}>{badge.text}</span>
                </div>
                <div className="score-row">
                  <div className="score-bar">
                    <div className={`score-fill ${b.score==null ? '' : fillCls}`} style={{width: `${pct}%`, opacity: b.score==null?0.2:1}} />
                  </div>
                  <span className="score-num">{b.score!=null ? `${pct.toFixed(2)}%` : '—'}</span>
                </div>
                {b.display_type==='variance' ? (
                  <div className="threshold">variance · threshold {b.threshold ?? 0.5}</div>
                ) : (
                  <div className="threshold">probability</div>
                )}
              </div>
            )
          })}
        </div>
        {receipt && (
          <div className={`banner ${bannerCls}`}>
            <span><strong>{receipt.confidence_label}</strong> · {(receipt.confidence*100).toFixed(2)}% &nbsp; <span style={{opacity:0.7,fontSize:11}}>{receipt.verdict}</span></span>
            <span style={{fontSize:10,opacity:0.8}}>{receipt.models_responded}/{receipt.models_total} responded</span>
          </div>
        )}
      </div>
    </div>
  )
}
