import React from 'react'

export default function Topology({ isDetecting, breakdown }) {
  // 7 input nodes positions
  const nodes = [
    { x: 60, y: 30 },{ x: 60, y: 60 },{ x: 60, y: 90 },{ x: 60, y: 120 },{ x: 60, y: 150 },{ x: 60, y: 180 },{ x: 60, y: 210 }
  ]
  const cx = 200, cy = 120
  const outX = 340, outY = 120

  const voteColor = (v) => {
    if (v === 'fake') return '#ff3b4a'
    if (v === 'real') return '#00ff88'
    if (v === 'error') return '#ffb800'
    return 'rgba(0,229,204,0.4)'
  }

  return (
    <div className="topology-wrap">
      <div className="panel-header">
        <span>Neural Consensus Topology Viewport</span>
        <span style={{color: isDetecting? '#00e5cc' : '#7a8e9e', fontSize:10}}>{isDetecting ? '● SYNCHRONIZING' : '○ IDLE'}</span>
      </div>
      <svg className="topo-svg" viewBox="0 0 400 240" preserveAspectRatio="xMidYMid meet">
        <defs>
          <radialGradient id="gGlow" cx="50%" cy="50%">
            <stop offset="0%" stopColor="#00e5cc" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#00e5cc" stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="c" />
            <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* connections input -> center */}
        {nodes.map((n,i)=>{
          const vote = breakdown && breakdown[i] ? breakdown[i].vote : null
          const col = vote ? voteColor(vote) : 'rgba(0,229,204,0.25)'
          return (
            <g key={i}>
              <line x1={n.x+14} y1={n.y} x2={cx-18} y2={cy} stroke={col} strokeWidth={isDetecting?1.6:1} opacity={isDetecting?0.9:0.55} />
              {isDetecting && (
                <circle r="3" fill={col} filter="url(#glow)">
                  <animateMotion dur={`${1.2 + i*0.15}s`} repeatCount="indefinite" path={`M ${n.x+14} ${n.y} L ${cx-18} ${cy}`} />
                </circle>
              )}
            </g>
          )
        })}
        {/* center -> output */}
        <line x1={cx+18} y1={cy} x2={outX-14} y2={outY} stroke={isDetecting ? '#00e5cc' : 'rgba(0,229,204,0.35)'} strokeWidth={isDetecting?2:1.2} />
        {isDetecting && (
          <circle r="3.5" fill="#00e5cc" filter="url(#glow)">
            <animateMotion dur="1s" repeatCount="indefinite" path={`M ${cx+18} ${cy} L ${outX-14} ${outY}`} />
          </circle>
        )}

        {/* input nodes */}
        {nodes.map((n,i)=>{
          const vote = breakdown && breakdown[i] ? breakdown[i].vote : null
          const col = vote ? voteColor(vote) : 'rgba(0,229,204,0.6)'
          const label = breakdown && breakdown[i] ? breakdown[i].model : `N${i+1}`
          return (
            <g key={'node'+i}>
              <circle cx={n.x} cy={n.y} r="12" fill="#0f1e2a" stroke={col} strokeWidth="1.2" opacity={0.95} />
              {isDetecting && <circle cx={n.x} cy={n.y} r="16" fill="url(#gGlow)" opacity="0.18"><animate attributeName="opacity" values="0.18;0.35;0.18" dur={`${0.9+i*0.1}s`} repeatCount="indefinite"/></circle>}
              <text x={n.x+20} y={n.y+4} fontSize="7" fill={col} fontFamily="Share Tech Mono">{label}</text>
            </g>
          )
        })}

        {/* central consensus node */}
        <g>
          <circle cx={cx} cy={cy} r="18" fill="rgba(0,229,204,0.12)" stroke="#00e5cc" strokeWidth="1.4" />
          <circle cx={cx} cy={cy} r="22" fill="none" stroke="rgba(0,229,204,0.18)" strokeWidth="1" strokeDasharray="3 3">
            {isDetecting && <animateTransform attributeName="transform" type="rotate" from={`0 ${cx} ${cy}`} to={`360 ${cx} ${cy}`} dur="4s" repeatCount="indefinite"/>}
          </circle>
          <text x={cx} y={cy+4} textAnchor="middle" fontSize="7" fill="#00e5cc" fontFamily="Share Tech Mono">CORE</text>
          {isDetecting && <circle cx={cx} cy={cy} r="28" fill="url(#gGlow)" opacity="0.22"><animate attributeName="r" values="22;32;22" dur="1.4s" repeatCount="indefinite"/></circle>}
        </g>

        {/* output node */}
        <g>
          <rect x={outX-14} y={outY-14} width="28" height="28" rx="6" fill="#0f1e2a" stroke={ breakdown && breakdown.length ? (breakdown.filter(b=>b.vote==='fake').length > breakdown.length/2 ? '#ff3b4a' : '#00ff88') : 'rgba(0,229,204,0.6)'} strokeWidth="1.2" />
          <text x={outX} y={outY+4} textAnchor="middle" fontSize="8" fill="#dff6f0" fontFamily="JetBrains Mono">◈</text>
          <text x={outX+20} y={outY+4} fontSize="7" fill="#7a8e9e" fontFamily="Share Tech Mono">OUTPUT</text>
        </g>
      </svg>
      <div style={{padding:'6px 10px', display:'flex',gap:10,flexWrap:'wrap',borderTop:'1px solid rgba(0,229,204,0.12)',background:'rgba(255,255,255,0.02)'}}>
        <span style={{fontSize:9,color:'#00ff88'}}>● REAL</span>
        <span style={{fontSize:9,color:'#ff3b4a'}}>● SYNTHETIC</span>
        <span style={{fontSize:9,color:'#ffb800'}}>● PASS/ERROR</span>
        <span style={{marginLeft:'auto',fontSize:9,color:'#7a8e9e',letterSpacing:'0.06em'}}>7-MODEL ENSEMBLE</span>
      </div>
    </div>
  )
}
