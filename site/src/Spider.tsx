// Kinds of good once overall goodness is removed, as a spider web: absolute (vs an even split) or relative (vs DeepSeek). Hover a model to highlight it. -- PI/OpenAI
import { useState } from 'react'
import { COLORS, FULL, meta } from './AlignmentChart'

type Row = { vs_deepseek: number[]; lo: number[]; hi: number[] }
export type SpiderData = { groups: string[]; sources: Record<'machiavelli' | 'daily_dilemmas', { deepseek_absolute: number[]; models: Record<string, Row> }> }

const INK = '#2f2e2a'
const pct = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(Math.round(100 * Math.expm1(v)))}%`

function Panel({ data, src, title, absolute, hover, setHover, rIn, rOut }:
  { data: SpiderData; src: 'machiavelli' | 'daily_dilemmas'; title: string; absolute: boolean; hover: string | null; setHover: (m: string | null) => void; rIn: number; rOut: number }) {
  const W = 520, cx = W / 2, cy = W / 2 + 10, R = 200, n = data.groups.length
  const s = data.sources[src]
  const rad = (v: number) => (v + rIn) / (rIn + rOut) * R  // the hub is the lowest value shown
  const pt = (k: number, v: number) => { const a = -Math.PI / 2 + 2 * Math.PI * k / n; return [cx + rad(v) * Math.cos(a), cy + rad(v) * Math.sin(a)] }
  const poly = (vals: number[]) => vals.map((v, k) => pt(k, v).join(',')).join(' ')
  const ticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60].filter(q => -rIn < Math.log1p(q / 100) && Math.log1p(q / 100) < rOut)
  const series: [string, number[], Row | null][] = Object.entries(s.models).filter(([m]) => m !== 'DeepSeek repeat' || !absolute)
    .map(([m, r]) => [m, absolute ? r.vs_deepseek.map((v, k) => v + s.deepseek_absolute[k]) : r.vs_deepseek, r] as [string, number[], Row])
  if (absolute) series.push(['DeepSeek', s.deepseek_absolute, null])
  const ref = absolute ? 'an even split' : 'DeepSeek'
  return (
    <svg viewBox={`0 0 ${W} ${W + 30}`} width="50%" role="img" aria-label={title}>
      <text x={cx} y={18} fontSize={19} fill={INK} textAnchor="middle">{title}</text>
      {ticks.map(q => <polygon key={q} points={poly(Array(n).fill(Math.log1p(q / 100)))} fill="none" stroke={q === 0 ? INK : '#d6d3c9'} strokeDasharray={q === 0 ? '5 4' : ''} strokeWidth={q === 0 ? 1.2 : .8} />)}
      {ticks.map(q => { const [x, y] = pt(.5, Math.log1p(q / 100)); return <text key={`t${q}`} x={x + 3} y={y} fontSize={11} fill="#77746a">{q ? `${q > 0 ? '+' : ''}${q}%` : '0'}</text> })}
      {data.groups.map((g, k) => { const [x, y] = pt(k, rOut * 1.02); const [lx, ly] = pt(k, rOut + (rIn + rOut) * .13); return <g key={g}>
        <line x1={cx} y1={cy} x2={x} y2={y} stroke="#cfcbbf" /><text x={lx} y={ly + 5} fontSize={14} fill={INK} textAnchor="middle">{g}</text></g> })}
      <g transform={`translate(${cx},${cy}) scale(1.1)`} fill="#333" stroke="#333" strokeLinecap="round">  {/* a small spider at the hub */}
        {[-1, 1].map(sd => <g key={sd} fill="none" strokeWidth={1.4}>
          {[[-3, -8, -12, -16, -16], [-2, -11, -4, -18, 2], [0, -10, 3, -17, 10], [1, -7, 8, -12, 16]].map(([ay, kx, ky, fx, fy], i) =>
            <polyline key={i} points={`${sd * -2},${ay} ${sd * kx},${ky} ${sd * fx},${fy}`} />)}
          <line x1={sd * 1.5} y1={-6} x2={sd * 2.5} y2={-9} strokeWidth={1.8} /></g>)}
        <circle cx={0} cy={7} r={6.5} stroke="none" /><circle cx={0} cy={-2.5} r={3.5} stroke="none" />
        <circle cx={-1.2} cy={-3} r={.8} fill="#fff" stroke="none" /><circle cx={1.2} cy={-3} r={.8} fill="#fff" stroke="none" />
      </g>
      {series.map(([m, vals, r]) => {
        const col = m === 'DeepSeek repeat' || m === 'DeepSeek' ? COLORS.DeepSeek : COLORS[m]
        const dim = hover && hover !== m ? .15 : 1
        const name = m === 'DeepSeek repeat' ? 'DeepSeek repeat (its 2 answers compared)' : FULL[m]
        return (
          <g key={m} opacity={dim} onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)}>
            <title>{`${name}${META_OK(m) ? '\n' + meta(m) : ''}`}</title>
            <polygon points={poly(vals)} fill="none" stroke={col} strokeWidth={m === 'DeepSeek' ? 3 : hover === m ? 3 : 1.8} strokeDasharray={m === 'DeepSeek repeat' ? '2 3' : ''} />
            {vals.map((v, k) => {
              const sig = r && (m === 'DeepSeek repeat' ? false : r.lo[k] > 0 || r.hi[k] < 0)
              const [x, y] = pt(k, v)
              return <circle key={k} cx={x} cy={y} r={sig ? 4.5 : 3} fill={sig ? col : '#fff'} stroke={col} strokeWidth={1.4}>
                <title>{`${name}\n${data.groups[k]}: ${pct(v)} odds vs ${ref}` + (r ? `\nvs DeepSeek ${pct(r.vs_deepseek[k])} [${pct(r.lo[k])}, ${pct(r.hi[k])}] (95%)` : '')}</title></circle>
            })}
          </g>
        )
      })}
    </svg>
  )
}
const META_OK = (m: string) => m !== 'DeepSeek repeat'

export default function Spider({ data, absolute }: { data: SpiderData; absolute: boolean }) {
  const [hover, setHover] = useState<string | null>(null)
  const all = (['machiavelli', 'daily_dilemmas'] as const).flatMap(src => {
    const s = data.sources[src]
    return Object.entries(s.models).filter(([m]) => m !== 'DeepSeek repeat' || !absolute).flatMap(([, r]) => r.vs_deepseek.map((v, k) => v + (absolute ? s.deepseek_absolute[k] : 0)))
      .concat(absolute ? s.deepseek_absolute : [])
  })
  const rIn = 1.1 * Math.max(.05, -Math.min(...all)), rOut = 1.12 * Math.max(...all)
  const models = Object.keys(data.sources.machiavelli.models).filter(m => m !== 'DeepSeek repeat').concat(absolute ? ['DeepSeek'] : ['DeepSeek repeat'])
  return (
    <div>
      <div>
        <Panel data={data} src="machiavelli" title="Game scenarios" absolute={absolute} hover={hover} setHover={setHover} rIn={rIn} rOut={rOut} />
        <Panel data={data} src="daily_dilemmas" title="Everyday dilemmas" absolute={absolute} hover={hover} setHover={setHover} rIn={rIn} rOut={rOut} />
      </div>
      <div className="legend">{models.map(m => <span key={m} onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)}
        style={{ color: m.startsWith('DeepSeek') ? COLORS.DeepSeek : COLORS[m], fontWeight: 600, marginRight: 16, cursor: 'default' }}>{m.startsWith('DeepSeek') ? '— ' : '● '}{m}</span>)}</div>
    </div>
  )
}
