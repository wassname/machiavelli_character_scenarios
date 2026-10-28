// AD&D-style alignment graph (after the 1979 DMG, Appendix III): lawful left, good up, zero = a chooser that splits its % evenly.
// Zooms on the left (the clusters sit top-left), overview on the right. Region geometry from scripts/evaluation/plot_dnd_classic.py classic(). -- PI/OpenAI
import { useState } from 'react'

type Pair = [number, number]
export type ModelRow = { model: string; offset: Pair; absolute: Pair; ci95_refit: [Pair, Pair] }
export type SourceData = {
  deepseek_anchor: Pair  // [lawful, good] nats vs an even split
  models: Record<string, ModelRow>  // offsets vs DeepSeek, [lawful, good]
  key_model: string
  variance_shares: Record<'questions' | 'answers' | 'calibration', Pair>
}
export type DndData = { sources: Record<'daily_dilemmas' | 'machiavelli', SourceData> }

export const COLORS: Record<string, string> = { Muse: '#0467DF', Qwen: '#6950EF', Gemini: '#E37400', Luna: '#111111', Haiku: '#8C4A2F', DeepSeek: '#8a8a8a' }
const LOGOS: Record<string, string> = { Muse: 'meta', Qwen: 'qwen', Gemini: 'google', Luna: 'openai', Haiku: 'anthropic', DeepSeek: 'deepseek' }
const FULL: Record<string, string> = { Muse: 'Meta Muse Spark 1.3', Qwen: 'Qwen 3.8 27B', Gemini: 'Google Gemini 3.8 Flash', Luna: 'OpenAI GPT-5.6 Luna', Haiku: 'Anthropic Claude Haiku 5.5', DeepSeek: 'DeepSeek v4.1 Flash (reference)' }
const INK = '#4b4a43', FAINT = '#8f8b7c'
const f2 = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

// Layout: two zooms stacked on the left, overview on the right.
const Z = 400, ZX = 40, ZY = [50, 50 + Z + 130]
const M = 640, MX = ZX + Z + 90, MY = 70, EXT = 3
const S = M / (2 * EXT)  // px per nat in the overview
const sx = (lawful: number) => MX + M / 2 - lawful * S  // lawful on the left
const sy = (good: number) => MY + M / 2 - good * S
const W = MX + M + 40, H = ZY[1] + Z + 70

// One quadrant outline in display units (x = chaotic, y = good), mirrored to the other three.
const LOBE: Pair[] = [[-1, 3], [-.52, 2.25], [-.32, 1.05], [-.28, .78], [-.20, .44], [-.44, .20], [-.78, .28], [-1.05, .32], [-2.25, .52], [-3, 1]]
const lobePath = (mx: number, my: number) => {
  const p = LOBE.map(([x, y]) => [MX + M / 2 + x * mx * S, MY + M / 2 - y * my * S])
  return `M${p[0]} C${p[1]} ${p[2]} ${p[3]} C${p[4]} ${p[5]} ${p[6]} C${p[7]} ${p[8]} ${p[9]}`
}

type Box = [number, number, number, number]
const hit = (a: Box, b: Box) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
function placeLabels(pts: { name: string; x: number; y: number }[]) {
  // Greedy: first candidate spot that clears every marker and earlier label.
  const marks: Box[] = pts.map(p => [p.x - 13, p.y - 13, p.x + 13, p.y + 13])
  const taken: Box[] = []
  const cands: [number, number, 'start' | 'end' | 'middle'][] = [[17, 6, 'start'], [-17, 6, 'end'], [0, -19, 'middle'], [0, 31, 'middle'], [19, -14, 'start'], [-19, -14, 'end'], [19, 26, 'start'], [-19, 26, 'end'], [36, 6, 'start'], [-36, 6, 'end']]
  return pts.map(p => {
    const w = 10 * p.name.length
    for (const [dx, dy, a] of cands) {
      const x0 = a === 'start' ? p.x + dx : a === 'end' ? p.x + dx - w : p.x + dx - w / 2
      const b: Box = [x0, p.y + dy - 14, x0 + w, p.y + dy + 3]
      if (!marks.some(m => hit(b, m)) && !taken.some(t => hit(b, t))) { taken.push(b); return { ...p, tx: p.x + dx, ty: p.y + dy, a } }
    }
    throw new Error(`no label spot for ${p.name}`)
  })
}

function Zoom({ y, title, dashed, src }: { y: number; title: string; dashed: boolean; src: SourceData }) {
  // Offsets from DeepSeek, ±R nats; DeepSeek sits at the centre by definition. Hover a model to see its own 95% interval.
  const [hover, setHover] = useState<string | null>(null)
  const R = .2, x = ZX, w = Z, k = (w / 2 - 20) / R
  const ix = (lawful: number) => x + w / 2 - lawful * k
  const iy = (good: number) => y + w / 2 - good * k
  const rows: [string, Pair][] = [['DeepSeek', [0, 0]], ...Object.entries(src.models).map(([n, r]) => [n, r.offset] as [string, Pair])]
  const labels = placeLabels(rows.map(([n, [l, g]]) => ({ name: n, x: ix(l), y: iy(g) })))
  const key = src.models[src.key_model].ci95_refit
  const kw = (key[0][1] - key[0][0]) * k, kh = (key[1][1] - key[1][0]) * k
  const parts = (['questions', 'calibration', 'answers'] as const).map(n => [n, src.variance_shares[n][0]] as const)
  const shown = hover && hover !== 'DeepSeek' ? src.models[hover] : null
  let bx = ix(0) - kw / 2
  return (
    <g>
      <text x={x} y={y - 14} fontSize={20} fill={INK}>{title} <tspan fill={FAINT}>vs DeepSeek</tspan></text>
      <rect x={x} y={y} width={w} height={w} fill="#fffdf6" stroke={INK} strokeWidth={1} strokeDasharray={dashed ? '6 4' : ''} />
      <line x1={x} x2={x + w} y1={iy(0)} y2={iy(0)} stroke="#e1ded2" />
      <line y1={y} y2={y + w} x1={ix(0)} x2={ix(0)} stroke="#e1ded2" />
      <text x={x + 8} y={y + w - 8} fontSize={13} fill={FAINT}>← more lawful</text>
      <text x={x + w - 8} y={y + w - 8} fontSize={13} fill={FAINT} textAnchor="end">more chaotic →</text>
      <text x={x + w / 2 + 6} y={y + 18} fontSize={13} fill={FAINT}>more good ↑</text>
      {/* uncertainty key: one typical model's 95% interval drawn around DeepSeek as a scale (not a test); braces below = where its width comes from */}
      {(() => {
        const cx = ix(0), cy = iy(0)
        return <g>
          <g stroke="#b9b5a6" strokeWidth={6} strokeOpacity={.4}><line x1={cx - kw / 2} x2={cx + kw / 2} y1={cy} y2={cy} /><line x1={cx} x2={cx} y1={cy - kh / 2} y2={cy + kh / 2} /></g>
          <text x={cx} y={cy - kh / 2 - 6} fontSize={13} fill={FAINT} textAnchor="middle">uncertainty (95%)</text>
          {parts.map(([n, f]) => { const len = f * kw, g = <g key={n}>
            <path d={`M${bx + 1},${y + w + 10} v7 h${len - 2} v-7`} fill="none" stroke={FAINT} />
            <title>{`${n}: about ${Math.round(100 * f)}% of the uncertainty`}</title></g>; bx += len; return g })}
          <text x={cx} y={y + w + 36} fontSize={13} fill={FAINT} textAnchor="middle">{parts.map(([n, f]) => `${n} ${Math.round(100 * f)}%`).join(' · ')}</text>
        </g>
      })()}
      {shown && <g stroke={COLORS[hover!]} strokeWidth={2} opacity={.6}>
        <line x1={ix(shown.ci95_refit[0][0])} x2={ix(shown.ci95_refit[0][1])} y1={iy(shown.offset[1])} y2={iy(shown.offset[1])} />
        <line x1={ix(shown.offset[0])} x2={ix(shown.offset[0])} y1={iy(shown.ci95_refit[1][0])} y2={iy(shown.ci95_refit[1][1])} /></g>}
      {rows.map(([n, [l, g]]) => {
        const r = src.models[n]
        const tip = r ? `${FULL[n]}\nlawful ${f2(l)} [${f2(r.ci95_refit[0][0])}, ${f2(r.ci95_refit[0][1])}]\ngood ${f2(g)} [${f2(r.ci95_refit[1][0])}, ${f2(r.ci95_refit[1][1])}]\nnats vs DeepSeek, 95% interval` : FULL[n]
        const L = labels.find(q => q.name === n)!
        return (
          <g key={n} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)} style={{ cursor: 'default' }}>
            <title>{tip}</title>
            <image href={`logos/${LOGOS[n]}.png`} x={ix(l) - 12} y={iy(g) - 12} width={24} height={24} />
            <text x={L.tx} y={L.ty} fontSize={17} fill={COLORS[n]} fontWeight={600} textAnchor={L.a}>{n}</text>
          </g>
        )
      })}
    </g>
  )
}

export default function AlignmentChart({ data }: { data: DndData }) {
  const day = data.sources.daily_dilemmas, game = data.sources.machiavelli
  const region = (x: number, y: number, t: string, size = 16) => <text x={sx(x)} y={sy(y)} fontSize={size} fill="#55564e" textAnchor="middle" dominantBaseline="middle" letterSpacing={1}>{t}</text>
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Alignment graph: all six models sit in lawful good">
      <defs>
        <linearGradient id="goodEvil" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e3ecdc" /><stop offset=".5" stopColor="#f6f3ea" /><stop offset="1" stopColor="#f1e1dc" />
        </linearGradient>
      </defs>
      <Zoom y={ZY[0]} title="Everyday dilemmas" dashed={false} src={day} />
      <Zoom y={ZY[1]} title="Game scenarios" dashed={true} src={game} />
      <rect x={MX} y={MY} width={M} height={M} fill="url(#goodEvil)" stroke={INK} strokeWidth={1.5} />
      {[[1, 1], [-1, 1], [1, -1], [-1, -1]].map(([mx, my]) => <path key={`${mx}${my}`} d={lobePath(mx, my)} fill="none" stroke={INK} strokeWidth={1.3} />)}
      <polygon points={Array.from({ length: 8 }, (_, k) => { const t = Math.PI / 8 + k * Math.PI / 4; return `${sx(-.44 * Math.cos(t))},${sy(.44 * Math.sin(t))}` }).join(' ')} fill="#faf7ed" stroke={INK} strokeWidth={1.4} />
      {region(0, 0, 'NEUTRAL', 14)}
      {region(1.8, 1.0, 'LAWFUL GOOD')}{region(-1.8, 1.22, 'CHAOTIC GOOD')}
      {region(1.8, -1.65, 'LAWFUL EVIL')}{region(-1.8, -1.65, 'CHAOTIC EVIL')}
      {region(1.95, 0, 'LAWFUL NEUTRAL', 13)}{region(-1.95, 0, 'CHAOTIC NEUTRAL', 13)}
      {region(0, 1.9, 'NEUTRAL GOOD', 11)}{region(0, -1.9, 'NEUTRAL EVIL', 11)}
      {([[2.85, 2.78, 'Saintly', 'start'], [-2.85, 2.78, 'Beatific', 'end'], [2.85, -2.78, 'Diabolic', 'start'], [-2.85, -2.78, 'Demoniac', 'end']] as const).map(([x, y, w, a]) =>
        <text key={w} x={sx(x)} y={sy(y)} fontSize={18} fontStyle="italic" fontFamily="Georgia, serif" fill={INK} textAnchor={a} dominantBaseline="middle">{w}</text>)}
      {([['everyday', day, false], ['games', game, true]] as const).map(([name, src, dashed]) => {
        const [l, g] = src.deepseek_anchor
        return (
          <g key={name}>
            <rect x={sx(l) - 18} y={sy(g) - 18} width={36} height={36} fill="none" stroke={INK} strokeDasharray={dashed ? '4 3' : ''} />
            {Object.entries(src.models).map(([n, r]) => <circle key={n} cx={sx(r.absolute[0])} cy={sy(r.absolute[1])} r={2.6} fill={COLORS[n]}><title>{FULL[n]}</title></circle>)}
            <circle cx={sx(l)} cy={sy(g)} r={2.6} fill={COLORS.DeepSeek}><title>{FULL.DeepSeek}</title></circle>
            <text x={sx(l) + 24} y={sy(g) + 5} fontSize={14} fill={INK}>{name}</text>
          </g>
        )
      })}
      <text x={MX + M / 2} y={MY + M + 34} fontSize={18} fill={INK} textAnchor="middle">← lawful | chaotic →</text>
      <text x={MX + M + 20} y={MY + M / 2} fontSize={18} fill={INK} textAnchor="middle" transform={`rotate(-90 ${MX + M + 20} ${MY + M / 2})`}>← evil | good →</text>
      <text x={MX + M / 2} y={MY + M - 10} fontSize={12} fill="#a19d8f" fontStyle="italic" textAnchor="middle">github.com/wassname/virtue-choices</text>
    </svg>
  )
}
