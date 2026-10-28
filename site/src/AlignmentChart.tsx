// AD&D-style alignment graph (after the 1979 DMG, Appendix III): lawful left, good up, zero = a chooser that splits its % evenly.
// Region geometry copied from scripts/evaluation/plot_dnd_classic.py classic(). -- PI/OpenAI
import type { ReactNode } from 'react'

export type ModelRow = { offset: [number, number]; absolute: [number, number]; ci95_refit?: [[number, number], [number, number]] }
export type SourceData = {
  deepseek_anchor: [number, number]  // [lawful, good] nats vs an even split
  models: Record<string, ModelRow>    // offsets vs DeepSeek, [lawful, good]
  key_model: string
  variance_shares?: Record<'questions' | 'answers' | 'calibration', [number, number]>
}
export type DndData = { sources: Record<'daily_dilemmas' | 'machiavelli', SourceData> }

export const COLORS: Record<string, string> = { Muse: '#0467DF', Qwen: '#6950EF', Gemini: '#E37400', Luna: '#111111', Haiku: '#8C4A2F', DeepSeek: '#8a8a8a' }
const LOGOS: Record<string, string> = { Muse: 'meta', Qwen: 'qwen', Gemini: 'google', Luna: 'openai', Haiku: 'anthropic', DeepSeek: 'deepseek' }
const INK = '#4b4a43'

const SIZE = 900, PAD = 60, EXT = 3
const S = (SIZE - 2 * PAD) / (2 * EXT)  // px per nat
const sx = (lawful: number) => SIZE / 2 - lawful * S  // lawful on the left
const sy = (good: number) => SIZE / 2 - good * S

// One quadrant outline in display coordinates (x = chaotic, y = good), mirrored to the other three.
const LOBE: [number, number][] = [[-1, 3], [-.52, 2.25], [-.32, 1.05], [-.28, .78], [-.20, .44], [-.44, .20], [-.78, .28], [-1.05, .32], [-2.25, .52], [-3, 1]]
const lobePath = (mx: number, my: number) => {
  const p = LOBE.map(([x, y]) => [SIZE / 2 + x * mx * S, SIZE / 2 - y * my * S])
  return `M${p[0]} C${p[1]} ${p[2]} ${p[3]} C${p[4]} ${p[5]} ${p[6]} C${p[7]} ${p[8]} ${p[9]}`
}

type Box = [number, number, number, number]  // x0, y0, x1, y1
const hit = (a: Box, b: Box) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]

function placeLabels(pts: { name: string; x: number; y: number }[]) {
  // Greedy: first candidate spot that clears every marker and earlier label; leader line when moved far.
  const marks: Box[] = pts.map(p => [p.x - 10, p.y - 10, p.x + 10, p.y + 10])
  const taken: Box[] = []
  const cands: [number, number, 'start' | 'end' | 'middle'][] = [[13, 4, 'start'], [-13, 4, 'end'], [0, -14, 'middle'], [0, 24, 'middle'], [16, -12, 'start'], [-16, -12, 'end'], [16, 22, 'start'], [-16, 22, 'end'], [30, 4, 'start'], [-30, 4, 'end'], [0, -30, 'middle'], [0, 40, 'middle']]
  return pts.map(p => {
    const w = 9 * p.name.length
    for (const [dx, dy, a] of cands) {
      const x0 = a === 'start' ? p.x + dx : a === 'end' ? p.x + dx - w : p.x + dx - w / 2
      const b: Box = [x0, p.y + dy - 11, x0 + w, p.y + dy + 2]
      if (!marks.some(m => hit(b, m)) && !taken.some(t => hit(b, t))) { taken.push(b); return { ...p, tx: p.x + dx, ty: p.y + dy, a, far: Math.hypot(dx, dy) > 25 } }
    }
    throw new Error(`no label spot for ${p.name}`)
  })
}

function Inset({ x, y, w, title, src }: { x: number; y: number; w: number; title: string; src: SourceData }) {
  // Zoom on offsets from DeepSeek, ±R nats; DeepSeek sits at the centre by definition.
  const R = .2, k = (w / 2 - 16) / R  // px per nat
  const ix = (lawful: number) => x + w / 2 - lawful * k
  const iy = (good: number) => y + w / 2 - good * k
  const rows: [string, [number, number]][] = [['DeepSeek', [0, 0]], ...Object.entries(src.models).map(([n, r]) => [n, r.offset] as [string, [number, number]])]
  const labels = placeLabels(rows.map(([n, [l, g]]) => ({ name: n, x: ix(l), y: iy(g) })))
  const key = src.models[src.key_model]?.ci95_refit
  return (
    <g>
      <rect x={x} y={y} width={w} height={w} fill="#fffdf6" stroke={INK} strokeWidth={1} />
      <line x1={x} x2={x + w} y1={iy(0)} y2={iy(0)} stroke="#d8d5c8" />
      <line y1={y} y2={y + w} x1={ix(0)} x2={ix(0)} stroke="#d8d5c8" />
      {key && (() => {  // one uncertainty key: the median model's 95% interval, centred on DeepSeek, so a model outside it differs beyond noise
        const hw = (key[0][1] - key[0][0]) * k / 2, hh = (key[1][1] - key[1][0]) * k / 2, cx = ix(0), cy = iy(0)
        return <g stroke="#b9b5a6" strokeWidth={5} strokeOpacity={.45}><line x1={cx - hw} x2={cx + hw} y1={cy} y2={cy} /><line x1={cx} x2={cx} y1={cy - hh} y2={cy + hh} />
</g>
      })()}
      <text x={x} y={y - 10} fontSize={18} fill={INK}>{title} <tspan fill="#888">vs DeepSeek</tspan></text>
      {rows.map(([n, [l, g]]) => <image key={n} href={`logos/${LOGOS[n]}.png`} x={ix(l) - 9} y={iy(g) - 9} width={18} height={18} />)}
      {labels.map(L => (
        <g key={L.name}>
          {L.far && <line x1={L.x} y1={L.y} x2={L.tx} y2={L.ty - 4} stroke={COLORS[L.name]} strokeWidth={.6} />}
          <text x={L.tx} y={L.ty} fontSize={15} fill={COLORS[L.name]} fontWeight={600} textAnchor={L.a}>{L.name}</text>
        </g>
      ))}

      <text x={x + w} y={y + w + 18} fontSize={13} fill="#888" textAnchor="end">±{R} nats</text>
      {key && src.variance_shares && (() => {  // where the typical interval's lawful width comes from; pieces sum to the cross arm above
        const total = (key[0][1] - key[0][0]) * k, parts = (['questions', 'calibration', 'answers'] as const).map(n => [n, src.variance_shares![n][0]] as const)
        let x0 = ix(0) - total / 2
        const braces = parts.map(([n, f]) => { const len = f * total, g = <path key={n} d={`M${x0 + 1},${y + w + 28} v6 h${len - 2} v-6`} fill="none" stroke="#8f8b7c" />; x0 += len; return g })
        return <g>{braces}<text x={ix(0)} y={y + w + 52} fontSize={13} fill="#8f8b7c" textAnchor="middle">{parts.map(([n, f]) => `${n} ${Math.round(100 * f)}%`).join(' · ')}</text></g>
      })()}
    </g>
  )
}

export default function AlignmentChart({ data }: { data: DndData }) {
  const day = data.sources.daily_dilemmas, game = data.sources.machiavelli
  const clusters: [string, SourceData, boolean][] = [['everyday', day, true], ['games', game, false]]
  const label = (x: number, y: number, t: ReactNode, size = 15) => <text x={sx(x)} y={sy(y)} fontSize={size} fill="#55564e" textAnchor="middle" dominantBaseline="middle" letterSpacing={1}>{t}</text>
  const insets = { everyday: { x: SIZE + 20, y: PAD + 20, w: 300 }, games: { x: SIZE + 20, y: PAD + 450, w: 300 } }
  return (
    <svg viewBox={`0 0 ${SIZE + 340} ${SIZE + 30}`} width="100%" role="img" aria-label="Alignment graph: all six models sit in lawful good">
      <defs>
        <linearGradient id="goodEvil" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e3ecdc" /><stop offset=".5" stopColor="#f6f3ea" /><stop offset="1" stopColor="#f1e1dc" />
        </linearGradient>
      </defs>
      <rect x={PAD} y={PAD} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD} fill="url(#goodEvil)" stroke={INK} strokeWidth={1.5} />
      {[[1, 1], [-1, 1], [1, -1], [-1, -1]].map(([mx, my]) => <path key={`${mx}${my}`} d={lobePath(mx, my)} fill="none" stroke={INK} strokeWidth={1.3} />)}
      <polygon points={Array.from({ length: 8 }, (_, k) => { const t = Math.PI / 8 + k * Math.PI / 4; return `${sx(-.44 * Math.cos(t))},${sy(.44 * Math.sin(t))}` }).join(' ')} fill="#faf7ed" stroke={INK} strokeWidth={1.4} />
      {label(0, 0, 'NEUTRAL', 14)}
      {label(1.8, 1.22, 'LAWFUL GOOD')}{label(-1.8, 1.22, 'CHAOTIC GOOD')}
      {label(1.8, -1.65, 'LAWFUL EVIL')}{label(-1.8, -1.65, 'CHAOTIC EVIL')}
      {label(1.95, 0, 'LAWFUL NEUTRAL', 13)}{label(-1.95, 0, 'CHAOTIC NEUTRAL', 13)}
      {label(0, 1.9, 'NEUTRAL GOOD', 11)}{label(0, -1.9, 'NEUTRAL EVIL', 11)}
      {([[2.85, 2.8, 'Saintly', 'start'], [-2.85, 2.8, 'Beatific', 'end'], [2.85, -2.8, 'Diabolic', 'start'], [-2.85, -2.8, 'Demoniac', 'end']] as const).map(([x, y, w, a]) =>
        <text key={w} x={sx(x)} y={sy(y)} fontSize={18} fontStyle="italic" fontFamily="Georgia, serif" fill={INK} textAnchor={a} dominantBaseline="middle">{w}</text>)}
      {clusters.map(([name, src, filled]) => {
        const [l, g] = src.deepseek_anchor
        const box = { x: sx(l) - 22, y: sy(g) - 22, w: 44 }
        const ins = insets[name as 'everyday' | 'games']
        return (
          <g key={name}>
            <rect x={box.x} y={box.y} width={box.w} height={box.w} fill={filled ? '#6d6a5c' : 'none'} fillOpacity={filled ? .18 : 0} stroke={INK} strokeDasharray={filled ? '' : '4 3'} />
            {Object.entries(src.models).map(([n, r]) => <circle key={n} cx={sx(r.absolute[0])} cy={sy(r.absolute[1])} r={3} fill={COLORS[n]} />)}
            <circle cx={sx(l)} cy={sy(g)} r={3} fill={COLORS.DeepSeek} />
            <text x={box.x + box.w / 2} y={box.y - 7} fontSize={14} fill={INK} textAnchor="middle">{name}</text>
            <line x1={box.x + box.w} y1={box.y + box.w / 2} x2={ins.x} y2={ins.y + ins.w / 2} stroke={INK} strokeWidth={.7} strokeDasharray={filled ? '' : '4 3'} />
            <Inset {...ins} title={name === 'everyday' ? 'Everyday dilemmas' : 'Game scenarios'} src={src} />
          </g>
        )
      })}
      <text x={SIZE / 2} y={SIZE - PAD + 34} fontSize={18} fill={INK} textAnchor="middle">← lawful | chaotic →</text>
      <text x={PAD - 30} y={SIZE / 2} fontSize={18} fill={INK} textAnchor="middle" transform={`rotate(-90 ${PAD - 30} ${SIZE / 2})`}>← evil | good →</text>
      <text x={PAD} y={SIZE + 22} fontSize={13} fill="#999">github.com/wassname/virtue-choices</text>
    </svg>
  )
}
