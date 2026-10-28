// AD&D-style alignment graph (after the 1979 DMG, Appendix III): lawful left, good up, zero = a chooser that splits its % evenly.
// Three full-width figures: the overview, then one zoom per source with its uncertainty key beside it (off the graph).
// Region geometry from scripts/evaluation/plot_dnd_classic.py classic(). -- PI/OpenAI
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

export const COLORS: Record<string, string> = { Muse: '#0467DF', Qwen: '#6950EF', Gemini: '#E37400', Luna: '#111111', Haiku: '#8C4A2F', DeepSeek: '#6f6f6f' }
const LOGOS: Record<string, string> = { Muse: 'meta', Qwen: 'qwen', Gemini: 'google', Luna: 'openai', Haiku: 'anthropic', DeepSeek: 'deepseek' }
const FULL: Record<string, string> = { Muse: 'Meta Muse Spark 1.3', Qwen: 'Qwen 3.8 27B', Gemini: 'Google Gemini 3.8 Flash', Luna: 'OpenAI GPT-5.6 Luna', Haiku: 'Anthropic Claude Haiku 5.5', DeepSeek: 'DeepSeek v4.1 Flash (reference)' }
// Run metadata for tooltips: OpenRouter id, reasoning request, mean tokens per answer and cost of the 1,000-question v12 run (2 answers each).
const META: Record<string, { slug: string; inTok: number; outTok: number; reasoningTok: number; cost: number }> = {
  DeepSeek: { slug: 'deepseek/deepseek-v4.1-flash', inTok: 685, outTok: 749, reasoningTok: 565, cost: .70 },
  Muse: { slug: 'meta/muse-spark-1.3-contributor', inTok: 806, outTok: 1164, reasoningTok: 960, cost: .55 },
  Qwen: { slug: 'qwen/qwen3.8-27b', inTok: 711, outTok: 657, reasoningTok: 525, cost: 2.66 },
  Gemini: { slug: 'google/gemini-3.8-flash', inTok: 670, outTok: 664, reasoningTok: 490, cost: 5.98 },
  Luna: { slug: 'openai/gpt-5.6-luna', inTok: 726, outTok: 215, reasoningTok: 80, cost: .81 },
  Haiku: { slug: 'anthropic/claude-haiku-5.5', inTok: 1340, outTok: 308, reasoningTok: 20, cost: .58 },
}
const meta = (n: string) => { const m = META[n]; return `${m.slug}\nreasoning budget 10k tokens; per answer ${m.inTok} in, ${m.outTok} out (${m.reasoningTok} reasoning)\nrun cost $${m.cost.toFixed(2)} for 2,000 answers` }
const INK = '#2f2e2a'
const f2 = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`
const REPO = 'github.com/wassname/virtue-choices'

// ---------- overview ----------
const M = 760, MX = 50, MY = 20, EXT = 3
const S = M / (2 * EXT)  // px per nat
const sx = (lawful: number) => MX + M / 2 - lawful * S  // lawful on the left
const sy = (good: number) => MY + M / 2 - good * S
const LOBE: Pair[] = [[-1, 3], [-.52, 2.25], [-.32, 1.05], [-.28, .78], [-.20, .44], [-.44, .20], [-.78, .28], [-1.05, .32], [-2.25, .52], [-3, 1]]  // one quadrant (x = chaotic, y = good), mirrored
const lobePath = (mx: number, my: number) => {
  const p = LOBE.map(([x, y]) => [MX + M / 2 + x * mx * S, MY + M / 2 - y * my * S])
  return `M${p[0]} C${p[1]} ${p[2]} ${p[3]} C${p[4]} ${p[5]} ${p[6]} C${p[7]} ${p[8]} ${p[9]}`
}

export function Overview({ data }: { data: DndData }) {
  const day = data.sources.daily_dilemmas, game = data.sources.machiavelli
  const region = (x: number, y: number, t: string, size = 17) => <text x={sx(x)} y={sy(y)} fontSize={size} fill="#4f4d45" textAnchor="middle" dominantBaseline="middle" letterSpacing={1}>{t}</text>
  return (
    <svg viewBox={`0 0 ${MX + M + 50} ${MY + M + 50}`} width="100%" role="img" aria-label="Alignment graph: all six models sit in lawful good">
      <defs>
        <linearGradient id="goodEvil" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#e3ecdc" /><stop offset=".5" stopColor="#f6f3ea" /><stop offset="1" stopColor="#f1e1dc" />
        </linearGradient>
      </defs>
      <rect x={MX} y={MY} width={M} height={M} fill="url(#goodEvil)" stroke={INK} strokeWidth={1.5} />
      {[[1, 1], [-1, 1], [1, -1], [-1, -1]].map(([mx, my]) => <path key={`${mx}${my}`} d={lobePath(mx, my)} fill="none" stroke={INK} strokeWidth={1.3} />)}
      <polygon points={Array.from({ length: 8 }, (_, k) => { const t = Math.PI / 8 + k * Math.PI / 4; return `${sx(-.44 * Math.cos(t))},${sy(.44 * Math.sin(t))}` }).join(' ')} fill="#faf7ed" stroke={INK} strokeWidth={1.4} />
      {region(0, 0, 'NEUTRAL', 15)}
      {region(1.8, 1.0, 'LAWFUL GOOD')}{region(-1.8, 1.22, 'CHAOTIC GOOD')}
      {region(1.8, -1.65, 'LAWFUL EVIL')}{region(-1.8, -1.65, 'CHAOTIC EVIL')}
      {region(1.95, 0, 'LAWFUL NEUTRAL', 14)}{region(-1.95, 0, 'CHAOTIC NEUTRAL', 14)}
      {region(0, 1.9, 'NEUTRAL GOOD', 12)}{region(0, -1.9, 'NEUTRAL EVIL', 12)}
      {([[2.85, 2.78, 'Saintly', 'start'], [-2.85, 2.78, 'Beatific', 'end'], [2.85, -2.78, 'Diabolic', 'start'], [-2.85, -2.78, 'Demoniac', 'end']] as const).map(([x, y, w, a]) =>
        <text key={w} x={sx(x)} y={sy(y)} fontSize={20} fontStyle="italic" fontFamily="Georgia, serif" fill={INK} textAnchor={a} dominantBaseline="middle">{w}</text>)}
      {([['everyday', day, false], ['games', game, true]] as const).map(([name, src, dashed]) => {
        const [l, g] = src.deepseek_anchor
        return (
          <g key={name}>
            <rect x={sx(l) - 20} y={sy(g) - 20} width={40} height={40} fill="none" stroke={INK} strokeDasharray={dashed ? '4 3' : ''} />
            {Object.entries(src.models).map(([n, r]) => <circle key={n} cx={sx(r.absolute[0])} cy={sy(r.absolute[1])} r={3} fill={COLORS[n]}><title>{`${FULL[n]}\nlawful ${f2(r.absolute[0])}, good ${f2(r.absolute[1])} nats vs an even split\n${meta(n)}`}</title></circle>)}
            <circle cx={sx(l)} cy={sy(g)} r={3} fill={COLORS.DeepSeek}><title>{`${FULL.DeepSeek}\nlawful ${f2(l)}, good ${f2(g)} nats vs an even split\n${meta('DeepSeek')}`}</title></circle>
            <text x={sx(l) + 27} y={sy(g) + 6} fontSize={16} fill={INK}>{name}</text>
          </g>
        )
      })}
      <text x={MX + M / 2} y={MY + M + 34} fontSize={19} fill={INK} textAnchor="middle">← lawful | chaotic →</text>
      <text x={MX - 20} y={MY + M / 2} fontSize={19} fill={INK} textAnchor="middle" transform={`rotate(-90 ${MX - 20} ${MY + M / 2})`}>← evil | good →</text>
      <text x={MX + M / 2} y={MY + M - 10} fontSize={13} fill="#a19d8f" fontStyle="italic" textAnchor="middle">{REPO}</text>
    </svg>
  )
}

// ---------- zoom: one source, full width, uncertainty key beside it ----------
type Box = [number, number, number, number]
const hit = (a: Box, b: Box) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
function placeLabels(pts: { name: string; x: number; y: number }[]) {
  // Greedy: first candidate spot that clears every marker and earlier label.
  const marks: Box[] = pts.map(p => [p.x - 16, p.y - 16, p.x + 16, p.y + 16])
  const taken: Box[] = []
  const cands: [number, number, 'start' | 'end' | 'middle'][] = [[22, 7, 'start'], [-22, 7, 'end'], [0, -24, 'middle'], [0, 38, 'middle'], [24, -18, 'start'], [-24, -18, 'end'], [24, 32, 'start'], [-24, 32, 'end'], [44, 7, 'start'], [-44, 7, 'end']]
  return pts.map(p => {
    const w = 12.5 * p.name.length
    for (const [dx, dy, a] of cands) {
      const x0 = a === 'start' ? p.x + dx : a === 'end' ? p.x + dx - w : p.x + dx - w / 2
      const b: Box = [x0, p.y + dy - 17, x0 + w, p.y + dy + 4]
      if (!marks.some(m => hit(b, m)) && !taken.some(t => hit(b, t))) { taken.push(b); return { ...p, tx: p.x + dx, ty: p.y + dy, a } }
    }
    throw new Error(`no label spot for ${p.name}`)
  })
}

const Z = 640, R = .2, ZK = (Z / 2 - 24) / R  // zoom square, ±R nats, px per nat
export function Zoom({ title, src }: { title: string; src: SourceData }) {
  const [hover, setHover] = useState<string | null>(null)
  const x = 50, y = 50
  const ix = (lawful: number) => x + Z / 2 - lawful * ZK
  const iy = (good: number) => y + Z / 2 - good * ZK
  const rows: [string, Pair][] = [['DeepSeek', [0, 0]], ...Object.entries(src.models).map(([n, r]) => [n, r.offset] as [string, Pair])]
  const labels = placeLabels(rows.map(([n, [l, g]]) => ({ name: n, x: ix(l), y: iy(g) })))
  const shown = hover && hover !== 'DeepSeek' ? src.models[hover] : null
  // off-graph uncertainty key, same scale: one typical model's 95% interval; braces = rough share of its width by source
  const key = src.models[src.key_model].ci95_refit
  const kw = (key[0][1] - key[0][0]) * ZK, kh = (key[1][1] - key[1][0]) * ZK
  const parts = (['questions', 'calibration', 'answers'] as const).map(n => [n, src.variance_shares[n][0]] as const)
  const kx = x + Z + 50, ky = y + 60, cx = kx + kw / 2, cy = ky + 30 + kh / 2
  let bx = kx
  return (
    <svg viewBox={`0 0 ${kx + 420} ${y + Z + 50}`} width="100%" role="img" aria-label={`${title}: each model compared with DeepSeek`}>
      <text x={x} y={y - 16} fontSize={24} fill={INK}>{title}, each model vs DeepSeek</text>
      <rect x={x} y={y} width={Z} height={Z} fill="#ffffff" stroke={INK} strokeWidth={1.2} />
      <line x1={x} x2={x + Z} y1={iy(0)} y2={iy(0)} stroke="#bdbab0" />
      <line y1={y} y2={y + Z} x1={ix(0)} x2={ix(0)} stroke="#bdbab0" />
      <text x={x + 10} y={y + Z - 12} fontSize={16} fill={INK}>← more lawful</text>
      <text x={x + Z - 10} y={y + Z - 12} fontSize={16} fill={INK} textAnchor="end">more chaotic →</text>
      <text x={ix(0) + 8} y={y + 24} fontSize={16} fill={INK}>more good ↑</text>
      {shown && <g stroke={COLORS[hover!]} strokeWidth={2.5} opacity={.6}>
        <line x1={ix(shown.ci95_refit[0][0])} x2={ix(shown.ci95_refit[0][1])} y1={iy(shown.offset[1])} y2={iy(shown.offset[1])} />
        <line x1={ix(shown.offset[0])} x2={ix(shown.offset[0])} y1={iy(shown.ci95_refit[1][0])} y2={iy(shown.ci95_refit[1][1])} /></g>}
      {rows.map(([n, [l, g]]) => {
        const r = src.models[n]
        const tip = (r ? `${FULL[n]}\nlawful ${f2(l)} [${f2(r.ci95_refit[0][0])}, ${f2(r.ci95_refit[0][1])}], good ${f2(g)} [${f2(r.ci95_refit[1][0])}, ${f2(r.ci95_refit[1][1])}] nats vs DeepSeek (95%)` : FULL[n]) + '\n' + meta(n)
        const L = labels.find(q => q.name === n)!
        return (
          <g key={n} onMouseEnter={() => setHover(n)} onMouseLeave={() => setHover(null)}>
            <title>{tip}</title>
            <image href={`logos/${LOGOS[n]}.png`} x={ix(l) - 15} y={iy(g) - 15} width={30} height={30} />
            <text x={L.tx} y={L.ty} fontSize={21} fill={COLORS[n]} fontWeight={700} textAnchor={L.a}>{n}</text>
          </g>
        )
      })}
      <text x={x + Z - 10} y={y + 24} fontSize={14} fill={INK} textAnchor="end">±{R} nats</text>
      <text x={kx} y={ky} fontSize={18} fill={INK}>uncertainty (95%, typical model)</text>
      <g stroke={INK} strokeWidth={1.6}><line x1={cx - kw / 2} x2={cx + kw / 2} y1={cy} y2={cy} /><line x1={cx} x2={cx} y1={cy - kh / 2} y2={cy + kh / 2} /></g>
      {parts.map(([n, f]) => { const len = f * kw, g = <g key={n}>
        <path d={`M${bx + 1},${cy + kh / 2 + 14} v8 h${len - 2} v-8`} fill="none" stroke={INK} />
        <title>{`${n}: about ${Math.round(100 * f)}% of the uncertainty`}</title></g>; bx += len; return g })}
      {parts.map(([n, f], i) => <text key={n} x={kx} y={cy + kh / 2 + 46 + 22 * i} fontSize={16} fill={INK}>{n} {Math.round(100 * f)}%</text>)}
    </svg>
  )
}
