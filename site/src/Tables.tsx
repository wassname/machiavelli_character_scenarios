// Appendix tables, publication style (markdown-tables skill): key, then main score, one number per column, sorted best first,
// best bolded, arrow on headers with a preferred direction, baseline row in italics, notes last. -- PI/OpenAI
import { COLORS, FULL } from './AlignmentChart'

const f = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`
const Name = ({ m }: { m: string }) => m === 'DeepSeek repeat'
  ? <i title="DeepSeek's two answers compared with each other: what noise alone gives">DeepSeek repeat (noise)</i>
  : <span title={FULL[m]} style={{ color: COLORS[m], fontWeight: 600 }}>{m}</span>


const NOTES: Record<string, string> = {
  Luna: 'clear style that holds in both settings',
  Haiku: 'stable within each setting, but different between them',
}
export function ConsistencyTable({ rows }: { rows: [string, number, number, number][] }) {
  // One number: correlation of the model's style (66 virtues, overall goodness removed, vs DeepSeek) between game scenes and everyday dilemmas.
  // The split-half checks (same setting, two random halves of the questions) go in the tooltip.
  const models = rows.filter(r => r[0] !== 'DeepSeek repeat').sort((a, b) => b[3] - a[3])
  const base = rows.filter(r => r[0] === 'DeepSeek repeat')
  const best = Math.max(...models.map(r => r[3]))
  return (
    <table className="results">
      <thead><tr><th>Model</th><th className="num">same style in games and everyday↑</th><th>notes</th></tr></thead>
      <tbody>{[...models, ...base].map(r => {
        const isBase = r[0] === 'DeepSeek repeat'
        const tip = `split-half check (same setting, two random halves of the questions): games ${f(r[2])}, everyday ${f(r[1])}`
        return <tr key={r[0]} className={isBase ? 'base' : ''} title={tip}>
          <td><Name m={r[0]} /></td><td className="num">{!isBase && r[3] === best ? <b>{f(r[3])}</b> : f(r[3])}</td>
          <td className="note">{isBase ? 'noise floor: DeepSeek compared with itself' : NOTES[r[0]] ?? (r[3] < .5 ? 'unclear: below 0.5' : '')}</td></tr>
      })}</tbody>
    </table>
  )
}
