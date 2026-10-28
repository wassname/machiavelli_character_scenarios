import AlignmentChart, { type DndData } from './AlignmentChart'
import raw from './data/dnd_web.json'

// Data: outputs/20261027_random1000_v12/calibration_heldout/dnd_web.json, copied into src/data by `just site-build` (bundled, so the page also works from file://). -- PI/OpenAI
type WebData = DndData

export default function App() {
  const data = structuredClone(raw) as unknown as WebData
  for (const s of Object.values(data.sources)) delete s.models.DeepSeek  // DeepSeek is the anchor, drawn separately
  if (location.search.includes('shot')) return <main><AlignmentChart data={data} /></main>  // chart only, for the README screenshot
  return (
    <main>
      <h1>Which options do AI models choose in moral dilemmas?</h1>
      <figure id="alignment">
        <AlignmentChart data={data} />
        <figcaption>
          Right: the alignment graph from the 1979 <i>Dungeon Masters Guide</i>, with its centre at a chooser that splits its percentages evenly. All six models sit close together in lawful good.
          Left: the two clusters zoomed in, each model compared with DeepSeek. Hover a model for its full name and its own 95% interval.
          <div className="footnote">Units: nats (log-odds) toward the more lawful or more good option, on DeepSeek's scale; +1 nat is about 2.7 times the odds. Region boundaries are illustrative (±3 nats; neutral octagon 0.44).
          500 everyday dilemmas, 499 game scenarios. The cross is one typical model's 95% interval, shown as a scale, not a significance test. The braces are rough weights for where its width comes from; the sources overlap, so they are not an exact split.</div>
        </figcaption>
      </figure>
    </main>
  )
}
