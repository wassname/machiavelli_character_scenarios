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
          Where each model's choices sit on the alignment graph from the 1979 <i>Dungeon Masters Guide</i>. The centre is a chooser that splits its percentages evenly.
          All six models sit close together in lawful good; the boxes on the right zoom in on how each differs from DeepSeek, and the grey cross is a typical 95% interval.
          <div className="footnote">Units: nats (log-odds) toward the more lawful or more good option, on DeepSeek's scale. Region boundaries are illustrative (±3 nats; neutral octagon 0.44).
          500 everyday dilemmas, 499 game scenarios. The braces split the cross's lawful width into its sources; the sources overlap, so the split is approximate.</div>
        </figcaption>
      </figure>
    </main>
  )
}
