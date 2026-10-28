import { Overview, Zoom, type DndData } from './AlignmentChart'
import raw from './data/dnd_web.json'

// Data: outputs/20261027_random1000_v12/calibration_heldout/dnd_web.json, copied into src/data by `just site-build`. -- PI/OpenAI
export default function App() {
  const data = structuredClone(raw) as unknown as DndData
  for (const s of Object.values(data.sources)) delete (s.models as Record<string, unknown>).DeepSeek  // DeepSeek is the anchor, drawn separately
  const shot = new URLSearchParams(location.search).get('shot')  // chart only, for README screenshots
  const day = <Zoom title="Everyday dilemmas" src={data.sources.daily_dilemmas} />
  const game = <Zoom title="Game scenarios" src={data.sources.machiavelli} />
  if (shot) return <main>{shot === 'overview' ? <Overview data={data} /> : shot === 'everyday' ? day : game}</main>
  return (
    <main>
      <h1>Which options do AI models choose in moral dilemmas?</h1>
      <figure id="alignment">
        <Overview data={data} />
        <figcaption>
          To a first approximation, they all choose lawful good. The alignment graph is from the 1979 <i>Dungeon Masters Guide</i>; its centre is a chooser that splits its percentages evenly.
          The two small boxes hold the six models in everyday dilemmas and in game scenes.
          <div className="footnote">Units: nats (log-odds) toward the more lawful or more good option, on DeepSeek's scale; +1 nat is about 2.7 times the odds. Region boundaries are illustrative (±3 nats; neutral octagon 0.44). 500 everyday dilemmas, 499 game scenarios.</div>
        </figcaption>
      </figure>
      <figure id="zoom-everyday">{day}</figure>
      <figure id="zoom-games">
        {game}
        <figcaption>
          Zoom in, and some lean more chaotic or less good. Hover a model for its full name, its own 95% interval and run details.
          <div className="footnote">The uncertainty cross is one typical model's 95% interval, at the same scale as the zoom; it is a scale, not a significance test. The braces are rough weights for where its width comes from: differences between questions, the calibration step, and the 2 answers per question. These sources overlap, so the split is approximate.</div>
        </figcaption>
      </figure>
    </main>
  )
}
