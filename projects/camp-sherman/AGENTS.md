# Camp Sherman viewer

- Preserve the main house, two lofts, studio, garage, shelter, and surrounding landscape from the authored model.
- Keep orbit, free flight, room viewpoints, whole-property overview, roof cutaway, tree visibility, and desktop/touch controls working.
- In Fly mode, clicking the scene enables mouse look; clicking again or Escape stops it. Mouse dragging must also work, including when browser pointer lock is unavailable or refused. Touch dragging remains supported.
- Visual direction: professional architectural photography in warm late-afternoon sunlight through the trees. Preserve natural wood, dark metal roofing and stone, realistic material scale, soft furnishings and a planted woodland setting.
- Postprocessing must preserve cutaway and hidden-object state. Glass and alpha-cutout foliage participate in beauty rendering but must not become solid objects in the ambient-occlusion normal pass. Cap AO resolution independently, with lower mobile sampling.
- Keep sun shadows centered on the occupied part of the property, including the outbuildings, while caching small camera movements. Preserve distant needle coverage and smooth alpha-cutout edges.
- Fly mode uses camera-relative WASD/arrow movement, Shift for 4x speed, Space to rise and Control to descend. No gravity, wall/furniture/roof collision or site-boundary clamping. Released controls hover; Escape and focus loss clear movement. Touch controls include rise, descent and speed boost.
- Preserve authored collision metadata for source-model validation, but do not apply it to interactive flight.
- Keep all runtime resources local. Preserve Three.js, three-mesh-bvh, and Meshoptimizer license notices and material/source credits.
- Source modules and tests live here. `npm run build` bundles and copies runtime files into `public/camp-sherman/`; commit the rebuilt public bundle with source edits.
- The original PDF, correspondence, address, geographic source records, and editable Blender authoring project remain outside this public repository.
- Run the viewer unit and browser suites after behavior changes. Run root site tests and build before publication.
