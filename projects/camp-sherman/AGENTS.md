# Camp Sherman viewer

- Preserve the main house, two lofts, studio, garage, shelter, and surrounding landscape from the authored model.
- Keep orbit, walking, room viewpoints, whole-property overview, roof cutaway, tree visibility, and desktop/touch controls working.
- Collision must prevent passage through walls and furniture while permitting doorways and both loft stairs. Roof visibility does not change physical collision.
- Keep all runtime resources local. Preserve Three.js, three-mesh-bvh, and Meshoptimizer license notices and material/source credits.
- Source modules and tests live here. `npm run build` bundles and copies runtime files into `public/camp-sherman/`; commit the rebuilt public bundle with source edits.
- The original PDF, correspondence, address, geographic source records, and editable Blender authoring project remain outside this public repository.
- Run the viewer unit and browser suites after behavior changes. Run root site tests and build before publication.
