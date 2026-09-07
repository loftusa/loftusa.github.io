# Camp Sherman

A static Three.js fly-through of a house and woodland property reconstructed in Blender. The project is served at `/camp-sherman/` and linked from Recent Projects.

```text
Private drawing + terrain sources → Blender authoring → full-detail .blend
                                                   → UV1 ambient occlusion → GLB → WebP + Meshopt
                                                               ↓
Viewer source → esbuild → public/camp-sherman/ → Next.js rewrite
```

The public runtime consists of `index.html`, CSS, the bundled viewer, `scene.json`, a compressed GLB, a local forest HDR, a preview image, and source/license notes. Drawing files and private source records are not included.

## Development

Use Node 22 or later:

```sh
cd projects/camp-sherman
npm ci
npm test
npm run build
npm run serve
```

Open `http://localhost:8086/`. `npm run build` regenerates the bundle and copies the HTML/CSS/licenses into the public directory. It preserves model assets and the manifest. After editing the viewer, commit source and rebuilt runtime files together.

For browser regression checks, install Chromium with `npx playwright install chromium`, then run `npx playwright test`. Tests intercept the model request with a small room fixture. `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` can select an installed Chromium. Real-model navigation and visual checks must also be repeated after replacing the GLB; the synthetic fixture cannot establish architectural correctness.

## Model contract

All positions use meters with Y up. `scene.json` declares local scene/HDR paths, home camera and target, walking bounds, optional practical lights, and named viewpoints. A viewpoint's `position` is eye height (1.65 m above its floor); `mode: "orbit"` selects an aerial overview.

Blender node extras carry `collider`, `walkable`, and `layer: "roof" | "vegetation"`. Invisible trunk collision proxies use `collision_only: true`. Repeated tree instances share geometry. The viewer retains the authored collision metadata for source-model validation; interactive free flight bypasses ground and obstacle queries. Roof and tree visibility remain independent.

Navigation supports free flight through walls, floors, roofs and furniture, with no gravity or site-boundary clamping. The modeled stairs and open doors remain part of the scene. Use room shortcuts or Reset view to return to the house.

The existing internal `walk` mode identifier remains compatible with authored manifests and visibility styles; the visible control is labeled Fly. The legacy walking solver and collision checks remain available for source-model validation, but the viewer uses `advanceFlight`.

In Fly mode, click the scene and move the mouse to look around; click again or press Escape to stop. You can also hold the left mouse button and drag. Move in the direction you look with WASD or arrow keys. Hold Shift for 4× speed (10 m/s versus 2.5 m/s), Space to rise, or Control to descend. Release movement keys to hover; Escape and focus loss stop held movement. Click-to-look falls back to ordinary mouse movement when an embedded browser cannot capture the pointer; moving outside the scene stops this fallback. On touch screens, drag to look and hold the direction buttons to move; Up, Down and Fast control height and speed.

## Replacing the scene

Export a GLB with node extras and meter units. Convert textures to WebP, then apply Meshopt compression; reversing these steps decodes the geometry during texture conversion. Keep the uncompressed editable model separately. Copy the resulting file to `public/camp-sherman/assets/camp-sherman.glb` and update its manifest and preview together. Check free flight through geometry, room viewpoints, cutaway, tree visibility, desktop/mobile layout, and model load errors before publishing. The private walking-route audit remains a source-model check, not the active navigation contract.

The source and fidelity notes visible to visitors are maintained in `credits.html`.

## Rendering quality

Warm late-afternoon sunlight and the local environment light are composed in linear HDR with multisample antialiasing, bounded GTAO contact shadows and a single ACES display transform. Ambient-occlusion atlases in the GLB use UV1; base-color, roughness and normal maps retain their authored UV0 scale. Glass and alpha-cutout foliage are excluded only during the GTAO normal capture. Mobile rendering caps pixel ratio, AO resolution and antialiasing samples independently. The scene still renders only when the camera or controls change.

The editable Blender project retains calibrated CC0 photographic surfaces, original woven materials, curved furniture, and separately baked AO. The richer representative forest shares geometry across repeated trees; collision proxies remain separate.

Sun shadows follow selected viewpoints and walking or panning beyond 16 meters. The detailed 42-meter shadow radius expands in stable steps for the property overview, up to 100 meters; unchanged views reuse the cached shadow map. Fine needle cutouts retain distant alpha coverage and use multisample edge smoothing.
