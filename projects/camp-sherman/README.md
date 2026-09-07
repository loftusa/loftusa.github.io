# Camp Sherman

A static Three.js walk-through of a house and woodland property reconstructed in Blender. The project is served at `/camp-sherman/` and linked from Recent Projects.

```text
Private drawing + terrain sources → Blender authoring → full-detail .blend
                                                   → GLB → WebP + Meshopt
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

Blender node extras carry `collider`, `walkable`, and `layer: "roof" | "vegetation"`. Invisible trunk collision proxies use `collision_only: true`. Repeated tree instances share geometry. The viewer accelerates collision queries with a BVH and keeps roof/tree visibility independent of collision.

Navigation supports walking and the modeled stairs. It does not provide jumping, crouching, opening doors, or editable furniture. Doors are modeled open. Use room shortcuts to reach any room directly.

## Replacing the scene

Export a GLB with node extras and meter units. Convert textures to WebP, then apply Meshopt compression; reversing these steps decodes the geometry during texture conversion. Keep the uncompressed editable model separately. Copy the resulting file to `public/camp-sherman/assets/camp-sherman.glb` and update its manifest and preview together. Check walking routes, room viewpoints, collision, cutaway, tree visibility, desktop/mobile layout, and model load errors before publishing.

The source and fidelity notes visible to visitors are maintained in `credits.html`.
