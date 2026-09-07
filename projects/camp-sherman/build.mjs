import {build} from 'esbuild';
import {copyFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root=path.dirname(fileURLToPath(import.meta.url));
const output=path.resolve(root,'../../public/camp-sherman');
await mkdir(output,{recursive:true});
await build({entryPoints:[path.join(root,'viewer.mjs')],bundle:true,format:'esm',minify:true,outfile:path.join(root,'viewer.bundle.mjs')});
for(const name of ['index.html','styles.css','credits.html','viewer.bundle.mjs','THREE-LICENSE.txt','THREE-MESH-BVH-LICENSE.txt','MESHOPT-LICENSE.txt']) {
  await copyFile(path.join(root,name),path.join(output,name));
}
console.log('Built public/camp-sherman. Existing model, manifest, HDR, and preview assets preserved.');
