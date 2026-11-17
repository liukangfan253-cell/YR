import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { parseArgs } from 'node:util';

const root = fileURLToPath(new URL('../../../../../', import.meta.url));
const defaultImages = path.join(root, 'V', 'Figure_V1', 'Figure1', 'b', 'Images');
const { values: args } = parseArgs({
    options: {
        'input-pptx': { type: 'string' },
        'output-dir': { type: 'string' },
        'node-modules': { type: 'string' },
    },
});
const modules = args['node-modules'] || path.join(
    os.homedir(), '.cache', 'codex-runtimes', 'codex-primary-runtime',
    'dependencies', 'node', 'node_modules',
);
const { FileBlob, PresentationFile } = await import(pathToFileURL(path.join(
    modules, '@oai', 'artifact-tool', 'dist', 'artifact_tool.mjs',
)).href);
const require = createRequire(path.join(modules, 'figure-export.cjs'));
const sharp = require('sharp');

let input = args['input-pptx'];
if (!input) {
    const decks = (await fs.readdir(defaultImages))
        .filter(name => name.toLowerCase().endsWith('.pptx') && !name.startsWith('~$'));
    if (decks.length !== 1) throw new Error('Use --input-pptx to select the workflow deck.');
    input = path.join(defaultImages, decks[0]);
}
const destination = path.resolve(args['output-dir'] || defaultImages);
await fs.mkdir(destination, { recursive: true });
const stem = path.join(destination, path.parse(input).name);
const deck = await PresentationFile.importPptx(await FileBlob.load(input));
if (deck.slides.items.length !== 1) throw new Error('Figure1b expects a one-slide deck.');
const slide = deck.slides.items[0];
const raster = Buffer.from(await (await deck.export({
    slide, format: 'png', scale: 600 / 96,
})).arrayBuffer());
const { width } = await sharp(raster).metadata();
await sharp(raster).resize(Math.round(width / 2)).withMetadata({ density: 300 })
    .png().toFile(stem + '.png');
await sharp(raster).withMetadata({ density: 600 }).tiff({ compression: 'lzw' })
    .toFile(stem + '.tiff');
console.log(`Exported Figure1b to ${destination}`);
