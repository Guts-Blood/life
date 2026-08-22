#!/usr/bin/env node

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

function loadSharp() {
  const candidates = [
    "sharp",
    path.join(
      os.homedir(),
      ".cache",
      "codex-runtimes",
      "codex-primary-runtime",
      "dependencies",
      "node",
      "node_modules",
      "sharp",
    ),
  ];

  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (error) {
      if (error.code !== "MODULE_NOT_FOUND") {
        throw error;
      }
    }
  }

  throw new Error(
    "Missing the 'sharp' package. Run this in Codex Desktop, or install sharp for your Node.js runtime.",
  );
}

function collectSvgFiles(directory) {
  const files = [];

  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.name === ".git" || entry.name === "node_modules") {
      continue;
    }

    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectSvgFiles(entryPath));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".svg")) {
      files.push(entryPath);
    }
  }

  return files.sort();
}

async function main() {
  const sharp = loadSharp();
  const bootcampRoot = path.resolve(__dirname, "..");
  const svgFiles = collectSvgFiles(bootcampRoot);

  if (svgFiles.length === 0) {
    throw new Error(`No SVG files found under ${bootcampRoot}`);
  }

  for (const svgPath of svgFiles) {
    const pngPath = svgPath.replace(/\.svg$/i, ".png");
    const info = await sharp(svgPath)
      .png({ compressionLevel: 9 })
      .toFile(pngPath);
    const relativePngPath = path.relative(bootcampRoot, pngPath);
    console.log(`rendered ${relativePngPath} (${info.width}x${info.height})`);
  }

  console.log(`Rendered ${svgFiles.length} PNG previews.`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
