import { EOL } from 'node:os';
import * as path from 'path';
import * as fs from 'fs/promises';
import { inputPath, SAMPLES_COUNT } from './constants';
import { SourceType, ShapeType } from './types';

/**
 * Generate samples for each shape in both train and validate directories
 */
(async function main(): Promise<void> {
  console.info('Generating samples...');

  await ensureInputDir(SourceType.train);
  await ensureInputDir(SourceType.validate);

  for (const shape of Object.values(ShapeType)) {
    for (const dir of Object.values(SourceType)) {
      for (let i = 0; i < SAMPLES_COUNT; i++) {
        await generateSample(shape, i, dir);
      }
    }
    console.log(`Generated samples for -> ${shape}`);
  }
  console.info('Samples generated successfully');
})();

/**
 * Generate a sample for a given shape
 * @param name - name of the shape
 * @param postfix - postfix for the file name
 * @param dir - directory to save the file
 */
async function generateSample(name: ShapeType, postfix: number, dir: SourceType): Promise<void> {
  const template = await fs.readFile(getTemplateFilePath(name), {
    encoding: 'utf-8',
  });
  const [header, ...body] = template.split(EOL);

  const output = header + EOL + generateImage(body);
  const outputFilePath = getOutputFilePath(name, postfix, dir);
  await fs.writeFile(outputFilePath, output);
}

function generateImage(body: string[]): string {
  const symbolDelimiter = ' ';

  const matrix: number[][] = body.map((line) => line.split(symbolDelimiter).map(Number));
  const noize = Math.floor(Math.random() * 5) + 1;

  for (let i = 0; i < noize; i++) {
    const x = Math.floor(Math.random() * 7); // 0 to 6
    const y = Math.floor(Math.random() * 7); // 0 to 6
    matrix[x][y] = 1 - matrix[x][y]; // Flip the value
  }
  return matrix.map((line) => line.join(symbolDelimiter)).join(EOL);
}

async function ensureInputDir(dir: SourceType): Promise<void> {
  try {
    await fs.access(path.join(inputPath, dir));
  } catch {
    await fs.mkdir(path.join(inputPath, dir));
  }
}

function getTemplateFilePath(name: ShapeType): string {
  return path.join(inputPath, 'templates', `${name}.txt`);
}

function getOutputFilePath(name: ShapeType, postfix: number, dir: SourceType): string {
  return path.join(inputPath, dir, `${name}_${postfix}.txt`);
}
