import { EOL } from 'node:os';
import * as path from 'path';
import * as fs from 'fs/promises';
import { inputPath } from './constants';
import { SourceType } from './types';

interface FileContent {
  readonly expected: number[];
  readonly input: number[][];
}

export async function iterateThoughtFiles(
  sourceType: SourceType,
  callback: ({ expected, input }: FileContent) => void,
): Promise<void> {
  const trainPath = path.join(inputPath, sourceType);
  const files = await fs.readdir(trainPath);
  const symbolDelimiter = ' ';

  for (const file of files) {
    const content = await fs.readFile(path.join(trainPath, file), {
      encoding: 'utf-8',
    });
    const [expectedLine, ...inputLines] = content.split(EOL);
    const expected = expectedLine.split(symbolDelimiter).map(Number);
    const input = inputLines.map((line) => line.split(symbolDelimiter).map(Number));
    callback({ expected, input });
  }
}
