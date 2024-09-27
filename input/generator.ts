import * as path from 'path';
import * as fs from 'fs';

const trainDirPath = path.join(__dirname, 'train');
const validateDirPath = path.join(__dirname, 'validate');

const examplesCount = 20;

/**
 * Generate samples for each shape in both train and validate directories
 */
(async function main(): Promise<void> {
  for (let i = 0; i < examplesCount; i++) {
    await generateSample('circle', i, 'train');
    await generateSample('square', i, 'train');
    await generateSample('triangle', i, 'train');
    await generateSample('circle', i, 'validate');
    await generateSample('square', i, 'validate');
    await generateSample('triangle', i, 'validate');
  }
})();

/**
 * Generate a sample for a given shape
 * @param name - name of the shape
 * @param postfix - postfix for the file name
 * @param dir - directory to save the file
 */
async function generateSample(name: 'circle' | 'square' | 'triangle', postfix: number, dir: 'train' | 'validate'): Promise<void> {
    // Read the file
    const filePath = path.join(trainDirPath, `${name}.txt`);
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const lines = fileContent.split('\n');

    // First line (header or result)
    const res = lines[0];

    // Create a matrix from the remaining lines
    const matrix: number[][] = lines.slice(1).map(line => line.split(' ').map(Number));

    // Add random noise to the matrix
    const noize = Math.floor(Math.random() * 5) + 1;
    for (let i = 0; i < noize; i++) {
        const x = Math.floor(Math.random() * 7); // 0 to 6
        const y = Math.floor(Math.random() * 7); // 0 to 6
        matrix[x][y] = 1 - matrix[x][y]; // Flip the value
    }

    // Write the modified matrix to a new file
    const newFilePath = `${dir}/${name}_${postfix}.txt`;
    let output = res + '\n';
    for (const line of matrix) {
        output += line.join(' ') + '\n';
    }
    fs.writeFileSync(newFilePath, output);
}