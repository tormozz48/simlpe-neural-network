import * as path from 'path'

// data generation
export const inputPath = path.join(process.cwd(), 'input')
export const SAMPLES_COUNT = 20

// training
export const LEARNING_BARRIER = 0.001
export const MAX_BARRIER = 3
export const MAX_ERR = 0.1
export const EPOCHS = 50

// neural net
export const OUTPUT_SIZE = 3
export const INPUT_SIZE = 7
export const NEURON_SIZE = 14
export const LAYER_COUNT = 1
