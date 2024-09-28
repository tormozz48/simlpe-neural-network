import {
  EPOCHS,
  INPUT_SIZE,
  LAYER_COUNT,
  LEARNING_BARRIER,
  MAX_BARRIER,
  MAX_ERR,
  NEURON_SIZE,
  OUTPUT_SIZE,
} from './constants'
import { iterateThoughtFiles } from './dataLoader'
import { NeuralNet } from './neuralNet'
import { SourceType } from './types'
;(async () => {
  try {
    await main()
  } catch (error) {
    console.error(error)
  }
})()

async function main(): Promise<void> {
  const neuralNet = new NeuralNet({
    inputSize: process.env.INPUT_SIZE
      ? parseInt(process.env.INPUT_SIZE)
      : INPUT_SIZE,
    neuronSize: process.env.NEURON_SIZE
      ? parseInt(process.env.NEURON_SIZE)
      : NEURON_SIZE,
    outputSize: process.env.OUTPUT_SIZE
      ? parseInt(process.env.OUTPUT_SIZE)
      : OUTPUT_SIZE,
    layerCount: process.env.LAYER_COUNT
      ? parseInt(process.env.LAYER_COUNT)
      : LAYER_COUNT,
  })

  await train(neuralNet)
  await validateResults(neuralNet)
}

async function train(neuralNet: NeuralNet): Promise<void> {
  let oldMeanError = 10
  let countBarrier = 0
  let meanError = 0

  let epoch = 0
  for (epoch = 0; epoch < EPOCHS; epoch++) {
    meanError = 0
    let meanCount = 0

    await iterateThoughtFiles(SourceType.train, ({ expected, input }) => {
      neuralNet.setExpected(expected)
      neuralNet.setInput(input)
      neuralNet.train()

      meanError += neuralNet.err()
      meanCount++
    })

    meanError /= meanCount
    if (
      Math.abs(oldMeanError - meanError) < LEARNING_BARRIER ||
      meanError > oldMeanError
    ) {
      countBarrier++
    } else {
      countBarrier = 0
    }
    oldMeanError = meanError
    if (countBarrier === MAX_BARRIER && meanError < MAX_ERR) {
      break
    }
  }

  console.info(`Train stopped on ${epoch} epoch`)
  console.info(`Mean error on train is ${meanError}`)
}

async function validateResults(neuralNet: NeuralNet): Promise<void> {
  let meanError = 0
  let meanCount = 0
  await iterateThoughtFiles(SourceType.validate, ({ expected, input }) => {
    neuralNet.setExpected(expected)
    neuralNet.setInput(input)
    const classified = neuralNet.apply()

    if (!expected[classified]) {
      console.warn('Error on image')
      console.warn('Expected: ', expected)
      console.warn('Actual: ', classified)
    }
    meanError += neuralNet.err()
    meanCount++
  })

  meanError /= meanCount
  console.info(`Mean error on validation is ${meanError}`)
}
