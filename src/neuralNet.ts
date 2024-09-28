import assert from 'node:assert'

const MIN_ALPHA = 0.1
const MAX_ALPHA = 0.3
const MAX_ERR = 0.1

interface NeuralNetParams {
  readonly outputSize: number
  readonly inputSize: number
  readonly neuronSize: number
  readonly layerCount: number
}

export class NeuralNet {
  private readonly outputSize: number
  private readonly inputSize: number
  private readonly neuronSize: number
  private readonly layerCount: number

  private alpha: number = 0
  private layers: number[][]
  private expected: number[]
  private delta: number[][]
  private weights: number[][][]

  constructor(params: NeuralNetParams) {
    this.outputSize = params.outputSize
    this.inputSize = params.inputSize
    this.neuronSize = params.neuronSize
    this.layerCount = params.layerCount

    this.expected = new Array(this.outputSize)
    this.layers = this.initializeLayers()
    this.delta = this.initializeDelta()
    this.weights = this.initializeWeights()
  }

  public setInput(input: number[][]): void {
    assert(input.length === this.inputSize, 'Invalid input size')

    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        this.layers[0][i * this.inputSize + j] = input[i][j]
      }
    }
  }

  public setExpected(input: number[]): void {
    assert(input.length === this.outputSize, 'Invalid expected size')

    this.expected = input
  }

  public train(): void {
    this.clearNeuralNet()
    this.countNeuralNet()
    this.recalculateAlpha()

    if (this.err() > MAX_ERR) {
      this.adjustWeights()
    }
  }

  public apply(): number {
    this.clearNeuralNet()
    this.countNeuralNet()

    return this.layers[this.layerCount + 1].indexOf(
      Math.max(...this.layers[this.layerCount + 1]),
    )
  }

  public err(): number {
    let error = 0
    for (let i = 0; i < this.outputSize; i++) {
      error += Math.abs(this.expected[i] - this.layers[this.layerCount + 1][i])
    }
    return error / 2
  }

  /**
   * Initialize the layers of the neural net
   * total layers = input layer + inner layers + output layer
   */
  private initializeLayers(): number[][] {
    const layers = new Array<number[]>(this.layerCount + 2)
    layers[0] = new Array<number>(this.inputSize * this.inputSize)
    layers[this.layerCount + 1] = new Array(this.outputSize)

    for (let layer = 0; layer < this.layerCount; layer++) {
      layers[layer + 1] = new Array(this.neuronSize) // Inner layers
    }
    return layers
  }

  private initializeDelta(): number[][] {
    const delta = new Array<number[]>(this.layerCount + 1)
    delta[this.layerCount] = new Array<number>(this.outputSize)
    for (let layer = 0; layer < this.layerCount; layer++) {
      delta[layer] = new Array(this.neuronSize)
    }
    return delta
  }

  private initializeWeights(): number[][][] {
    const randomWeight = () => 0.3 * Math.random()
    const weight = new Array<number[][]>(this.layerCount + 1)

    weight[0] = new Array<number[]>(this.neuronSize)
    for (let neuron = 0; neuron < this.neuronSize; neuron++) {
      weight[0][neuron] = new Array<number>(this.inputSize * this.inputSize)
      for (let input = 0; input < this.inputSize * this.inputSize; input++) {
        weight[0][neuron][input] = randomWeight()
      }
    }

    weight[this.layerCount] = new Array<number[]>(this.outputSize)
    for (let neuron = 0; neuron < this.outputSize; neuron++) {
      weight[this.layerCount][neuron] = new Array<number>(this.neuronSize)
      for (let input = 0; input < this.neuronSize; input++) {
        weight[this.layerCount][neuron][input] = randomWeight()
      }
    }

    for (let layer = 1; layer < this.layerCount; layer++) {
      weight[layer] = new Array<number[]>(this.neuronSize)
      for (let neuron = 0; neuron < this.neuronSize; neuron++) {
        weight[layer][neuron] = new Array<number>(this.neuronSize)
        for (let input = 0; input < this.neuronSize; input++) {
          weight[layer][neuron][input] = randomWeight()
        }
      }
    }
    return weight
  }

  private activation(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  /**
   * Count the neural net
   * - for each layer in the neural net
   *  - for each neuron in the layer
   *    - for each neuron in the previous layer
   *      - calculate the sum of multiplication of the inputs and the weight
   *
   *    - apply the activation function to the sum
   *
   * F(∑j=1..n(k-1) Wijk * Yjk-1)
   */
  private countNeuralNet(): void {
    for (let layer = 0; layer <= this.layerCount; layer++) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        for (
          let input = 0;
          input < this.weights[layer][neuron].length;
          input++
        ) {
          this.layers[layer + 1][neuron] +=
            this.layers[layer][input] * this.weights[layer][neuron][input]
        }
        this.layers[layer + 1][neuron] = this.activation(
          this.layers[layer + 1][neuron],
        )
      }
    }
  }

  private clearNeuralNet(): void {
    for (let layer = 0; layer <= this.layerCount; layer++) {
      for (let neuron = 0; neuron < this.layers[layer + 1].length; neuron++) {
        this.layers[layer + 1][neuron] = 0
      }
    }
  }

  private recalculateAlpha(): void {
    const error = this.err()
    const relativeError = (2 * Math.abs(error)) / this.outputSize
    this.alpha = relativeError * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA
  }

  private adjustWeights(): void {
    for (let neuron = 0; neuron < this.outputSize; neuron++) {
      const t = this.expected[neuron]
      const y = this.layers[this.layerCount + 1][neuron]
      this.delta[this.layerCount][neuron] = y * (1 - y) * (t - y)
    }

    for (let layer = this.layerCount - 1; layer >= 0; layer--) {
      for (let input = 0; input < this.layers[layer + 1].length; input++) {
        let nextSum = 0
        for (
          let nextNeuron = 0;
          nextNeuron < this.layers[layer + 2].length;
          nextNeuron++
        ) {
          nextSum +=
            this.delta[layer + 1][nextNeuron] *
            this.weights[layer + 1][nextNeuron][input]
        }
        const y = this.layers[layer + 1][input]
        this.delta[layer][input] = y * (1 - y) * nextSum
      }
    }

    for (let layer = 0; layer < this.layerCount + 1; layer++) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron++) {
        for (
          let input = 0;
          input < this.weights[layer][neuron].length;
          input++
        ) {
          this.weights[layer][neuron][input] +=
            this.alpha * this.delta[layer][neuron] * this.layers[layer][input]
        }
      }
    }
  }
}
