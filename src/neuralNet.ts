import assert from 'node:assert';

const OUTPUT_SIZE = 3;
const INPUT_SIZE = 7;
const NEURON_SIZE = 14;
const LAYER_COUNT = 1;
const MIN_ALPHA = 0.1;
const MAX_ALPHA = 0.3;
const MAX_ERR = 0.1;

interface NeuralNetParams {
  readonly outputSize?: number;
  readonly inputSize?: number;
  readonly neuronSize?: number;
  readonly layerCount?: number;
}

export class NeuralNet {
  private readonly outputSize: number;
  private readonly inputSize: number;
  private readonly neuronSize: number;
  private readonly layerCount: number;
  
  private alpha: number = 0;
  private layers: number[][];
  private expected: number[];
  private delta: number[][];
  private weight: number[][][];

  constructor(params: NeuralNetParams = {}) {
    this.outputSize = params.outputSize ?? OUTPUT_SIZE;
    this.inputSize = params.inputSize ?? INPUT_SIZE;
    this.neuronSize = params.neuronSize ?? NEURON_SIZE;
    this.layerCount = params.layerCount ?? LAYER_COUNT;

    this.expected = [];
    this.layers = [];
    this.delta = [];
    this.weight = [];
  }

  public setIntput(input: number[][]): void {
    assert(input.length === this.inputSize, 'Invalid input size');

    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        this.layers[0][i * this.inputSize + j] = input[i][j];
      }
    }
  }

  public setExpected(input: number[]): void {
    assert(input.length === this.outputSize, 'Invalid expected size');

    this.expected = input;
  }

  public train(): void {
    this.clearNeuralNet();
    this.countNeuralNet();
    this.recalculateAlpha();

    if (this.err() > MAX_ERR) {
      this.adjustWeight();
    }
  }

  public apply(): number {
    this.clearNeuralNet();
    this.countNeuralNet();

    return this.layers[this.layerCount + 1].indexOf(Math.max(...this.layers[this.layerCount + 1]));
  }

  private activation(x: number): number {
    return 1 / (1 + Math.exp(-x));
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
      for (let neuron = 0; neuron < this.weight[layer].length; neuron++) {
        for (let input = 0; input < this.weight[layer][neuron].length; input++) {
          this.layers[layer + 1][neuron] += this.layers[layer][input] * this.weight[layer][neuron][input];
        }
        this.layers[layer + 1][neuron] = this.activation(this.layers[layer + 1][neuron]);
      }
    }
  }

  private clearNeuralNet(): void {
    for (let layer = 0; layer <= this.layerCount; layer++) {
      for (let neuron = 0; neuron < this.layers[layer + 1].length; neuron++) {
        this.layers[layer + 1][neuron] = 0;
      }
    }
  }

  private recalculateAlpha(): void {
    const error = this.err();
    const relativeError = 2 * Math.abs(error) / this.outputSize;
    this.alpha = relativeError * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA;
  }

  private adjustWeight(): void {}

  private err(): number {
    let error = 0;
    for (let i = 0; i < this.outputSize; i++) {
      error += Math.abs(this.expected[i] - this.layers[this.layerCount + 1][i]);
    }
    return error / 2;
  }
}