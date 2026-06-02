/**
 * Helper to generate Gaussian (normal) distributed random numbers
 * using Box-Muller transform.
 */
export function randomGaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Represents a single layer in the Neural Network.
 */
export class Layer {
  constructor(inputSize, outputSize, randomize = true) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = []; // 2D array [outputSize][inputSize]
    this.biases = [];  // Array [outputSize]
    
    if (randomize) {
      this.randomize();
    } else {
      this.zeroInit();
    }
  }

  /**
   * Initialize weights and biases with small random values.
   * Using He or Xavier-like initialization for weights.
   */
  randomize() {
    const scale = Math.sqrt(2.0 / this.inputSize);
    for (let i = 0; i < this.outputSize; i++) {
      this.biases.push(randomGaussian() * 0.1);
      const row = [];
      for (let j = 0; j < this.inputSize; j++) {
        row.push(randomGaussian() * scale);
      }
      this.weights.push(row);
    }
  }

  /**
   * Initialize weights and biases with zeros.
   */
  zeroInit() {
    for (let i = 0; i < this.outputSize; i++) {
      this.biases.push(0);
      const row = new Array(this.inputSize).fill(0);
      this.weights.push(row);
    }
  }

  /**
   * Feed inputs through this layer.
   * Returns an array of output values after tanh activation.
   */
  feedForward(inputs) {
    const outputs = [];
    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += inputs[j] * this.weights[i][j];
      }
      // Tanh activation function maps outputs to [-1, 1]
      outputs.push(Math.tanh(sum));
    }
    return outputs;
  }

  /**
   * Mutate the weights and biases of this layer.
   * @param {number} rate - Probability of mutation [0, 1]
   * @param {number} amount - Standard deviation of Gaussian change
   */
  mutate(rate, amount) {
    // Mutate biases
    for (let i = 0; i < this.outputSize; i++) {
      if (Math.random() < rate) {
        this.biases[i] += randomGaussian() * amount;
      }
      // Mutate weights
      for (let j = 0; j < this.inputSize; j++) {
        if (Math.random() < rate) {
          this.weights[i][j] += randomGaussian() * amount;
        }
      }
    }
  }

  /**
   * Create a deep copy of this layer.
   */
  copy() {
    const clone = new Layer(this.inputSize, this.outputSize, false);
    clone.biases = [...this.biases];
    clone.weights = this.weights.map(row => [...row]);
    return clone;
  }

  /**
   * Perform uniform crossover between this layer and a partner layer.
   * @param {Layer} partner
   */
  crossover(partner) {
    const child = new Layer(this.inputSize, this.outputSize, false);
    for (let i = 0; i < this.outputSize; i++) {
      // Crossover biases
      child.biases[i] = Math.random() < 0.5 ? this.biases[i] : partner.biases[i];
      
      // Crossover weights
      for (let j = 0; j < this.inputSize; j++) {
        child.weights[i][j] = Math.random() < 0.5 ? this.weights[i][j] : partner.weights[i][j];
      }
    }
    return child;
  }
}

/**
 * Feedforward Neural Network composed of multiple Layers.
 */
export class NeuralNetwork {
  /**
   * @param {number[]} layerSizes - Array specifying number of neurons in each layer.
   *                                e.g. [11, 8, 6, 2] for 11 inputs, hidden layers of 8 & 6, and 2 outputs.
   */
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    this.layers = [];
    
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(new Layer(layerSizes[i], layerSizes[i + 1]));
    }
  }

  /**
   * Feedforward input array through all network layers.
   * Returns activation values for every node in each layer, 
   * which is highly useful for real-time visualization of the network graph.
   * The final output of the network is the last array in the returned list.
   * @param {number[]} inputs
   * @returns {number[][]} - Array of activations for each layer, including input layer.
   */
  feedForward(inputs) {
    const activations = [inputs];
    let currentInputs = inputs;
    
    for (let i = 0; i < this.layers.length; i++) {
      currentInputs = this.layers[i].feedForward(currentInputs);
      activations.push(currentInputs);
    }
    
    return activations;
  }

  /**
   * Convenience method to get just the final output values.
   * @param {number[]} inputs
   */
  getOutput(inputs) {
    const activations = this.feedForward(inputs);
    return activations[activations.length - 1];
  }

  /**
   * Mutates all layers in the network.
   * @param {number} rate - Probability of mutation [0, 1]
   * @param {number} amount - Mutation strength
   */
  mutate(rate, amount) {
    for (let layer of this.layers) {
      layer.mutate(rate, amount);
    }
  }

  /**
   * Creates a deep copy of the Neural Network.
   */
  copy() {
    const clone = new NeuralNetwork(this.layerSizes);
    clone.layers = this.layers.map(layer => layer.copy());
    return clone;
  }

  /**
   * Performs crossover layer-by-layer with a partner network.
   * @param {NeuralNetwork} partner
   */
  crossover(partner) {
    const child = new NeuralNetwork(this.layerSizes);
    child.layers = this.layers.map((layer, index) => {
      return layer.crossover(partner.layers[index]);
    });
    return child;
  }
}
