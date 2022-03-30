import {
	undefinedOrNull
} from "../utilities.js";

import {
	FullyConnLayer,
	LocalResponseNormalizationLayer,
	DropoutLayer,
	InputLayer,
	SoftmaxLayer,
	RegressionLayer,
	ConvLayer,
	PoolLayer,
	ReluLayer,
	SigmoidLayer,
	TanhLayer,
	MaxoutLayer,
	SVMLayer
} from "./layers.js";

const layerAliases = new Map(
	[
		["fc", FullyConnLayer],
		["lrn", LocalResponseNormalizationLayer],
		["dropout", DropoutLayer],
		["input", InputLayer],
		["softmax", SoftmaxLayer],
		["regression", RegressionLayer],
		["conv", ConvLayer],
		["pool", PoolLayer],
		["relu", ReluLayer],
		["sigmoid", SigmoidLayer],
		["tanh", TanhLayer],
		["maxout", MaxoutLayer],
		["svm", SVMLayer]
	]
);

// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
class NeuralNetwork {
	/**
	 * @private
	 */
	constructor() {
		this.layers = [];
	}

	/**
	 * @private
	 */
	static #prefixDef(def) {
		const {
			type,
			numClasses,
			numNeurons,
			activation,
			biasPref
		} = def;

		switch (type) {
			case "softmax":
			case "svm":
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				return {
					numNeurons: numClasses,
					type: "fc"
				};

			case "regression":
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				return {
					numNeurons,
					type: "fc"
				};

			case "fc":
			case "conv":
				if (!biasPref) {
					return {
						...def,
						biasPref: activation === "relu" ? 0.1 : 0.0
					};
				}
				// relus like a bit of positive bias to get gradients early

				// otherwise it's technically possible that a relu unit will never turn on (by
				// chance) and will never get any gradient and never contribute any computation.
				// Dead relu.

				return [];
			default:
				return [];
		}
	}

	/**
	 * @private
	 */
	static #suffixDef(def) {
		const {
			activation,
			groupSize
		} = def;

		switch (activation) {
			case "relu":
			case "sigmoid":
			case "tanh":
				return {
					type: activation
				};
			case "maxout":
				// create maxout activation, and pass along group size, if provided
				return {
					groupSize: groupSize || 2,
					type: activation
				};
			case undefined:
				return [];
			default:
				throw new Error(`ERROR unsupported activation ${activation}`);
		}
	}

	// desugar layerDefs for adding activation, dropout layers etc
	/**
	 * @private
	 */
	static #desugar(defs) {
		return defs
			.map((def) => {
				const {
					type,
					dropProb
				} = def;

				const newDef = [
					this.#prefixDef(def),
					def,
					this.#suffixDef(def)
				];

				if (!undefinedOrNull(dropProb) && type !== "dropout") {
					newDef.push({
						dropProb,
						type: "dropout"
					});
				}

				return newDef;
			})
			.flat(2);
	}

	// takes a list of layer definitions and creates the network layer objects
	/**
	 * @private
	 */
	makeLayers(defs) {
		// few checks
		if (defs.length < 2) {
			throw new Error("Error! At least one input layer and one loss layer are required.");
		}

		if (defs[0].type !== "input") {
			throw new Error("Error! First layer must be the input layer, to declare size of inputs");
		}

		const desugaredDefs = NeuralNetwork.#desugar(defs);

		// create the layers
		this.layers = [];

		const {
			layers
		} = this;

		for (const [index, def] of desugaredDefs.entries()) {
			const {
				type
			} = def;

			let newDef = def;

			if (index > 0) {
				const {
					outWidth,
					outHeight,
					outDepth
				} = layers[index - 1];

				newDef = {
					...newDef,
					inDepth: outDepth,
					inWidth: outWidth,
					inHeight: outHeight
				};
			}

			if (layerAliases.has(type)) {
				const Layer = layerAliases.get(type);

				layers.push(new Layer(newDef));
			}
			else {
				throw new Error(`ERROR: UNRECOGNIZED LAYER TYPE: ${type}`);
			}
		}
	}

	// forward prop the network.
	// The trainer class passes isTraining = true, but when this function is
	// called from outside (not from the trainer), it defaults to prediction mode
	/**
	 * @private
	 */
	forward(volume, isTraining = false) {
		let act = this.layers[0].forward(volume, isTraining);

		for (let index = 1; index < this.layers.length; index++) {
			act = this.layers[index].forward(act, isTraining);
		}

		return act;
	}

	/**
	 * @private
	 */
	getCostLoss(volume, y) {
		this.forward(volume, false);
		const loss = this.layers.at(-1).backward(y);

		return loss;
	}

	// backprop: compute gradients wrt all parameters
	/**
	 * @private
	 */
	backward(reward) {
		const {
			layers: {
				length
			}
		} = this;

		const loss = this.layers.at(-1).backward(reward); // last layer assumed to be loss layer

		for (let index = length - 2; index >= 0; index--) { // first layer assumed input
			this.layers[index].backward();
		}

		return loss;
	}

	/**
	 * @private
	 */
	getParamsAndGrads() {
		const {
			layers
		} = this;

		// accumulate parameters and gradients for the entire network
		return layers.map((layer) => layer.getParamsAndGrads()).flat();
	}

	/**
	 * @private
	 */
	getPrediction() {
		const {
			layers
		} = this;

		// this is a convenience function for returning the argmax
		// prediction, assuming the last layer of the net is a softmax
		const lastLayer = layers.at(-1);

		if (lastLayer.layerType !== "softmax") {
			throw new Error("getPrediction function assumes softmax as last layer of the net!");
		}

		const weights = lastLayer.outAct.w;

		let maxWeight = weights[0];
		let maxIndex = 0;

		for (const [index, weight] of weights.entries()) {
			if (weight > maxWeight) {
				maxWeight = weight;
				maxIndex = index;
			}
		}

		return maxIndex; // return index of the class with highest class probability
	}

	/**
	 * @private
	 */
	toJSON() {
		const {
			layers
		} = this;

		return {
			layers: layers.map((layer) => layer.toJSON())
		};
	}

	/**
	 * @private
	 */
	static fromJSON({
		layers
	}) {
		const network = new NeuralNetwork();

		network.layers = layers.map((jsonLayer) => {
			const {
				layerType: type
			} = jsonLayer;

			const Layer = layerAliases.get(type);

			return Layer.fromJSON(jsonLayer);
		});

		return network;
	}
}

export default NeuralNetwork;
