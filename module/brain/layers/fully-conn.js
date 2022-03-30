import Volume from "../volume.js";

/**
 * FullyConn is fully connected dot products
 * @private
 */
const FullyConnLayer = class {
	/**
	 * @private
	 */
	constructor(
		{
			numNeurons,
			filters,
			outDepth = numNeurons || filters,
			l1_decayMul: l1DecayMul = 0,
			l2_decayMul: l2DecayMul = 1,
			inWidth,
			inHeight,
			inDepth,
			numInputs,
			biasPref: bias
		}
	) {
		Object.assign(
			this,
			{
				outDepth,
				l1_decayMul: l1DecayMul,
				l2_decayMul: l2DecayMul,
				numInputs: inWidth * inHeight * inDepth,
				outWidth: 1,
				outHeight: 1,
				layerType: "fc",
				filters: Array(outDepth)
					.fill()
					.map(() => new Volume(1, 1, numInputs)),
				biases: new Volume(1, 1, outDepth, bias)
			}
		);
	}

	/**
	 * @private
	 */
	forward(volume) {
		const {
			outDepth,
			numInputs,
			filters,
			biases: {
				w: biasesWeights
			}
		} = this;

		this.inAct = volume;

		const {
			w: volumeWeights
		} = volume;

		const newVolume = new Volume(1, 1, outDepth, 0.0);

		const {
			w: newVolumeWeights
		} = newVolume;

		for (const [
			filterIndex,
			{
				w: filterWeights
			}
		] of filters.entries()) {
			let initialValue = 0.0;

			for (let inputIndex = 0; inputIndex < numInputs; inputIndex++) {
				// for efficiency use Vols directly for now

				initialValue += volumeWeights[inputIndex] * filterWeights[inputIndex];
			}

			initialValue += biasesWeights[filterIndex];
			newVolumeWeights[filterIndex] = initialValue;
		}

		this.outAct = newVolume;

		return this.outAct;
	}

	/**
	 * @private
	 */
	backward() {
		const {
			inAct: volume,
			filters,
			outAct: {
				dw: outActGradientWeights
			},
			numInputs,
			biases: {
				dw: biasesGradientWeights
			}
		} = this;

		volume.dw = zeros(volume.w.length); // zero out the gradient in input Vol

		const {
			w: volumeWeights,
			dw: volumeGradientWeights
		} = volume;

		// compute gradient wrt weights and data
		for (const [
			filterIndex,
			{
				w: filterWeights,
				dw: filterGradientWeights
			}
		] of filters.entries()) {
			const chainGradient = outActGradientWeights[filterIndex];

			for (let inputIndex = 0; inputIndex < numInputs; inputIndex++) {
				// grad wrt input data
				volumeGradientWeights[inputIndex] += filterWeights[inputIndex] * chainGradient;

				// grad wrt params
				filterGradientWeights[inputIndex] += volumeWeights[inputIndex] * chainGradient;
			}

			biasesGradientWeights[filterIndex] += chainGradient;
		}
	}

	/**
	 * @private
	 */
	getParamsAndGrads() {
		const {
			outDepth,
			filters,
			l1_decayMul: l1DecayMul,
			l2_decayMul: l2DecayMul,
			biases: {
				biasesWeights,
				biasesGradientWeights
			}
		} = this;

		return [
			...Array(outDepth)
				.fill()
				.map((empty, index) => ({
					params: filters[index].w,
					grads: filters[index].dw,
					l2_decayMul: l2DecayMul,
					l1_decayMul: l1DecayMul
				})),
			{
				params: biasesWeights,
				grads: biasesGradientWeights,
				l1_decayMul: 0,
				l2_decayMul: 0
			}
		];
	}

	/**
	 * @private
	 */
	toJSON() {
		const {
			biases,
			filters,
			inDepth,
			l1_decayMul: l1DecayMul,
			l2_decayMul: l2DecayMul,
			layerType,
			outDepth,
			outWidth,
			outHeight,
			pad,
			stride,
			width,
			height
		} = this;

		return {
			biases: biases.toJSON(),
			filters: filters.map((filter) => filter.toJSON()),
			height,
			inDepth,
			l1_decayMul: l1DecayMul,
			l2_decayMul: l2DecayMul,
			layerType,
			outDepth,
			outHeight,
			outWidth,
			pad,
			stride,
			width
		};
	}

	/**
	 * @private
	 */
	static fromJSON(
		{
			biases,
			filters,
			inDepth,
			l1_decayMul: l1DecayMul = 1,
			l2_decayMul: l2DecayMul = 1,
			layerType,
			outDepth,
			outWidth,
			outHeight,
			pad = 0,
			stride,
			width,
			height
		}
	) {
		const layer = new FullyConnLayer();

		Object.assign(
			layer,
			{
				biases: Volume.fromJSON(biases),
				filters: filters.map((filter) => Volume.fromJSON(filter)),
				height,
				inDepth,
				l1_decayMul: l1DecayMul,
				l2_decayMul: l2DecayMul,
				layerType,
				outDepth,
				outHeight,
				outWidth,
				pad,
				stride,
				width
			}
		);

		return layer;
	}
};

export default FullyConnLayer;
