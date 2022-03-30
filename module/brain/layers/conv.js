import Volume from "../volume.js";

/**
 * ConvLayer does convolutions (so weight sharing spatially)
 * @private
 */
const ConvLayer = class {
	/**
	 * @private
	 */
	constructor({
		filters: outDepth,
		width, // filter size. Should be odd if possible, it's cleaner.
		height = width,
		inDepth,
		inWidth,
		inHeight,
		stride = 1, // stride at which we apply filters to input volume
		pad = 0, // amount of 0 padding to add around borders of input volume
		l1_decayMul: l1DecayMul = 0,
		l2_decayMul: l2DecayMul = 1,
		biasPref: bias = 0
	}) {
		// note we are doing floor, so if the strided convolution of the filter doesnt fit into the
		// input volume exactly, the output volume will be trimmed and not contain the (incomplete)
		// computed final application.

		Object.assign(
			this,
			{
				biases: new Volume(1, 1, outDepth, bias),
				filters: Array(outDepth)
					.fill()
					.map(() => new Volume(width, height, inDepth)),
				height,
				inDepth,
				inHeight,
				inWidth,
				l1_decayMul: l1DecayMul,
				l2_decayMul: l2DecayMul,
				layerType: "conv",
				outDepth,
				outHeight: Math.floor((inHeight + pad * 2 - height) / stride + 1),
				outWidth: Math.floor((inWidth + pad * 2 - width) / stride + 1),
				pad,
				stride,
				width
			}
		);
	}

	/**
	 * @private
	 */
	forward(volume) {
		const {
			outWidth,
			outHeight,
			outDepth,
			stride,
			filters,
			biases: {
				w: biasesWeights
			},
			pad
		} = this;

		// optimized code by @mdda that achieves 2x speedup over previous version
		this.inAct = volume;

		const newVolume = new Volume(outWidth, outHeight, outDepth, 0);

		const {
			width: volumeWidth,
			height: volumeHeight,
			depth: volumeDepth,
			w: volumeWeights
		} = volume;

		for (const [filterIndex, filter] of filters.entries()) {
			const {
				width: filterWidth,
				height: filterHeight,
				depth: filterDepth,
				w: filterWeights
			} = filter;

			let x = -pad;
			let y = -pad;

			for (
				let outHeightIndex = 0;
				outHeightIndex < outHeight;
				y += stride, outHeightIndex++
			) {
				x = -pad;

				for (
					let outWidthIndex = 0;
					outWidthIndex < outWidth;
					x += stride, outWidthIndex++
				) {
					// convolve centered at this particular location
					let initialValue = 0;

					for (let filterY = 0; filterY < filterHeight; filterY++) {
						// coordinates in the original input array coordinates
						const originalY = y + filterY;

						for (let filterX = 0; filterX < filterWidth; filterX++) {
							const originalX = x + filterX;

							if (
								originalY >= 0 &&
								originalY < volumeHeight &&
								originalX >= 0 &&
								originalX < volumeWidth
							) {
								for (
									let filterDepthIndex = 0;
									filterDepthIndex < filterDepth;
									filterDepthIndex++
								) {
									// avoid function call overhead (x2) for efficiency, compromise
									// modularity :(

									const filterWeightIndex = (
										(
											filterWidth *
											filterY
										) +
										filterX
									) *
										filterDepth +
										filterDepthIndex;

									const volumeWeightIndex = (
										(
											volumeWidth *
											originalY
										) +
										originalX
									) *
										volumeDepth +
										filterDepthIndex;

									initialValue += filterWeights[filterWeightIndex] *
										volumeWeights[volumeWeightIndex];
								}
							}
						}
					}

					initialValue += biasesWeights[filterIndex];
					newVolume.set(outWidthIndex, outHeightIndex, filterIndex, initialValue);
				}
			}
		}

		this.outAct = newVolume;

		return this.outAct;
	}

	/**
	 * @private
	 */
	backward() {
		const {
			stride,
			inAct: volume,
			outDepth,
			filters,
			pad,
			outWidth,
			outHeight,
			outAct
		} = this;

		// zero out gradient wrt bottom data, we're about to fill it
		volume.dw = zeros(volume.w.length);

		const {
			width: volumeWidth,
			height: volumeHeight,
			w: volumeWeights,
			dw: volumeGradientWeights
		} = volume;

		for (const [filterIndex, filter] of filters.entries()) {
			const {
				width: filterWidth,
				height: filterHeight,
				depth: filterDepth,
				w: filterWeights,
				dw: filterGradientWeights
			} = filter;

			let x = -pad;
			let y = -pad;

			for (let outHeightIndex = 0; outHeightIndex < outHeight; y += stride, outHeightIndex++) {
				x = -pad;

				for (let outWidthIndex = 0; outWidthIndex < outWidth; x += stride, outWidthIndex++) {
					// convolve centered at this particular location
					// gradient from above, from chain rule
					const chainGradient = outAct.getGrad(outWidthIndex, outHeightIndex, filterIndex);

					for (let filterY = 0; filterY < filterHeight; filterY++) {
						// coordinates in the original input array coordinates
						const originalY = y + filterY;

						for (let filterX = 0; filterX < filterWidth; filterX++) {
							const originalX = x + filterX;

							if (
								originalY >= 0 &&
								originalY < volumeHeight &&
								originalX >= 0 &&
								originalX < volumeWidth
							) {
								for (
									let filterDepthIndex = 0;
									filterDepthIndex < filterDepth;
									filterDepthIndex++
								) {
									// avoid function call overhead (x2) for efficiency, compromise
									// modularity :(
									const filterGradientWeightIndex = (
										(
											volumeWidth *
											originalY
										) +
										originalX
									) *
										volume.depth +
										filterDepthIndex;

									const volumeGradientWeightIndex = (
										(
											filterWidth *
											filterY
										) +
										filterX
									) *
										filterDepth +
										filterDepthIndex;

									filterGradientWeights[volumeGradientWeightIndex] += volumeWeights[filterGradientWeightIndex] * chainGradient;
									volumeGradientWeights[filterGradientWeightIndex] += filterWeights[volumeGradientWeightIndex] * chainGradient;
								}
							}
						}
					}

					this.biases.dw[filterIndex] += chainGradient;
				}
			}
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
		const layer = new ConvLayer();

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

export default ConvLayer;
