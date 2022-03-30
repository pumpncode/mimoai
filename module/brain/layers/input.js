/**
 * @private
 */
class InputLayer {
	/**
	 * @private
	 */
	constructor(
		{
			depth = 0,
			width = 1,
			height = 1
		} = {
				depth: 0,
				width: 1,
				height: 1
			}
	) {
		// required: depth
		this.outDepth = depth;

		// optional: default these dimensions to 1
		this.outWidth = width;
		this.outHeight = height;

		// computed
		this.layerType = "input";
	}

	/**
	 * @private
	 */
	forward(volume) {
		this.inAct = volume;
		this.outAct = volume;

		return this.outAct; // simply identity function for now
	}

	/**
	 * @private
	 */
	backward() { }

	/**
	 * @private
	 */
	getParamsAndGrads() {
		return [];
	}

	/**
	 * @private
	 */
	toJSON() {
		const {
			outDepth,
			outWidth,
			outHeight,
			layerType
		} = this;

		return {
			outDepth,
			outWidth,
			outHeight,
			layerType
		};
	}

	/**
	 * @private
	 */
	static fromJSON({
		outDepth,
		outWidth,
		outHeight,
		layerType
	}) {
		const layer = new InputLayer();

		Object.assign(layer, {
			outDepth,
			outWidth,
			outHeight,
			layerType
		});

		return layer;
	}
}

export default InputLayer;
