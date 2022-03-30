/**
 * implements an L2 regression cost layer,
 * so penalizes \sumI(||xI - yI||^2), where x is its input
 * and y is the user-provided array of "correct" values.
 * @private
 */
class RegressionLayer {
	/**
	 * @private
	 */
	constructor({
		inWidth,
		inHeight,
		inDepth
	}) {
		const product = inWidth * inHeight * inDepth;

		Object.assign(
			this,
			{
				numInputs: product,
				outDepth: product,
				outWidth: 1,
				outHeight: 1,
				layerType: "regression"
			}
		);
	}

	/**
	 * @private
	 */
	forward(volume) {
		this.inAct = volume;
		this.outAct = volume;

		return volume; // identity function
	}

	// y is a list here of size numInputs
	// or it can be a number if only one value is regressed
	// or it can be a struct {dim: i, val: x} where we only want to
	// regress on dimension i and asking it to have value x
	/**
	 * @private
	 */
	backward(y) {
		// compute and accumulate gradient wrt weights and bias of this layer
		const x = this.inAct;
		x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
		let loss = 0.0;

		if (y instanceof Array || y instanceof Float64Array) {
			for (var i = 0; i < this.outDepth; i++) {
				var dy = x.w[i] - y[i];
				x.dw[i] = dy;
				loss += 0.5 * dy * dy;
			}
		}
		else if (typeof y === "number") {
			// lets hope that only one number is being regressed
			var dy = x.w[0] - y;
			x.dw[0] = dy;
			loss += 0.5 * dy * dy;
		}
		else {
			// assume it is a struct with entries .dim and .val
			// and we pass gradient only along dimension dim to be equal to val
			var i = y.dim;
			const yi = y.val;
			var dy = x.w[i] - yi;
			x.dw[i] = dy;
			loss += 0.5 * dy * dy;
		}

		return loss;
	}

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
		const json = {};
		json.outDepth = this.outDepth;
		json.outWidth = this.outWidth;
		json.outHeight = this.outHeight;
		json.layerType = this.layerType;
		json.numInputs = this.numInputs;

		return json;
	}

	/**
	 * @private
	 */
	fromJSON(json) {
		this.outDepth = json.outDepth;
		this.outWidth = json.outWidth;
		this.outHeight = json.outHeight;
		this.layerType = json.layerType;
		this.numInputs = json.numInputs;
	}
}

export default RegressionLayer;
