/**
 * An inefficient dropout layer
 * Note this is not most efficient implementation since the layer before
 * computed all these activations and now we're just going to drop them :(
 * same goes for backward pass. Also, if we wanted to be efficient at test time
 * we could equivalently be clever and upscale during train and copy pointers during test
 * todo: make more efficient.
 * @private
 */
class DropoutLayer {
	/**
	 * @private
	 */
	constructor(opt) {
		var opt = opt || {};

		// computed
		this.outWidth = opt.inWidth;
		this.outHeight = opt.inHeight;
		this.outDepth = opt.inDepth;
		this.layerType = "dropout";
		this.dropProb = typeof opt.dropProb !== "undefined" ? opt.dropProb : 0.5;
		this.dropped = global.zeros(this.outWidth * this.outHeight * this.outDepth);
	}

	/**
	 * @private
	 */
	forward(V, isTraining) {
		this.inAct = V;

		if (typeof isTraining === "undefined") {
			isTraining = false;
		} // default is prediction mode

		const V2 = V.clone();
		const N = V.w.length;

		if (isTraining) {
			// do dropout
			for (var i = 0; i < N; i++) {
				if (Math.random() < this.dropProb) {
					V2.w[i] = 0; this.dropped[i] = true;
				} // drop!
				else {
					this.dropped[i] = false;
				}
			}
		}
		else {
			// scale the activations during prediction
			for (var i = 0; i < N; i++) {
				V2.w[i] *= this.dropProb;
			}
		}

		this.outAct = V2;

		return this.outAct; // dummy identity function for now
	}

	/**
	 * @private
	 */
	backward() {
		const V = this.inAct; // we need to set dw of this
		const chainGrad = this.outAct;
		const N = V.w.length;
		V.dw = global.zeros(N); // zero out gradient wrt data

		for (let i = 0; i < N; i++) {
			if (!this.dropped[i]) {
				V.dw[i] = chainGrad.dw[i]; // copy over the gradient
			}
		}
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
		json.dropProb = this.dropProb;

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
		this.dropProb = json.dropProb;
	}
}

export default DropoutLayer;
