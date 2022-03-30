/**
 * @private
 */
class SVMLayer {
	/**
	 * @param opt
	 * @example
	 */
	constructor(opt) {
		var opt = opt || {};

		// computed
		this.numInputs = opt.inWidth * opt.inHeight * opt.inDepth;
		this.outDepth = this.numInputs;
		this.outWidth = 1;
		this.outHeight = 1;
		this.layerType = "svm";
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;
		this.outAct = V; // nothing to do, output raw scores

		return V;
	}

	/**
	 * @param y
	 * @example
	 */
	backward(y) {
		// compute and accumulate gradient wrt weights and bias of this layer
		const x = this.inAct;
		x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

		// we're using structured loss here, which means that the score
		// of the ground truth should be higher than the score of any other
		// class, by a margin
		const yscore = x.w[y]; // score of ground truth
		const margin = 1.0;

		let loss = 0.0;

		for (let i = 0; i < this.outDepth; i++) {
			if (y === i) {
				continue;
			}

			const ydiff = -yscore + x.w[i] + margin;

			if (ydiff > 0) {
				// violating dimension, apply loss
				x.dw[i] += 1;
				x.dw[y] -= 1;
				loss += ydiff;
			}
		}

		return loss;
	}

	/**
	 * @example
	 */
	getParamsAndGrads() {
		return [];
	}

	/**
	 * @example
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
	 * @param json
	 * @example
	 */
	fromJSON(json) {
		this.outDepth = json.outDepth;
		this.outWidth = json.outWidth;
		this.outHeight = json.outHeight;
		this.layerType = json.layerType;
		this.numInputs = json.numInputs;
	}
}

export default SVMLayer;
