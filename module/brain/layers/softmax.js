// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible heightstem that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)
/**
 * @private
 */
class SoftmaxLayer {
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
		this.layerType = "softmax";
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;

		const A = new Volume(1, 1, this.outDepth, 0.0);

		// compute max activation
		const as = V.w;

		let amax = V.w[0];

		for (var i = 1; i < this.outDepth; i++) {
			if (as[i] > amax) {
				amax = as[i];
			}
		}

		// compute exponentials (carefully to not blow up)
		const es = global.zeros(this.outDepth);

		let esum = 0.0;

		for (var i = 0; i < this.outDepth; i++) {
			const e = Math.exp(as[i] - amax);
			esum += e;
			es[i] = e;
		}

		// normalize and output to sum to one
		for (var i = 0; i < this.outDepth; i++) {
			es[i] /= esum;
			A.w[i] = es[i];
		}

		this.es = es; // save these for backprop
		this.outAct = A;

		return this.outAct;
	}

	/**
	 * @param y
	 * @example
	 */
	backward(y) {
		// compute and accumulate gradient wrt weights and bias of this layer
		const x = this.inAct;
		x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

		for (let i = 0; i < this.outDepth; i++) {
			const indicator = i === y ? 1.0 : 0.0;
			const mul = -(indicator - this.es[i]);
			x.dw[i] = mul;
		}

		// loss is the class negative log likelihood
		return -Math.log(this.es[y]);
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

export default SoftmaxLayer;
