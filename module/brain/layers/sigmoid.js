// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
/**
 * @private
 */
class SigmoidLayer {
	/**
	 * @param opt
	 * @example
	 */
	constructor(opt) {
		var opt = opt || {};

		// computed
		this.outWidth = opt.inWidth;
		this.outHeight = opt.inHeight;
		this.outDepth = opt.inDepth;
		this.layerType = "sigmoid";
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;
		const V2 = V.cloneAndZero();
		const N = V.w.length;
		const V2w = V2.w;
		const Vw = V.w;

		for (let i = 0; i < N; i++) {
			V2w[i] = 1.0 / (1.0 + Math.exp(-Vw[i]));
		}

		this.outAct = V2;

		return this.outAct;
	}

	/**
	 * @example
	 */
	backward() {
		const V = this.inAct; // we need to set dw of this
		const V2 = this.outAct;
		const N = V.w.length;
		V.dw = global.zeros(N); // zero out gradient wrt data

		for (let i = 0; i < N; i++) {
			const v2wi = V2.w[i];
			V.dw[i] = v2wi * (1.0 - v2wi) * V2.dw[i];
		}
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
	}
}

export default SigmoidLayer;
