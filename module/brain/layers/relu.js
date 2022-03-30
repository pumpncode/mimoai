// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
/**
 * @private
 */
class ReluLayer {
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
		this.layerType = "relu";
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;
		const V2 = V.clone();
		const N = V.w.length;
		const V2w = V2.w;

		for (let i = 0; i < N; i++) {
			if (V2w[i] < 0) {
				V2w[i] = 0;
			} // threshold at 0
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
			if (V2.w[i] <= 0) {
				V.dw[i] = 0;
			} // threshold
			else {
				V.dw[i] = V2.dw[i];
			}
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

export default ReluLayer;
