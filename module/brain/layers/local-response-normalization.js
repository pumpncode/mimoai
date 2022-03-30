// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
/**
 * @private
 */
class LocalResponseNormalizationLayer {
	/**
	 * @param opt
	 * @example
	 */
	constructor(opt) {
		var opt = opt || {};

		// required
		this.k = opt.k;
		this.n = opt.n;
		this.alpha = opt.alpha;
		this.beta = opt.beta;

		// computed
		this.outWidth = opt.inWidth;
		this.outHeight = opt.inHeight;
		this.outDepth = opt.inDepth;
		this.layerType = "lrn";

		// checks
		if (this.n % 2 === 0) {
			console.log("WARNING n should be odd for LRN layer");
		}
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;

		const A = V.cloneAndZero();
		this.S_cache_ = V.cloneAndZero();
		const n2 = Math.floor(this.n / 2);

		for (let x = 0; x < V.width; x++) {
			for (let y = 0; y < V.height; y++) {
				for (let i = 0; i < V.depth; i++) {
					const ai = V.get(x, y, i);

					// normalize in a window of size n
					let den = 0.0;

					for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
						const aa = V.get(x, y, j);
						den += aa * aa;
					}

					den *= this.alpha / this.n;
					den += this.k;
					this.S_cache_.set(x, y, i, den); // will be useful for backprop
					den **= this.beta;
					A.set(x, y, i, ai / den);
				}
			}
		}

		this.outAct = A;

		return this.outAct; // dummy identity function for now
	}

	/**
	 * @example
	 */
	backward() {
		// evaluate gradient wrt data
		const V = this.inAct; // we need to set dw of this
		V.dw = global.zeros(V.w.length); // zero out gradient wrt data
		const A = this.outAct; // computed in forward pass

		const n2 = Math.floor(this.n / 2);

		for (let x = 0; x < V.width; x++) {
			for (let y = 0; y < V.height; y++) {
				for (let i = 0; i < V.depth; i++) {
					const chainGrad = this.outAct.getGrad(x, y, i);
					const S = this.S_cache_.get(x, y, i);
					const SB = S ** this.beta;
					const SB2 = SB * SB;

					// normalize in a window of size n
					for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
						const aj = V.get(x, y, j);

						let g = -aj * this.beta * S ** (this.beta - 1) * this.alpha / this.n * 2 * aj;

						if (j === i) {
							g += SB;
						}

						g /= SB2;
						g *= chainGrad;
						V.addGrad(x, y, j, g);
					}
				}
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
		json.k = this.k;
		json.n = this.n;
		json.alpha = this.alpha; // normalize by size
		json.beta = this.beta;
		json.outWidth = this.outWidth;
		json.outHeight = this.outHeight;
		json.outDepth = this.outDepth;
		json.layerType = this.layerType;

		return json;
	}

	/**
	 * @param json
	 * @example
	 */
	fromJSON(json) {
		this.k = json.k;
		this.n = json.n;
		this.alpha = json.alpha; // normalize by size
		this.beta = json.beta;
		this.outWidth = json.outWidth;
		this.outHeight = json.outHeight;
		this.outDepth = json.outDepth;
		this.layerType = json.layerType;
	}
}

export default LocalResponseNormalizationLayer;
