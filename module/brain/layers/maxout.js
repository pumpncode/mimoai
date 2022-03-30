// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size groupSize. Ideally of course,
// the input size should be exactly divisible by groupSize
/**
 * @private
 */
class MaxoutLayer {
	/**
	 * @param opt
	 * @example
	 */
	constructor(opt) {
		var opt = opt || {};

		// required
		this.groupSize = typeof opt.groupSize !== "undefined" ? opt.groupSize : 2;

		// computed
		this.outWidth = opt.inWidth;
		this.outHeight = opt.inHeight;
		this.outDepth = Math.floor(opt.inDepth / this.groupSize);
		this.layerType = "maxout";

		this.switches = global.zeros(this.outWidth * this.outHeight * this.outDepth); // useful for backprop
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;
		const N = this.outDepth;
		const V2 = new Volume(this.outWidth, this.outHeight, this.outDepth, 0.0);

		// optimization branch. If we're operating on 1D arrays we dont have
		// to worry about keeping track of x,y,d coordinates inside
		// input volumes. In convnets we do :(
		if (this.outWidth === 1 && this.outHeight === 1) {
			for (var i = 0; i < N; i++) {
				var ix = i * this.groupSize; // base index offset
				var a = V.w[ix];
				var ai = 0;

				for (var j = 1; j < this.groupSize; j++) {
					var a2 = V.w[ix + j];

					if (a2 > a) {
						a = a2;
						ai = j;
					}
				}

				V2.w[i] = a;
				this.switches[i] = ix + ai;
			}
		}
		else {
			let n = 0; // counter for switches

			for (let x = 0; x < V.width; x++) {
				for (let y = 0; y < V.height; y++) {
					for (var i = 0; i < N; i++) {
						var ix = i * this.groupSize;
						var a = V.get(x, y, ix);
						var ai = 0;

						for (var j = 1; j < this.groupSize; j++) {
							var a2 = V.get(x, y, ix + j);

							if (a2 > a) {
								a = a2;
								ai = j;
							}
						}

						V2.set(x, y, i, a);
						this.switches[n] = ix + ai;
						n++;
					}
				}
			}
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
		const N = this.outDepth;
		V.dw = global.zeros(V.w.length); // zero out gradient wrt data

		// pass the gradient through the appropriate switch
		if (this.outWidth === 1 && this.outHeight === 1) {
			for (var i = 0; i < N; i++) {
				var chainGrad = V2.dw[i];
				V.dw[this.switches[i]] = chainGrad;
			}
		}
		else {
			// bleh okay, lets do this the hard way
			let n = 0; // counter for switches

			for (let x = 0; x < V2.width; x++) {
				for (let y = 0; y < V2.height; y++) {
					for (var i = 0; i < N; i++) {
						var chainGrad = V2.getGrad(x, y, i);
						V.setGrad(x, y, this.switches[n], chainGrad);
						n++;
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
		json.outDepth = this.outDepth;
		json.outWidth = this.outWidth;
		json.outHeight = this.outHeight;
		json.layerType = this.layerType;
		json.groupSize = this.groupSize;

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
		this.groupSize = json.groupSize;
		this.switches = global.zeros(this.groupSize);
	}
}

export default MaxoutLayer;
