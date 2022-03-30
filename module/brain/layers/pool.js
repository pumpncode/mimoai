/**
 * @private
 */
class PoolLayer {
	/**
	 * @param opt
	 * @example
	 */
	constructor(opt) {
		var opt = opt || {};

		// required
		this.width = opt.width; // filter size
		this.inDepth = opt.inDepth;
		this.inWidth = opt.inWidth;
		this.inHeight = opt.inHeight;

		// optional
		this.height = typeof opt.height !== "undefined" ? opt.height : this.width;
		this.stride = typeof opt.stride !== "undefined" ? opt.stride : 2;
		this.pad = typeof opt.pad !== "undefined" ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

		// computed
		this.outDepth = this.inDepth;
		this.outWidth = Math.floor((this.inWidth + this.pad * 2 - this.width) / this.stride + 1);
		this.outHeight = Math.floor((this.inHeight + this.pad * 2 - this.height) / this.stride + 1);
		this.layerType = "pool";
		// store switches for x,y coordinates for where the max comes from, for each output neuron
		this.switchx = global.zeros(this.outWidth * this.outHeight * this.outDepth);
		this.switchy = global.zeros(this.outWidth * this.outHeight * this.outDepth);
	}

	/**
	 * @param V
	 * @param isTraining
	 * @example
	 */
	forward(V, isTraining) {
		this.inAct = V;

		const A = new Volume(this.outWidth, this.outHeight, this.outDepth, 0.0);

		let n = 0; // a counter for switches

		for (let d = 0; d < this.outDepth; d++) {
			let x = -this.pad;
			let y = -this.pad;

			for (let ax = 0; ax < this.outWidth; x += this.stride, ax++) {
				y = -this.pad;

				for (let ay = 0; ay < this.outHeight; y += this.stride, ay++) {
					// convolve centered at this particular location
					let a = -99999; // hopefully small enough ;\
					let winx = -1; let
						winy = -1;

					for (let fx = 0; fx < this.width; fx++) {
						for (let fy = 0; fy < this.height; fy++) {
							const oy = y + fy;
							const ox = x + fx;

							if (oy >= 0 && oy < V.height && ox >= 0 && ox < V.width) {
								const v = V.get(ox, oy, d);

								// perform max pooling and store pointers to where
								// the max came from. This will speed up backprop
								// and can help make nice visualizations in future
								if (v > a) {
									a = v; winx = ox; winy = oy;
								}
							}
						}
					}

					this.switchx[n] = winx;
					this.switchy[n] = winy;
					n++;
					A.set(ax, ay, d, a);
				}
			}
		}

		this.outAct = A;

		return this.outAct;
	}

	/**
	 * @example
	 */
	backward() {
		// pooling layers have no parameters, so simply compute
		// gradient wrt data here
		const V = this.inAct;
		V.dw = global.zeros(V.w.length); // zero out gradient wrt data
		const A = this.outAct; // computed in forward pass

		let n = 0;

		for (let d = 0; d < this.outDepth; d++) {
			let x = -this.pad;
			let y = -this.pad;

			for (let ax = 0; ax < this.outWidth; x += this.stride, ax++) {
				y = -this.pad;

				for (let ay = 0; ay < this.outHeight; y += this.stride, ay++) {
					const chainGrad = this.outAct.getGrad(ax, ay, d);
					V.addGrad(this.switchx[n], this.switchy[n], d, chainGrad);
					n++;
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
		json.width = this.width;
		json.height = this.height;
		json.stride = this.stride;
		json.inDepth = this.inDepth;
		json.outDepth = this.outDepth;
		json.outWidth = this.outWidth;
		json.outHeight = this.outHeight;
		json.layerType = this.layerType;
		json.pad = this.pad;

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
		this.width = json.width;
		this.height = json.height;
		this.stride = json.stride;
		this.inDepth = json.inDepth;
		this.pad = typeof json.pad !== "undefined" ? json.pad : 0; // backwards compatibility
		this.switchx = global.zeros(this.outWidth * this.outHeight * this.outDepth); // need to re-init these appropriately
		this.switchy = global.zeros(this.outWidth * this.outHeight * this.outDepth);
	}
}

export default PoolLayer;
