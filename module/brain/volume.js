import {
	randomGaussian
} from "../utilities.js";

/**
 * Volume is the basic building block of all data in a net.
 * it is essentially just a 3D volume of numbers, with a
 * width (width), height (height), and depth (depth).
 * it is used to hold data for all filters, all volumes,
 * all weights, and also stores all gradients w.r.t.
 * the data. c is optionally a value to initialize the volume
 * with. If c is missing, fills the Volume with random numbers.
 * @private
 */
class Volume {
	/**
	 * @private
	 */
	constructor(widthOrInput, height, depth, initialValue) {
		// this is how you check if a variable is an array. Oh, Javascript :)
		if (Array.isArray(widthOrInput)) {
			// we were given a list in width, assume 1D volume and fill it up
			this.width = 1;
			this.height = 1;
			this.depth = widthOrInput.length;
			// we have to do the following copy because we want to use
			// fast typed arrays, not an ordinary javascript array
			this.w = new Float64Array(this.depth);
			this.dw = new Float64Array(this.depth);

			for (var i = 0; i < this.depth; i++) {
				this.w[i] = widthOrInput[i];
			}
		}
		else {
			// we were given dimensions of the vol
			this.width = widthOrInput;
			this.height = height;
			this.depth = depth;
			const n = widthOrInput * height * depth;
			this.w = new Float64Array(n);
			this.dw = new Float64Array(n);

			if (typeof initialValue === "undefined") {
				// weight normalization is done to equalize the output
				// variance of every neuron, otherwise neurons with a lot
				// of incoming connections have outputs of larger variance
				const scale = Math.sqrt(1.0 / (widthOrInput * height * depth));

				for (var i = 0; i < n; i++) {
					this.w[i] = randomGaussian(0.0, scale);
				}
			}
			else {
				for (var i = 0; i < n; i++) {
					this.w[i] = initialValue;
				}
			}
		}
	}

	/**
	 * @private
	 */
	get(x, y, d) {
		const ix = ((this.width * y) + x) * this.depth + d;

		return this.w[ix];
	}

	/**
	 * @private
	 */
	set(x, y, d, v) {
		const ix = ((this.width * y) + x) * this.depth + d;
		this.w[ix] = v;
	}

	/**
	 * @private
	 */
	add(x, y, d, v) {
		const ix = ((this.width * y) + x) * this.depth + d;
		this.w[ix] += v;
	}

	/**
	 * @private
	 */
	getGrad(x, y, d) {
		const ix = ((this.width * y) + x) * this.depth + d;

		return this.dw[ix];
	}

	/**
	 * @private
	 */
	setGrad(x, y, d, v) {
		const ix = ((this.width * y) + x) * this.depth + d;
		this.dw[ix] = v;
	}

	/**
	 * @private
	 */
	addGrad(x, y, d, v) {
		const ix = ((this.width * y) + x) * this.depth + d;
		this.dw[ix] += v;
	}

	/**
	 * @private
	 */
	cloneAndZero() {
		return new Volume(this.width, this.height, this.depth, 0.0);
	}

	/**
	 * @private
	 */
	clone() {
		const V = new Volume(this.width, this.height, this.depth, 0.0);
		const n = this.w.length;

		for (let i = 0; i < n; i++) {
			V.w[i] = this.w[i];
		}

		return V;
	}

	/**
	 * @private
	 */
	addFrom(V) {
		for (let k = 0; k < this.w.length; k++) {
			this.w[k] += V.w[k];
		}
	}

	/**
	 * @private
	 */
	addFromScaled(V, a) {
		for (let k = 0; k < this.w.length; k++) {
			this.w[k] += a * V.w[k];
		}
	}

	/**
	 * @private
	 */
	setConst(a) {
		for (let k = 0; k < this.w.length; k++) {
			this.w[k] = a;
		}
	}

	/**
	 * @private
	 */
	toJSON() {
		// todo: we may want to only save d most significant digits to save space
		const json = {};
		json.width = this.width;
		json.height = this.height;
		json.depth = this.depth;
		json.w = this.w;

		return json;
		// we wont back up gradients to save space
	}

	/**
	 * @private
	 */
	fromJSON(json) {
		this.width = json.width;
		this.height = json.height;
		this.depth = json.depth;

		const n = this.width * this.height * this.depth;
		this.w = new Float64Array(n);
		this.dw = new Float64Array(n);

		// copy over the elements.
		for (let i = 0; i < n; i++) {
			this.w[i] = json.w[i];
		}
	}
}

export default Volume;
