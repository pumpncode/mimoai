/**
 * a window stores _size_ number of values
 * and returns averages. Useful for keeping running
 * track of validation or training accuracy during SGD
 * @private
 */
const Window = class {
	/**
	 * @private
	 */
	constructor(size, minsize) {
		this.v = [];
		this.size = typeof size === "undefined" ? 100 : size;
		this.minsize = typeof minsize === "undefined" ? 20 : minsize;
		this.sum = 0;
	}

	/**
	 * @private
	 */
	add(x) {
		this.v.push(x);
		this.sum += x;

		if (this.v.length > this.size) {
			const xold = this.v.shift();
			this.sum -= xold;
		}
	}

	/**
	 * @private
	 */
	getAverage() {
		if (this.v.length < this.minsize) {
			return -1;
		}

		return this.sum / this.v.length;
	}

	/**
	 * @private
	 */
	reset(x) {
		this.v = [];
		this.sum = 0;
	}
};

export default Window;
