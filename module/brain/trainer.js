/**
 * @private
 */
class Trainer {
	/**
	 * @param net
	 * @param options
	 * @example
	 */
	constructor(net, options = {}) {
		this.net = net;

		this.learningRate = typeof options.learningRate !== "undefined" ? options.learningRate : 0.01;
		this.l1_decay = typeof options.l1_decay !== "undefined" ? options.l1_decay : 0.0;
		this.l2_decay = typeof options.l2_decay !== "undefined" ? options.l2_decay : 0.0;
		this.batchSize = typeof options.batchSize !== "undefined" ? options.batchSize : 1;
		this.method = typeof options.method !== "undefined" ? options.method : "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

		this.momentum = typeof options.momentum !== "undefined" ? options.momentum : 0.9;
		this.ro = typeof options.ro !== "undefined" ? options.ro : 0.95; // used in adadelta
		this.eps = typeof options.eps !== "undefined" ? options.eps : 1e-8; // used in adam or adadelta
		this.beta1 = typeof options.beta1 !== "undefined" ? options.beta1 : 0.9; // used in adam
		this.beta2 = typeof options.beta2 !== "undefined" ? options.beta2 : 0.999; // used in adam

		this.k = 0; // iteration counter
		this.gsum = []; // last iteration gradients (used for momentum calculations)
		this.xsum = []; // used in adam or adadelta

		// check if regression is expected
		if (this.net.layers[this.net.layers.length - 1].layerType === "regression") {
			this.regression = true;
		}

		else {
			this.regression = false;
		}
	}

	/**
	 * @param x
	 * @param y
	 * @example
	 */
	train(x, y) {
		var start = new Date().getTime();
		this.net.forward(x, true); // also set the flag that lets the net know we're just training
		var end = new Date().getTime();
		const fwdTime = end - start;

		var start = new Date().getTime();
		const costLoss = this.net.backward(y);

		let l2_decayLoss = 0.0;
		let l1_decayLoss = 0.0;
		var end = new Date().getTime();
		const bwdTime = end - start;

		if (this.regression && y.constructor !== Array) {
			console.log("Warning: a regression net requires an array as training output vector.");
		}

		this.k++;

		if (this.k % this.batchSize === 0) {
			const pglist = this.net.getParamsAndGrads();

			// initialize lists for accumulators. Will only be done once on first iteration
			if (this.gsum.length === 0 && (this.method !== "sgd" || this.momentum > 0.0)) {
				// only vanilla sgd doesnt need either lists
				// momentum needs gsum
				// adagrad needs gsum
				// adam and adadelta needs gsum and xsum
				for (var i = 0; i < pglist.length; i++) {
					this.gsum.push(global.zeros(pglist[i].params.length));

					if (this.method === "adam" || this.method === "adadelta") {
						this.xsum.push(global.zeros(pglist[i].params.length));
					}
					else {
						this.xsum.push([]); // conserve memory
					}
				}
			}

			// perform an update for all sets of weights
			for (var i = 0; i < pglist.length; i++) {
				const pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
				const p = pg.params;
				const g = pg.grads;

				// learning rate for some parameters.
				const l2_decayMul = typeof pg.l2_decayMul !== "undefined" ? pg.l2_decayMul : 1.0;
				const l1_decayMul = typeof pg.l1_decayMul !== "undefined" ? pg.l1_decayMul : 1.0;
				const l2_decay = this.l2_decay * l2_decayMul;
				const l1_decay = this.l1_decay * l1_decayMul;

				const plen = p.length;

				for (let j = 0; j < plen; j++) {
					l2_decayLoss += l2_decay * p[j] * p[j] / 2; // accumulate weight decay loss
					l1_decayLoss += l1_decay * Math.abs(p[j]);
					const l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
					const l2grad = l2_decay * p[j];

					const gij = (l2grad + l1grad + g[j]) / this.batchSize; // raw batch gradient

					const gsumi = this.gsum[i];
					const xsumi = this.xsum[i];

					if (this.method === "adam") {
						// adam update
						gsumi[j] = gsumi[j] * this.beta1 + (1 - this.beta1) * gij; // update biased first moment estimate
						xsumi[j] = xsumi[j] * this.beta2 + (1 - this.beta2) * gij * gij; // update biased second moment estimate
						const biasCorr1 = gsumi[j] * (1 - this.beta1 ** this.k); // correct bias first moment estimate
						const biasCorr2 = xsumi[j] * (1 - this.beta2 ** this.k); // correct bias second moment estimate
						var dx = -this.learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
						p[j] += dx;
					}
					else if (this.method === "adagrad") {
						// adagrad update
						gsumi[j] = gsumi[j] + gij * gij;
						var dx = -this.learningRate / Math.sqrt(gsumi[j] + this.eps) * gij;
						p[j] += dx;
					}
					else if (this.method === "windowgrad") {
						// this is adagrad but with a moving window weighted average
						// so the gradient is not accumulated over the entire history of the run.
						// it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
						gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
						var dx = -this.learningRate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
						p[j] += dx;
					}
					else if (this.method === "adadelta") {
						gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
						var dx = -Math.sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
						xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
						p[j] += dx;
					}
					else if (this.method === "nesterov") {
						var dx = gsumi[j];
						gsumi[j] = gsumi[j] * this.momentum + this.learningRate * gij;
						dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
						p[j] += dx;
					}
					else {
						// assume SGD
						if (this.momentum > 0.0) {
							// momentum update
							var dx = this.momentum * gsumi[j] - this.learningRate * gij; // step
							gsumi[j] = dx; // back this up for next iteration of momentum
							p[j] += dx; // apply corrected gradient
						}
						else {
							// vanilla sgd
							p[j] += -this.learningRate * gij;
						}
					}

					g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
				}
			}
		}

		// appending softmaxLoss for backwards compatibility, but from now on we will always use costLoss
		// in future, TODO: have to completely redo the way loss is done around the network as currently
		// loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
		// and it should all be computed correctly and automatically.
		return {
			fwdTime,
			bwdTime,
			l2_decayLoss,
			l1_decayLoss,
			costLoss,
			softmaxLoss: costLoss,
			loss: costLoss + l1_decayLoss + l2_decayLoss
		};
	}
}

export default Trainer;
