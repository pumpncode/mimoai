import {
	sum,
	randomIndex
} from "./utilities.js";

import Window from "./brain/window.js";

import NeuralNetwork from "./brain/neural-network.js";
import Trainer from "./brain/trainer.js";

import Experience from "./brain/experience.js";

// A Brain object does all the magic.
// over time it receives some inputs and some rewards
// and its job is to set the outputs to maximize the expected reward
const Brain = class {
	/**
	 * In number of time steps, of temporal memory
	 * the ACTUAL input to the net will be (x,a) temporalWindow times, and followed by current x
	 * so to have no information from previous time step going into value function, set to 0.
	 */
	temporalWindow = 1

	/**
	 * Size of experience replay memory.
	 */
	experienceSize = 30000

	/**
	 * Number of examples in experience replay memory before we begin learning.
	 */
	startLearnThreshold = Math.floor(Math.min(this.experienceSize * 0.1, 1000))

	/**
	 * Gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1].
	 */
	gamma = 0.8

	/**
	 * Number of steps we will learn for.
	 */
	learningStepsTotal = 100000

	/**
	 * How many steps of the above to perform only random actions (in the beginning)?
	 */
	learningStepsBurnin = 3000

	/**
	 * What epsilon value do we bottom out on? 0.0 => purely deterministic policy at end.
	 */
	epsilonMin = 0.05

	/**
	 * What epsilon to use at test time? (i.e. When learning is disabled).
	 */
	epsilonTestTime = 0.01

	/**
	 * Advanced feature. Sometimes a random action should be biased towards some values
	 * for example in flappy bird, we may want to choose to not flap more often
	 * this better sum to 1 by the way, and be of length this.numActions.
	 */
	randomActionDistribution = []

	hiddenLayerSizes = []

	/**
	 * And finally we need a Temporal Difference Learning trainer!
	 */
	tdtrainerOptions = {
		batchSize: 64,
		l2_decay: 0.01,
		learningRate: 0.01,
		momentum: 0.0
	}

	// experience replay
	experience = [];

	/**
	 * Various housekeeping variables.
	 */

	/**
	 * Incremented every backward().
	 */
	age = 0;

	/**
	 * Incremented every forward().
	 */
	forwardPasses = 0;

	/**
	 * Controls exploration exploitation tradeoff. Should be annealed over time.
	 */
	epsilon = 1.0;

	latestReward = 0;

	lastInputArray = [];

	averageRewardWindow = new Window(1000, 10);

	averageLossWindow = new Window(1000, 10);

	learning = true;

	constructor(
		numStates,
		numActions,
		options
	) {
		Object.assign(this, options);

		if (this.randomActionDistribution.length !== 0) {
			if (this.randomActionDistribution.length !== numActions) {
				console.error("TROUBLE. randomActionDistribution should be same length as numActions.");
			}

			const distributionSum = sum(this.randomActionDistribution);

			if (Math.abs(distributionSum - 1.0) > 0.0001) {
				console.error("TROUBLE. randomActionDistribution should sum to 1!");
			}
		}



		// states that go into neural net to predict optimal action look as
		// x0,a0,x1,a1,x2,a2,...xt
		// this variable controls the size of that temporal window. Actions are
		// encoded as 1-of-k hot vectors
		this.netInputs = numStates * this.temporalWindow + numActions * this.temporalWindow + numStates;
		this.numStates = numStates;
		this.numActions = numActions;
		this.windowSize = Math.max(this.temporalWindow, 2); // must be at least 2, but if we want more context even more
		this.stateWindow = new Array(this.windowSize);
		this.actionWindow = new Array(this.windowSize);
		this.rewardWindow = new Array(this.windowSize);
		this.netWindow = new Array(this.windowSize);

		/**
		 * Create [state -> value of all possible actions] modeling net for the value function
		 * this is an advanced usage feature, because size of the input to the network, and number of
		 * actions must check out. This is not very pretty Object Oriented programming but I can't see
		 * a way out of it :(.
		 */
		if (options.layerDefs) {
			if (options.layerDefs.length < 2) {
				console.error("TROUBLE! must have at least 2 layers");
			}

			if (options.layerDefs[0].type !== "input") {
				console.error("TROUBLE! first layer must be input layer!");
			}

			if (options.layerDefs[options.layerDefs.length - 1].type !== "regression") {
				console.error("TROUBLE! last layer must be input regression!");
			}

			if (options.layerDefs[0].outDepth * options.layerDefs[0].outWidth * options.layerDefs[0].outHeight !== this.netInputs) {
				console.error("TROUBLE! Number of inputs must be numStates * temporalWindow + numActions * temporalWindow + numStates!");
			}

			if (options.layerDefs[options.layerDefs.length - 1].numNeurons !== this.numActions) {
				console.error("TROUBLE! Number of regression neurons should be numActions!");
			}
		}
		else {
			this.layerDefs = [
				{
					outDepth: this.netInputs,
					outWidth: 1,
					outHeight: 1,
					type: "input"
				},
				...this.hiddenLayerSizes.map((size) => ({
					activation: "relu",
					numNeurons: size,
					type: "fc"
				})),
				{
					numNeurons: numActions,
					type: "regression"
				}
			];
		}

		this.valueNet = new NeuralNetwork();
		this.valueNet.makeLayers(this.layerDefs);

		this.tdtrainer = new Trainer(this.valueNet, this.tdtrainerOptions);
	}

	/**
	 * @example
	 * a bit of a helper function. It returns a random action
	 * we are abstracting this away because in future we may want to
	 * do more sophisticated things. For example some actions could be more
	 * or less likely at "rest"/default state.
	 */
	randomAction() {
		if (this.randomActionDistribution.length === 0) {
			return randomIndex(0, this.numActions)
		}

		// okay, lets do some fancier sampling:
		const p = Math.random();

		let cumprob = 0;

		for (let index = 0; index < this.numActions; index++) {
			cumprob += this.randomActionDistribution[index];

			if (p < cumprob) {
				return index;
			}
		}
	}

	policy(state) {
		// compute the value of doing any action in this state
		// and return the argmax action and its value
		const svol = new Volume(1, 1, this.netInputs);
		svol.w = state;
		const actionValues = this.valueNet.forward(svol);

		let maxk = 0;
		let maxval = actionValues.w[0];

		for (let index = 1; index < this.numActions; index++) {
			if (actionValues.w[index] > maxval) {
				maxk = index;
				maxval = actionValues.w[index];
			}
		}

		return {
			action: maxk,
			value: maxval
		};
	}

	getNetInput(xt) {
		const {
			windowSize
		} = this;

		// return s = (x,a,x,a,x,a,xt) state vector.
		// It's a concatenation of last windowSize (x,a) pairs and current state x
		let resultWindow = [];
		resultWindow = resultWindow.concat(xt); // start with current state
		// and now go backwards and append states and actions from history temporalWindow times

		for (let windowIndex = 0; windowIndex < this.temporalWindow; windowIndex++) {
			// state
			resultWindow = resultWindow.concat(this.stateWindow[windowSize - 1 - windowIndex]);
			// action, encoded as 1-of-k indicator vector. We scale it up a bit because
			// we dont want weight regularization to undervalue this information, as it only exists once
			const action1ofk = new Array(this.numActions);

			for (let actionIndex = 0; actionIndex < this.numActions; actionIndex++) {
				action1ofk[actionIndex] = 0.0;
			}

			action1ofk[this.actionWindow[windowSize - 1 - windowIndex]] = Number(this.numStates);
			resultWindow = resultWindow.concat(action1ofk);
		}

		return resultWindow;
	}

	forward(inputArray) {
		// compute forward (behavior) pass given the input neuron signals from body
		this.forwardPasses += 1;
		this.lastInputArray = inputArray; // back this up

		// create network input
		let action = this.randomAction();

		let netInput = [];

		if (this.forwardPasses > this.temporalWindow) {
			// we have enough to actually do something reasonable
			netInput = this.getNetInput(inputArray);

			if (this.learning) {
				// compute epsilon for the epsilon-greedy policy
				this.epsilon = Math.min(1.0, Math.max(this.epsilonMin, 1.0 - (this.age - this.learningStepsBurnin) / (this.learningStepsTotal - this.learningStepsBurnin)));
			}
			else {
				this.epsilon = this.epsilonTestTime; // use test-time value
			}

			const rf = Math.random();

			if (rf >= this.epsilon) {
				// otherwise use our policy to make decision
				const maxact = this.policy(netInput);

				({
					action
				} = maxact);
			}
		}

		// remember the state and action we took for backward pass
		this.netWindow.shift();
		this.netWindow.push(netInput);
		this.stateWindow.shift();
		this.stateWindow.push(inputArray);
		this.actionWindow.shift();
		this.actionWindow.push(action);

		return action;
	}

	backward(reward) {
		this.latestReward = reward;
		this.averageRewardWindow.add(reward);
		this.rewardWindow.shift();
		this.rewardWindow.push(reward);

		if (!this.learning) {
			return;
		}

		// various book-keeping
		this.age += 1;

		// it is time t+1 and we have to store (sT, aT, rT, s_{t+1}) as new experience
		// (given that an appropriate number of state measurements already exist, of course)
		if (this.forwardPasses > this.temporalWindow + 1) {
			const {
				windowSize
			} = this;

			const experience = new Experience();

			experience.state0 = this.netWindow[windowSize - 2];
			experience.action0 = this.actionWindow[windowSize - 2];
			experience.reward0 = this.rewardWindow[windowSize - 2];
			experience.state1 = this.netWindow[windowSize - 1];

			if (this.experience.length < this.experienceSize) {
				this.experience.push(experience);
			}
			else {
				// replace. finite memory!
				const ri = randomIndex(0, this.experienceSize);
				this.experience[ri] = experience;
			}
		}

		// learn based on experience, once we have some samples to go on
		// this is where the magic happens...
		if (this.experience.length > this.startLearnThreshold) {
			let avcost = 0.0;

			for (let index = 0; index < this.tdtrainer.batchSize; index++) {
				const randomExperienceIndex = randomIndex(0, this.experience.length);
				const experience = this.experience[randomExperienceIndex];
				const x = new Volume(1, 1, this.netInputs);
				x.w = experience.state0;
				const maxact = this.policy(experience.state1);
				const r = experience.reward0 + this.gamma * maxact.value;
				const ystruct = {
					dim: experience.action0,
					val: r
				};
				const loss = this.tdtrainer.train(x, ystruct);
				avcost += loss.loss;
			}

			avcost /= this.tdtrainer.batchSize;
			this.averageLossWindow.add(avcost);
		}
	}

	visSelf(element) {
		element.innerHTML = ""; // erase elt first

		// elt is a DOM element that this function fills with brain-related information
		const brainvis = document.createElement("div");

		// basic information
		const desc = document.createElement("div");

		let text = "";
		text += `experience replay size: ${this.experience.length}<br>`;
		text += `exploration epsilon: ${this.epsilon}<br>`;
		text += `age: ${this.age}<br>`;
		text += `average Q-learning loss: ${this.averageLossWindow.getAverage()}<br />`;
		text += `smooth-ish reward: ${this.averageRewardWindow.getAverage()}<br />`;
		desc.innerHTML = text;
		brainvis.appendChild(desc);

		element.appendChild(brainvis);
	}
};

export default Brain;
