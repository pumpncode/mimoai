import sum from "./sum.js";

const arithmeticMean = (numbers) => {
	return sum(numbers) / numbers.length;
}

export default arithmeticMean;