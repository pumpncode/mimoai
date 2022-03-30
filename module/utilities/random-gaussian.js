const randomGaussian = (from, to, n = 6) => {
	let s = 0;
	for (let i = 0; i < n; i++) {
		s += Math.random();
	}
	return ((s / n) * (to - from)) + from;
}

export default randomGaussian;