/**
 * @private
 */
const randomFromTo = (from = 0, to = 1) => (Math.random() * (to - from)) + from;

export default randomFromTo;
