import randomFromTo from "./random-from-to.js";

/**
 * @private
 */
const randomIndex = (start = 0, length) => Math.floor(randomFromTo(start, length));

export default randomIndex;
