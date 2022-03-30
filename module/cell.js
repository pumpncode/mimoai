const defaultEnvironment = {
	north: null,
	northEast: null,
	east: null,
	southEast: null,
	south: null,
	southWest: null,
	west: null,
	northWest: null
}

/**
 * @author nnmrts <nanomiratus@gmail.com>
 * @date 2021-09-09
 * @class Cell
 */
const Cell = class {

	constructor(
		{
			type = "empty",
			cursor = "inactive",
			directions = [],
			cars: {
				parked = 0,
				arriving = 0,
				departing = 0
			} = {
				parked: 0,
				arriving: 0,
				departing: 0
			},
			environment: {
				center = "land",
				north = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				northEast = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				east = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				southEast = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				south = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				southWest = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				west = new Cell({ environment: { center: "land", ...defaultEnvironment } }),
				northWest = new Cell({ environment: { center: "land", ...defaultEnvironment } })

			},
			color,
			direction
		} = {
				type: "empty",
				cursor: "inactive",
				directions: [],
				cars: {
					parked: 0,
					arriving: 0,
					departing: 0
				},
				environment: {
					center: "land",
					...defaultEnvironment
				},
				color: null,
				direction: null
			}
	) {
		Object.assign(
			this,
			{
				type,
				cursor,
				directions,
				cars: {
					parked,
					arriving,
					departing
				},
				environment: {
					north,
					northEast,
					east,
					southEast,
					south,
					southWest,
					west,
					northWest
				},
				color,
				direction
			}
		);
	}

	toString() {
		return "0";
	}
}

export default Cell;