import { readAll } from "io";
import { bmvbhash } from "blockhash-core";
import Brain from "./module/brain.js";
import { sum, arithmeticMean } from "./module/utilities.js"

const {
	run,
	readTextFile,
	writeTextFile
} = Deno;

const {
	round,
	sqrt,
	SQRT2
} = Math;

const execute = async (programArguments) => {
	const command = run({
		cmd: programArguments,
		stdout: "piped",
		stdin: "piped"
	});

	const output = new TextDecoder().decode(await readAll(command.stdout));

	command.stdout?.close();

	command.close();

	return output;
};

const pixelWidth = 2;

const screenshotArea = async ({
	origin: {
		x,
		y
	},
	size: {
		width,
		height
	}
}) => {
	await execute([
		"screencapture",
		"-x",
		"-R",
		[x, y, width, height].join(","),
		"screen.png"
	])
}

const getCenter = ({
	origin: {
		x,
		y
	},
	size: {
		width,
		height
	}
}) => {
	return {
		x: round(x + (width / pixelWidth)),
		y: round(y + (height / pixelWidth))
	}
}

const windowData = JSON.parse(
	await execute(["python3", "lswin.py", "-s", "Mini Motorways"])
)[0];

const {
	contentRectangle,
	contentRectangle: {
		origin: {
			y: contentRectangleY
		},
		size: {
			height: contentRectangleHeight
		}
	}
} = windowData;

const center = getCenter(contentRectangle);

const centerBottom = {
	...center,
	y: round(
		(
			contentRectangleY + (contentRectangleHeight)
		) -
		(
			(contentRectangleHeight) / 20
		)
	)
}

const moveCursorTo = async ({ x, y }, relative = false) => {
	let safeX = x;
	let safeY = y;


	if (relative) {
		// cliclick needs a plus in front of numbers to know if they're relative
		safeX = x >= 0 ? `+${x}` : x;
		safeY = y >= 0 ? `+${y}` : y;
	}
	else {
		// absolute negative values have to be prefixed with "="
		safeX = x < 0 ? `=${x}` : x;
		safeY = y < 0 ? `=${y}` : y;
	}

	await execute(["cliclick", `m:${safeX},${safeY}`]);
}

const click = async () => {
	await execute(["cliclick", "c:."]);
}

const rightClick = async () => {
	await execute(["cliclick", "rc:."]);
}

const clickDown = async () => {
	await execute(["cliclick", "dd:."]);
}

const clickUp = async ({ x, y } = center) => {
	await execute(["cliclick", `du:${x},${y}`]);
}

const wait = async (ms = 1000, callback = async () => { }) => {
	await execute(["cliclick", `w:${ms}`]);

	await callback();
}

const dragCursorTo = async ({ x, y }) => {
	await clickDown();

	await execute(["cliclick", `dm:${x},${y}`, `du:${x},${y}`]);

	await clickUp({ x, y });
}

const pressSpaceBar = async () => {
	await execute(["cliclick", "kp:space"]);
}



const hashLength = 9;
const hashBits = 2 * sqrt(hashLength);

const compressCells = (grid, hashes = false) => {
	// const cellWidth = (grid[0][1].x - grid[0][0].x) / pixelWidth;

	// console.log(JSON.stringify(grid[6][1].pixels))
	// console.log(grid[6][1].pixels);

	if (hashes) {
		return grid
			.map(row => row
				.map(
					cell => {
						const data = new Uint8ClampedArray(
							cell.pixels.map(pixelRow => pixelRow.map(rgb => [...rgb, 255])).flat(2)
						);

						return parseInt(
							bmvbhash(
								{
									data,
									width: 64,
									height: 64
								},
								hashBits
							), 16);
					}
				)
			).flat(2);
	}

	return grid.map(
		row => row
			.map(
				({ pixels }) => sum(
					pixels
						.map(
							(pixelRow, pixelRowIndex) => sum(
								pixelRow
									.map(
										(pixel, pixelIndex) => parseInt(pixel.map(number => number.toString(16)).join(""), 16) * (pixelIndex + 1)
									)
							) * (pixelRowIndex + 1)
						)
				)
			)
	).flat()
}

const getGameData = async (parts = ["grid", "rides", "day"]) => {

	await moveCursorTo(centerBottom);

	await click();

	await clickDown();

	await wait(300);

	await screenshotArea(contentRectangle);

	await clickUp(centerBottom);

	const readGameOutput = await execute(["python3", "read-game.py", "--parts", ...parts]);

	return readGameOutput.startsWith("{") ? JSON.parse(readGameOutput) : readGameOutput;

	// const rgba = new Uint8ClampedArray(grid[6][1].pixels.map(row => row.map(rgb => [...rgb, 255])).flat(2));

	// const imageData = { data: rgba, width: grid.length, height: grid.length };

	// const hash = bmvbhash(imageData, 16);

	// console.log(hash);

	// // console.log(rgba.join(" "))
	// console.log(rgb.join(" "))

	// for (const row of grid) {
	// 	for (const { x, y } of row) {
	// 		const {
	// 			origin: {
	// 				x: xOrigin,
	// 				y: yOrigin
	// 			}
	// 		} = contentRectangle;

	// 		await moveCursorTo({
	// 			x: Math.round(xOrigin + (x / pixelWidth) + (cellWidth / 2)),
	// 			y: Math.round(yOrigin + (y / pixelWidth) + (cellWidth / 2))
	// 		})
	// 	}
	// }
}


const dayNumberFromString = (dayString) => {
	const sanitizedDayString = dayString.replace(/[^A-Za-z]/g, "").toLocaleLowerCase();

	return ["so", "mo", "di", "mi", "do", "fr", "sa"].indexOf(sanitizedDayString);
}

const maxGridWidth = 16;
const maxGridHeight = 10;

const minGridWidth = 13;
const minGridHeight = 6;

const maxGridSize = maxGridWidth * maxGridHeight;

const numberOfExtraInputs = 1;

const numberOfInputs = maxGridSize + numberOfExtraInputs;

const actionStrings = [
	"top",
	"topRight",
	"right",
	"bottomRight",
	"bottom",
	"bottomLeft",
	"left",
	"topLeft",
	"delete"
];

const extraActions = [["nothing", 0, 0]];

const actionsPerCell = actionStrings.length;

const numberOfExtraOutputs = extraActions.length;

const movesPerLoop = 1;

const numberOfOutputs = (maxGridSize * (actionsPerCell * movesPerLoop)) + numberOfExtraOutputs;

const brain = new Brain(numberOfInputs, numberOfOutputs, {
	hiddenLayerSizes: [50, 50],
	temporalWindow: 100
})

const actionTriples = [
	...Array(maxGridHeight)
		.fill()
		.map(
			(row, rowIndex) => Array(maxGridWidth)
				.fill()
				.map(
					(cell, cellIndex) => actionStrings
						.map(actionString => [actionString, rowIndex, cellIndex])
				)
		)
		.flat(2),
	...extraActions
]

const act = async (grid, actionString, y, x) => {
	if (y < grid.length && x < grid[0].length) {
		const {
			origin: {
				x: originX,
				y: originY
			}
		} = contentRectangle;

		const cell = grid[y][x];

		const {
			x: cellX,
			y: cellY
		} = cell;

		const cellWidth = (grid[0][1].x - grid[0][0].x);

		const cellXReal = cellX / pixelWidth;
		const cellYReal = cellY / pixelWidth;
		const cellWidthReal = cellWidth / pixelWidth;

		const cellCenter = {
			x: round(originX + cellXReal + (cellWidthReal / 2)),
			y: round(originY + cellYReal + (cellWidthReal / 2))
		};

		const top = { y: round(cellCenter.y - cellWidthReal) };
		const right = { x: round(cellCenter.x + cellWidthReal) };
		const bottom = { y: round(cellCenter.y + cellWidthReal) };
		const left = { x: round(cellCenter.x - cellWidthReal) };

		await moveCursorTo(cellCenter);

		switch (actionString) {
			case "top":
				await dragCursorTo({ ...cellCenter, ...top });
				break;
			case "topRight":
				await dragCursorTo({ ...cellCenter, ...top, ...right });
				break;
			case "right":
				await dragCursorTo({ ...cellCenter, ...right });
				break;
			case "bottomRight":
				await dragCursorTo({ ...cellCenter, ...bottom, ...right });
				break;
			case "bottom":
				await dragCursorTo({ ...cellCenter, ...bottom });
				break;
			case "bottomLeft":
				await dragCursorTo({ ...cellCenter, ...bottom, ...left });
				break;
			case "left":
				await dragCursorTo({ ...cellCenter, ...left });
				break;
			case "topLeft":
				await dragCursorTo({ ...cellCenter, ...top, ...left });
				break;
			case "delete":
				await moveCursorTo(cellCenter);
				await rightClick();
				break;

			default:
				break;
		}
	}
};

const isGameOver = (gameData) => {
	return typeof gameData === "string" && gameData.startsWith("GAME OVER");
};

const restartGame = async () => {

	const {
		size: {
			height
		}
	} = contentRectangle;

	await moveCursorTo({ x: center.x, y: round(center.y + height / 6) });
	await click();
}

const isUpgradeScreen = (gameData) => {
	return typeof gameData === "string" && gameData.startsWith("Woche");
};

const clickUpgrade = async () => {
	const {
		size: {
			height,
			width
		}
	} = contentRectangle;

	await moveCursorTo({ x: round(center.x - width / 7), y: center.y });
	await click();
}

let gamePaused = false;

const togglePause = async () => {
	await moveCursorTo(centerBottom);

	await click();

	await pressSpaceBar();

	gamePaused = !gamePaused;
}

let weekNumber = 0;

let now = performance.now();

const actionsPerLoop = 50;

await moveCursorTo(center);
await click(center);

const rewards = []

for (let index = 0; index < 1000; index++) {
	await wait(10000);

	await togglePause();

	const gameData = await getGameData(["grid", "day", "rides", "roads"]);

	if (isGameOver(gameData)) {
		brain.backward(-100);
		now = 0;
		await restartGame();
	}
	else if (isUpgradeScreen(gameData)) {
		weekNumber = Number(gameData.replace(/^Woche /, ""));
		await clickUpgrade();
	}
	else {
		const {
			grid,
			grid: {
				length: gridHeight,
				0: {
					length: gridWidth
				}
			},
			day,
			rides,
			roads
		} = gameData;

		if (gridHeight >= minGridHeight && gridWidth >= minGridWidth && gridHeight <= maxGridHeight && gridWidth <= maxGridWidth) {
			const compressedCells = compressCells(grid, true);

			const dayNumber = dayNumberFromString(day);

			// const inputs = [dayNumber, ...compressedCells, ...Array(maxGridSize - compressedCells.length).fill(0)];
			const inputs = [dayNumber, ...compressedCells];

			for (let actionIndex = 0; actionIndex < actionsPerLoop; actionIndex++) {
				const actionNumber = brain.forward(inputs);

				const actionTriple = actionTriples[actionNumber];

				await act(grid, ...actionTriple);

				console.log(actionTriple)
			}

			const durationInSeconds = (performance.now() - now) * 1000;

			const ridesNumber = Number(rides.replace(/O/g, "0").replace(/[^\d]/g, ""));

			const ridesPerSecond = ridesNumber / durationInSeconds;

			const roadsNumber = Number(roads.replace(/O/g, "0").replace(/[^\d]/g, ""));

			const normalizedRoadReward = (10 - (9 * roadsNumber) / 500) / (weekNumber + 1);

			const reward = (ridesPerSecond + 1) * normalizedRoadReward;

			rewards.push(reward);

			const averageRewards = arithmeticMean(rewards);

			console.table({
				gridSize: `[${grid[0].length},${grid.length}]`,
				ridesPerSecond,
				roads: roadsNumber,
				week: weekNumber,
				reward,
				averageRewards
			})

			brain.backward(reward);
		}
	}

	if (index % 10 === 0) {
		console.log("saving network");
		await writeTextFile("./network.json", JSON.stringify(brain.valueNet.toJSON()));

		console.log("done saving network");
	}

	await wait(1000);

	await togglePause();
}