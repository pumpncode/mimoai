'''
Detects a grid.
'''
import ujson
import cv2
from math import floor
from operator import itemgetter
from argparse import ArgumentParser
from pytesseract import image_to_string

parser = ArgumentParser(description = 'Detect grid.')
parser.add_argument(
	"-f",
	"--filename",
	dest = "filename",
	help = "filename prefix of images"
)
parser.add_argument(
	"-d",
	"--dev",
	dest = "dev",
	help = "run in development mode (preview image)",
	action = "store_true"
)
parser.add_argument(
	"--parts",
	dest = "parts",
	help = "which parts to read",
	nargs = "+",
	default = ["grid", "rides", "day", "roads"]
)

args = parser.parse_args()
dev = args.dev
parts = args.parts

image = cv2.imread("screen.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


(imageHeight, imageWidth, channels) = image.shape

def hasTheRightAspectRatio(boundingRect):
	[x, y, w, h] = boundingRect

	return round((w / h) * 10) == 12 or round((h / w) * 10) == 12

def isBigEnough(boundingRect):
	[x, y, w, h] = boundingRect

	return w > round(imageHeight / 30) and w < round(imageHeight / 5)

def topIsBrighterThanBottom(boundingRect, image):
	[x, y, w, h] = boundingRect

	centerX = x + (w / 2)
	centerY = y + (h / 2)

	centerYAbove = centerY - (h / 6)
	centerYBelow = centerY + (h / 6)

	pixelAbove = image[round(centerYAbove), round(centerX)]
	pixelBelow = image[round(centerYBelow), round(centerX)]

	return sum(pixelAbove) > sum(pixelBelow)

def isAHouse(boundingRect, image):
	return (
		hasTheRightAspectRatio(boundingRect) and
		isBigEnough(boundingRect) and
		topIsBrighterThanBottom(boundingRect, image)
	)

def drawBoundingRects(image, contours):
	for contour in contours:
		[contourX, contourY, w, h] = contour
		point1 = (contourX, contourY)
		point2 = (contourX + w, contourY + h)

		cv2.rectangle(image, point1, point2, (0, 0, 255), 2)

def cellFromBuilding(contour, offset, scale):
	[contourX, contourY, w, h] = contour
	[offsetX, offsetY] = offset
	[scaleX, scaleY] = scale

	newX = (contourX - (w * offsetX))
	newY = (contourY - (h * offsetY))
	newW = scale * w

	return {"x": newX, "y": newY, "width": newW}

def topLeftCellFromSample(cell, imageWidth, imageHeight, borders):
	x, y, width = itemgetter("x", "y", "width")(cell)

	topBorder, leftBorder = borders

	widthPadding = (imageWidth * leftBorder)
	heightPadding = (imageHeight * topBorder)

	usableWidth = x - widthPadding
	usableHeight = y - heightPadding

	newX = x - (floor(usableWidth / width) * width)

	newY = y - (floor(usableHeight / width) * width)

	return {"x": newX, "y": newY, "width": width}

def bottomRightCellFromSample(cell, imageWidth, imageHeight, borders):
	x, y, width = itemgetter("x", "y", "width")(cell)

	bottomBorder, rightBorder = borders

	remainingWidth = imageWidth - x - width
	remainingHeight = imageHeight - y - width

	widthPadding = (imageWidth * rightBorder)
	heightPadding = (imageHeight * bottomBorder)

	usableWidth = remainingWidth - widthPadding
	usableHeight = remainingHeight - heightPadding

	newX = x + ((floor(usableWidth / width)) * width)

	newY = y + ((floor(usableHeight / width)) * width)

	return {"x": newX, "y": newY, "width": width}

def getCells(contour, cellSettings, borderSettings):
	[contourX, contourY, w, h] = contour

	if w < h:
		w, h = h, w

	[offset, scale] = cellSettings

	sampleCell = cellFromBuilding([contourX, contourY, w, h], offset, scale)

	[
		topBorder,
		bottomBorder,
		leftBorder,
		rightBorder,
	] = borderSettings

	topLeftBorder = [topBorder, leftBorder]

	topLeftCell = topLeftCellFromSample(sampleCell, imageWidth, imageHeight, topLeftBorder)

	bottomRightBorder = [bottomBorder, rightBorder]

	bottomRightCell = bottomRightCellFromSample(sampleCell, imageWidth, imageHeight, bottomRightBorder)

	return [sampleCell, topLeftCell, bottomRightCell]

def drawCell(image, cell):
	x, y, width = itemgetter("x", "y", "width")(cell)
	point1 = tuple(map(round, (x, y)))
	point2 = tuple(map(round, (x + width, y + width)))

	cv2.rectangle(image, point1, point2, (0, 0, 255), 2)

def drawAll(image, cellSettings, borderSettings, contours):
	for contour in contours:
		cells = getCells(contour, cellSettings, borderSettings)
		for cell in cells:
			drawCell(image, cell)

def arithmeticMean(numbers):
	return sum(numbers) / len(numbers)

def reciprocal(number):
	return 1 / number

def reciprocalSum(numbers):
	return sum(list(map(reciprocal, numbers)))

def harmonicMean(numbers):
	return len(numbers) / reciprocalSum(numbers)

def selectPart(collection, i):
	return list(map(lambda c: c[i], collection))

def getAverageCells(cells):
	keys = ["x", "y", "width"]
	indices = [1, 2]

	if len(cells) == 0:
		return [
			{
				"x": 1,
				"y": 1,
				"width": 10
			},
			{
				"x": 1,
				"y": 1,
				"width": 10
			}
		]

	return [
		{
			key: harmonicMean(selectPart(selectPart(cells, i), key)) for key in keys
		} for i in indices
	]

def drawAverage(image, cellSettings, borderSettings, contours):
	cells = list(map(lambda c: getCells(c, cellSettings, borderSettings), contours))

	averageCells = getAverageCells(cells)

	# print(cells)
	# print(averageCells)

	# drawCell(image, cells[0]) # sampleCell
	drawCell(image, averageCells[0]) # average topLeftCell
	drawCell(image, averageCells[1]) # average bottomRightCell

def getGrid(topLeftCell, bottomRightCell):
	topLeftX, topLeftY, width = itemgetter("x", "y", "width")(topLeftCell)
	bottomRightX, bottomRightY = itemgetter("x", "y", "width")(bottomRightCell)[:2]

	gridWidth = (bottomRightX + width) - topLeftX
	gridHeight = (bottomRightY + width) - topLeftY

	return list(
		map(
			lambda y: list(
				map(
					lambda x: {"x": round(x), "y": round(y), "width": round(width)},
					list(range(round(topLeftX), round(gridWidth + width), round(width)))
				)
			),
			list(range(round(topLeftY), round(gridHeight + width), round(width)))
		)
	)

def drawGrid(image, grid):
	for row in grid:
		for cell in row:
			drawCell(image, cell)

def linearConversion(oldRange, newRange, value):
	[oldMin, oldMax] = oldRange
	[newMin, newMax] = newRange

	return (((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin


sliderMax = 100
windowTitle = "slider"

defaultOffsetX = 120 / 1000
defaultOffsetY = 120 / 1000
defaultScaleX = 660 / 1000
defaultScaleY = 660 / 1000
defaultTopBorder = linearConversion([480, 1050], [15, 50], imageHeight) / 1000
defaultBottomBorder = linearConversion([480, 1050], [120, 140], imageHeight) / 1000
defaultLeftBorder = 30 / 1000
defaultRightBorder = 30 / 1000

borders = [
	["top", defaultTopBorder],
	["bottom", defaultBottomBorder],
	["left", defaultLeftBorder],
	["right", defaultRightBorder]
]

borderTrackbarNamesAndDefaults = list(map(lambda d: [f"{d[0]}Border", d[1]], borders))

def getSettings():
	[
		offsetX,
		offsetY,
		scaleX,
		scaleY,
		topBorder,
		bottomBorder,
		leftBorder,
		rightBorder
	] = [
		defaultOffsetX,
		defaultOffsetY,
		defaultScaleX,
		defaultScaleY,
		defaultTopBorder,
		defaultBottomBorder,
		defaultLeftBorder,
		defaultRightBorder
	]

	cellSettings = [[offsetX, offsetY], [scaleX, scaleY]]

	borderSettings = [topBorder, bottomBorder, leftBorder, rightBorder]

	return [cellSettings, borderSettings]

def cropCell(image, cell):
	x, y, width = itemgetter("x", "y", "width")(cell)

	crop = image.copy()[y:y + width, x:x + width]

	crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

	(cropHeight, cropWidth, cropChannels) = crop.shape

	targetWidth = 64

	crop = cv2.resize(
		crop,
		(targetWidth, targetWidth),
		cv2.INTER_AREA if cropWidth > targetWidth else cv2.INTER_LINEAR
	)

	return crop.tolist()

def cropRides(image):
	return image.copy()[
		round(imageHeight / 40):round(imageHeight / 15),
		round(imageWidth - (imageWidth / 4)):round(imageWidth - imageWidth / 8.7)
	]

def getRides(image):
	tesseractConfig = ["eng", "--psm 7 --oem 1"]

	croppedRides = cropRides(image)

	(ret, croppedRides) = cv2.threshold(croppedRides, 127, 255, cv2.THRESH_TOZERO)

	croppedRides = ~croppedRides

	return image_to_string(croppedRides, *tesseractConfig).replace("\n", "").replace("\f", "")

def cropDay(image):
	return image.copy()[
		round(imageHeight / 40):round(imageHeight / 15),
		round(imageWidth - (imageWidth / 12)):round(imageWidth - (imageWidth / 22))
	]

def getDay(image):
	tesseractConfig = ["eng", "--psm 7 --oem 1"]

	croppedDay = cropDay(image)

	(ret, croppedDay) = cv2.threshold(croppedDay, 127, 255, cv2.THRESH_TOZERO)

	croppedDay = ~croppedDay

	return image_to_string(croppedDay, *tesseractConfig).replace("\n", "").replace("\f", "")

def cropRoads(image):
	return image.copy()[
		round(imageHeight - (imageHeight / 16)):round(imageHeight - (imageHeight / 21.5)),
		round((imageWidth / 2) + (imageWidth / 65)):round((imageWidth / 2) + (imageWidth / 33))
	]

def getRoads(image):
	tesseractConfig = ["eng", "--psm 7 --oem 1"]

	croppedRoads = cropRoads(image)

	(ret, croppedRoads) = cv2.threshold(croppedRoads, 127, 255, cv2.THRESH_TOZERO)

	croppedRoads = ~croppedRoads

	return image_to_string(croppedRoads, *tesseractConfig).replace("\n", "").replace("\f", "")

def cropGameOver(image):
	return image.copy()[
		round((imageHeight / 2) - (imageHeight / 3)):round((imageHeight / 2) - (imageHeight / 4)),
		round((imageWidth / 2) - (imageWidth / 7)):round((imageWidth / 2) + (imageWidth / 7))
	]

def getGameOver(image):
	tesseractConfig = ["eng", "--psm 7 --oem 1"]

	croppedGameOver = cropGameOver(image)

	(ret, croppedGameOver) = cv2.threshold(croppedGameOver, 127, 255, cv2.THRESH_TOZERO)

	croppedGameOver = ~croppedGameOver

	return image_to_string(croppedGameOver, *tesseractConfig).replace("\n", "").replace("\f", "")

def cropUpgrade(image):
	return image.copy()[
		round(
			(imageHeight / 2) - (imageHeight / 4.5)
		):
		round(
			(imageHeight / 2) - (imageHeight / 7)
		),
		round(
			(imageWidth / 2) - (imageWidth / 4.5)
		):
		round(
			(imageWidth / 2) + (imageWidth / 10)
		)
	]

def getUpgrade(image):
	tesseractConfig = ["eng", "--psm 7 --oem 1"]

	croppedUpgrade = cropUpgrade(image)

	(ret, croppedUpgrade) = cv2.threshold(croppedUpgrade, 127, 255, cv2.THRESH_TOZERO)

	croppedUpgrade = ~croppedUpgrade

	return image_to_string(croppedUpgrade, *tesseractConfig).replace("\n", "").replace("\f", "")

def detectGrid(value=0):
	[cellSettings, borderSettings] = getSettings()

	cells = list(map(lambda c: getCells(c, cellSettings, borderSettings), contours))

	averageCells = getAverageCells(cells)

	grid = getGrid(*averageCells)

	if dev:
		print(cellSettings)
		print(borderSettings)

		image = cv2.imread("screen.png")

		# drawGrid(image, grid)
		drawBoundingRects(image, contours)

		[cellSettings, borderSettings] = getSettings()

		drawAverage(image, cellSettings, borderSettings, contours)

		drawAll(image, cellSettings, borderSettings, contours)

	return grid

def getContours(image):
	div = 32
	quantized = image // div * div + div // 2
	hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)

	hsvThreshold = cv2.inRange(hsv, (0, 100, 150), (255, 255, 255))

	blurred = cv2.GaussianBlur(hsvThreshold, (3, 3), 0)
	canny = cv2.Canny(blurred, 120, 255, 1)

	# Find contours
	contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	contours = list(map(cv2.boundingRect, contours))

	contours = list(filter(lambda b: isAHouse(b, quantized), list(map(list, contours))))

	return contours

if dev:
	contours = getContours(image)

	cv2.namedWindow(windowTitle)

	grid = detectGrid()

	drawGrid(image, grid)
	drawBoundingRects(image, contours)

	[cellSettings, borderSettings] = getSettings()

	drawAverage(image, cellSettings, borderSettings, contours)

	drawAll(image, cellSettings, borderSettings, contours)

	cv2.imwrite(f"./images/test/{args.filename}.png", image)

else:
	gameOverString = getGameOver(gray)
	upgradeString = getUpgrade(gray)

	if gameOverString == "Game Over":
		print("GAME OVER")
	elif upgradeString.startswith("Woche"):
		print(upgradeString)
	else:
		if "grid" in parts:
			contours = getContours(image)

			[cellSettings, borderSettings] = getSettings()

			cells = list(map(lambda c: getCells(c, cellSettings, borderSettings), contours))

			grid = getGrid(*getAverageCells(cells))

			gridData = list(
				map(
					lambda row: list(
						map(
							lambda cell: {key: cell[key] for key in ["x", "y"]} | {"pixels": cropCell(image, cell)},
							row
						)
					),
					grid
				)
			)
		else:
			gridData = None

		if "rides" in parts:
			rides = getRides(gray)
		else:
			rides = None

		if "day" in parts:
			day = getDay(gray)
		else:
			day = None

		if "roads" in parts:
			roads = getRoads(gray)
		else:
			roads = None

		data = {
			"grid": gridData,
			"rides": rides,
			"day": day,
			"roads": roads
		}

		print(ujson.dumps(data))
