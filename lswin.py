'''
Lists all windows with their coordinates and dimensions in JSON format.
'''

import json
from argparse import ArgumentParser

import Quartz
from AppKit import (
	NSWindow,
	NSClosableWindowMask as closable,
	NSResizableWindowMask as resizable,
	NSMiniaturizableWindowMask as miniaturizable,
	NSTitledWindowMask as titled
)

parser = ArgumentParser(description = 'Get list of windows.')
parser.add_argument(
	"-f",
	"--filter",
	dest = "filter",
	help = "filter windows by title"
)
parser.add_argument(
	"-s",
	"--filter-sub",
	dest = "filterSub",
	help = "filter windows by subtitle"
)

args = parser.parse_args()

wl = Quartz.CGWindowListCopyWindowInfo(
	Quartz.kCGWindowListOptionAll,
	Quartz.kCGNullWindowID
)

# print(wl)

wl = sorted(wl, key = lambda k: k.valueForKey_('kCGWindowOwnerPID'))

# print(wl)

def toDict(window):
	"""
	does stuff
	"""
	pid = int(window.valueForKey_('kCGWindowOwnerPID') or '?')
	windowId = int(window.valueForKey_('kCGWindowNumber') or '?')
	x = int(window.valueForKey_('kCGWindowBounds').valueForKey_('X'))
	y = int(window.valueForKey_('kCGWindowBounds').valueForKey_('Y'))
	width = int(window.valueForKey_('kCGWindowBounds').valueForKey_('Width'))
	height = int(window.valueForKey_('kCGWindowBounds').valueForKey_('Height'))
	title = (window.valueForKey_('kCGWindowOwnerName') or '')
	subtitle = (
		'' if window.valueForKey_('kCGWindowName') is None else
		(window.valueForKey_('kCGWindowName') or '')
	)

	frameRectangle = {
		"origin": {
			"x": x,
			"y": y
		},
		"size": {
			"width": width,
			"height": height
		}
	}

	windowMask = (closable and resizable and miniaturizable and titled)

	contentRectangle = {
		k: window._asdict()
		for k,
		window in NSWindow.contentRectForFrameRect_styleMask_(
			list(
				Quartz.CGRectMakeWithDictionaryRepresentation(
					window.valueForKey_('kCGWindowBounds'),
					None
				)
			)[1],
			windowMask
		)._asdict().items()
	}

	contentRectangle["origin"]["y"] = (
		contentRectangle["origin"]["y"] +
		(frameRectangle["size"]["height"] - contentRectangle["size"]["height"])
	)

	windowDict = {
		"pid": pid,
		"windowId": windowId,
		"title": title,
		"subtitle": subtitle,
		"contentRectangle": contentRectangle,
		"frameRectangle": frameRectangle
	}

	return windowDict

windowList = list(map(toDict, wl))

if args.filter is not None:
	windowList = list(filter(lambda d: d["title"] == args.filter, windowList))

if args.filterSub is not None:
	windowList = list(
		filter(lambda d: d["subtitle"] == args.filterSub,
									windowList)
	)

print(json.dumps(windowList))
