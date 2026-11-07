// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// Finds people in images with macOS's built-in Vision framework, so there is no model to download.
// Usage: person_check IMAGE [IMAGE ...]
// Prints one line of JSON per image: {"path": ..., "grid": "0101...", "boxes": [{"kind": "face"|"body"|"upper",
// "x": ..., "y": ..., "w": ..., "h": ..., "confidence": ...}]} with x, y, w, h from 0 to 1 and y counted
// from the top. "grid" is person segmentation, GRID_COLUMNS by GRID_ROWS cells, top row first, 1 where there is a
// person: it finds people seen from behind or out of focus, which the boxes miss. An image that can't be read gets
// {"path": ..., "error": ...}.

import CoreVideo
import Foundation
import Vision

let GRID_COLUMNS = 192, GRID_ROWS = 108  // also in person.py

func personGrid(_ request: VNGeneratePersonSegmentationRequest) -> String {
    guard let buffer = request.results?.first?.pixelBuffer else {
        return String(repeating: "0", count: GRID_COLUMNS * GRID_ROWS)
    }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer)
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    var cells = [Character]()
    cells.reserveCapacity(GRID_COLUMNS * GRID_ROWS)
    for y in 0..<GRID_ROWS {
        for x in 0..<GRID_COLUMNS {
            cells.append(base[(y * h / GRID_ROWS) * stride + (x * w / GRID_COLUMNS)] > 127 ? "1" : "0")
        }
    }
    return String(cells)
}

func detect(_ path: String, into line: inout [String: Any]) throws {
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
    let faces = VNDetectFaceRectanglesRequest()
    let bodies = VNDetectHumanRectanglesRequest()
    bodies.upperBodyOnly = false
    let uppers = VNDetectHumanRectanglesRequest()
    uppers.upperBodyOnly = true
    let people = VNGeneratePersonSegmentationRequest()
    people.qualityLevel = .accurate
    try handler.perform([faces, bodies, uppers, people])
    var out: [[String: Any]] = []
    func add(_ kind: String, _ found: [VNDetectedObjectObservation]?) {
        for o in found ?? [] {
            let b = o.boundingBox  // Vision counts y from the bottom
            out.append(["kind": kind, "x": b.minX, "y": 1 - b.maxY, "w": b.width, "h": b.height,
                        "confidence": o.confidence])
        }
    }
    add("face", faces.results)
    add("body", bodies.results)
    add("upper", uppers.results)
    line["boxes"] = out
    line["grid"] = personGrid(people)
}

for path in CommandLine.arguments.dropFirst() {
    var line: [String: Any] = ["path": path]
    do {
        try detect(path, into: &line)
    } catch {
        line["error"] = String(describing: error)
    }
    let data = try! JSONSerialization.data(withJSONObject: line)
    print(String(data: data, encoding: .utf8)!)
}
