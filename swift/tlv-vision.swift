// tlv-vision: Apple Vision and SoundAnalysis for tlv. A JSON job comes in on stdin; results go
// out as one JSON line each, in input order, so the caller can stream them.
//
//   tlv-vision images  {"paths": [image path], "text": bool, "detect": bool}
//       -> {"i", "text"?, "faces"?, "bodies"?} per image
//   tlv-vision video   {"path": video path, "samples": [{"t": seconds, "text": bool, "detect": bool}]}
//       -> {"t", "frame", "text"?, "faces"?, "bodies"?} per sample; samples sorted by t, and each
//          reads the frame on screen at t, decoded at full resolution
//   tlv-vision sounds  {"path": audio path, "window": s, "overlap": 0..1, "top": n, "min_conf": c}
//       -> {"start", "end", "labels": [{"label", "conf"}]} per classifier window
//
// Boxes are [x, y, w, h] and points [x, y], normalized to the upright image with a top-left
// origin; Vision's own origin is bottom-left, so every coordinate is flipped here.

import AVFoundation
import CoreMedia
import Foundation
import SoundAnalysis
import Vision

struct Failure: Error, CustomStringConvertible {
    let description: String
}

struct ImagesJob: Decodable {
    let paths: [String]
    let text: Bool
    let detect: Bool
}

struct Sample: Decodable {
    let t: Double
    let text: Bool
    let detect: Bool
}

struct VideoJob: Decodable {
    let path: String
    let samples: [Sample]
}

struct SoundsJob: Decodable {
    let path: String
    let window: Double
    let overlap: Double
    let top: Int
    let min_conf: Double
}

struct Line: Encodable {
    let text: String
    let conf: Double
    let box: [Double]
}

struct Face: Encodable {
    let box: [Double]
    let conf: Double
    let landmarks: [[Double]]?
    let roll: Double?
    let yaw: Double?
    let quality: Double?
}

struct Body: Encodable {
    let box: [Double]
    let conf: Double
}

struct Seen: Encodable {
    var i: Int?
    var t: Double?
    var frame: Double?
    var text: [Line]?
    var faces: [Face]?
    var bodies: [Body]?
}

struct Label: Encodable {
    let label: String
    let conf: Double
}

struct Window: Encodable {
    let start: Double
    let end: Double
    let labels: [Label]
}

func emit<T: Encodable>(_ value: T) throws {
    var line = try JSONEncoder().encode(value)
    line.append(0x0A)
    FileHandle.standardOutput.write(line)
}

func round4(_ value: Double) -> Double {
    (value * 10000).rounded() / 10000
}

/// A Vision rectangle (bottom-left origin) as [x, y, w, h] with a top-left origin.
func topLeft(_ rect: CGRect) -> [Double] {
    [rect.minX, 1 - rect.maxY, rect.width, rect.height].map { round4(Double($0)) }
}

/// The five alignment points of a face: the eye with the smaller x, the other eye, the nose
/// tip, then the mouth corner with the smaller x and the other. Eyes are the pupils, or the eye
/// outline's centroid when Vision has no pupil; the nose tip is the lowest nose point.
func fivePoints(_ face: VNFaceObservation) -> [[Double]]? {
    guard let marks = face.landmarks else { return nil }
    let box = face.boundingBox
    func points(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
        (region?.normalizedPoints ?? []).map {
            CGPoint(x: box.minX + $0.x * box.width, y: 1 - (box.minY + $0.y * box.height))
        }
    }
    func centroid(_ found: [CGPoint]) -> CGPoint? {
        found.isEmpty
            ? nil
            : CGPoint(
                x: found.map(\.x).reduce(0, +) / CGFloat(found.count),
                y: found.map(\.y).reduce(0, +) / CGFloat(found.count))
    }
    let lips = points(marks.outerLips)
    guard let one = centroid(points(marks.leftPupil)) ?? centroid(points(marks.leftEye)),
        let two = centroid(points(marks.rightPupil)) ?? centroid(points(marks.rightEye)),
        let nose = (points(marks.nose) + points(marks.noseCrest)).max(by: { $0.y < $1.y }),
        let mouthLeft = lips.min(by: { $0.x < $1.x }),
        let mouthRight = lips.max(by: { $0.x < $1.x })
    else { return nil }
    let eyes = [one, two].sorted { $0.x < $1.x }
    return [eyes[0], eyes[1], nose, mouthLeft, mouthRight].map { [round4(Double($0.x)), round4(Double($0.y))] }
}

/// People in the frame. Body pose finds the small players of a wide shot, which Vision's
/// person rectangles miss; each is boxed around the joints Vision is fairly sure of, padded for
/// the head and limbs the joints stop short of. Upper-body rectangles add the close-ups pose
/// misses. (Vision's full-body rectangles found nobody in broadcast frames, close-ups included.)
func people(_ poses: [VNHumanBodyPoseObservation], _ uppers: [VNHumanObservation]) throws -> [Body] {
    var found: [(box: CGRect, conf: Double)] = []
    for pose in poses {
        let joints = try pose.recognizedPoints(.all).values.filter { $0.confidence > 0.3 }
        guard joints.count >= 6 else { continue }
        let xs = joints.map(\.location.x)
        let ys = joints.map(\.location.y)
        let tight = CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
        let pad = tight.height * 0.1
        // Vision's y points up: one pad below the lowest joint, two above the highest for the head.
        let box = CGRect(x: tight.minX - pad, y: tight.minY - pad, width: tight.width + 2 * pad, height: tight.height + 3 * pad)
            .intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        found.append((box, joints.map { Double($0.confidence) }.reduce(0, +) / Double(joints.count)))
    }
    for upper in uppers where !found.contains(where: { overlap($0.box, upper.boundingBox) > 0.5 }) {
        found.append((upper.boundingBox, Double(upper.confidence)))
    }
    return found.map { Body(box: topLeft($0.box), conf: round4($0.conf)) }
}

/// The shared area of two rectangles as a fraction of the smaller one.
func overlap(_ a: CGRect, _ b: CGRect) -> Double {
    let common = a.intersection(b)
    if common.isNull { return 0 }
    return Double(common.width * common.height / min(a.width * a.height, b.width * b.height))
}

/// The requests run on every image; built once and reused, as Vision allows.
struct Looker {
    let text = VNRecognizeTextRequest()
    let faces = VNDetectFaceLandmarksRequest()
    let quality = VNDetectFaceCaptureQualityRequest()
    let poses = VNDetectHumanBodyPoseRequest()
    let uppers = VNDetectHumanRectanglesRequest()

    init() {
        text.recognitionLevel = .accurate
        // Score bugs and jersey numbers are not dictionary words.
        text.usesLanguageCorrection = false
        // A broadcast score bug is about 1/40 of the frame height; Vision's default skips text under 1/32.
        text.minimumTextHeight = 0.012
    }

    func look(_ handler: VNImageRequestHandler, text wantText: Bool, detect: Bool) throws -> Seen {
        var seen = Seen()
        try handler.perform((wantText ? [text] : []) + (detect ? [faces, poses, uppers] : []))
        if wantText {
            seen.text = (text.results ?? []).compactMap { line in
                line.topCandidates(1).first.map {
                    Line(text: $0.string, conf: round4(Double($0.confidence)), box: topLeft(line.boundingBox))
                }
            }
        }
        if detect {
            let found = faces.results ?? []
            var qualities: [UUID: Double] = [:]
            if !found.isEmpty {
                quality.inputFaceObservations = found
                try handler.perform([quality])
                for face in quality.results ?? [] {
                    if let score = face.faceCaptureQuality { qualities[face.uuid] = round4(Double(score)) }
                }
            }
            seen.faces = found.map { face in
                Face(
                    box: topLeft(face.boundingBox), conf: round4(Double(face.confidence)), landmarks: fivePoints(face),
                    roll: face.roll.map { round4($0.doubleValue) }, yaw: face.yaw.map { round4($0.doubleValue) },
                    quality: qualities[face.uuid])
            }
            seen.bodies = try people(poses.results ?? [], uppers.results ?? [])
        }
        return seen
    }
}

func runImages(_ job: ImagesJob) throws {
    let looker = Looker()
    for (i, path) in job.paths.enumerated() {
        let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
        var seen = try looker.look(handler, text: job.text, detect: job.detect)
        seen.i = i
        try emit(seen)
    }
}

/// How a track's display transform turns its stored frames upright.
func orientation(_ transform: CGAffineTransform) -> CGImagePropertyOrientation {
    switch (transform.a, transform.b, transform.c, transform.d) {
    case (0, 1, -1, 0): return .right
    case (0, -1, 1, 0): return .left
    case (-1, 0, 0, -1): return .down
    default: return .up
    }
}

func runVideo(_ job: VideoJob) async throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: job.path))
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw Failure(description: "\(job.path) has no video track")
    }
    let upright = orientation(try await track.load(.preferredTransform))
    let reader = try AVAssetReader(asset: asset)
    let frames = reader.outputProvider(
        for: AVAssetReaderTrackOutput(
            track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]))
    try reader.start()
    let looker = Looker()
    var next = 0
    var shown: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?
    func look(at sample: Sample, in frame: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>) throws {
        var seen = try frame.withUnsafeSampleBuffer { buffer in
            guard let pixels = buffer.imageBuffer else {
                throw Failure(description: "a decoded sample of \(job.path) has no image")
            }
            let handler = VNImageRequestHandler(cvPixelBuffer: pixels, orientation: upright, options: [:])
            return try looker.look(handler, text: sample.text, detect: sample.detect)
        }
        seen.t = sample.t
        seen.frame = round4(frame.presentationTimeStamp.seconds)
        try emit(seen)
    }
    // Frames arrive in display order; the frame on screen at t is the last one starting at or before t.
    while next < job.samples.count, let frame = try await frames.next() {
        while next < job.samples.count, job.samples[next].t < frame.presentationTimeStamp.seconds {
            try look(at: job.samples[next], in: shown ?? frame)
            next += 1
        }
        shown = frame
    }
    if let last = shown {
        while next < job.samples.count {
            try look(at: job.samples[next], in: last)
            next += 1
        }
    }
    reader.cancelReading()
}

final class SoundWindows: NSObject, SNResultsObserving {
    let top: Int
    let minConf: Double
    var failure: Error?

    init(top: Int, minConf: Double) {
        self.top = top
        self.minConf = minConf
    }

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard failure == nil, let result = result as? SNClassificationResult else { return }
        let labels = result.classifications
            .filter { $0.confidence >= minConf }
            .sorted { $0.confidence > $1.confidence }
            .prefix(top)
            .map { Label(label: $0.identifier, conf: round4($0.confidence)) }
        let start = result.timeRange.start.seconds
        do {
            try emit(Window(start: round4(start), end: round4(start + result.timeRange.duration.seconds), labels: labels))
        } catch {
            failure = error
        }
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {
        failure = error
    }
}

func runSounds(_ job: SoundsJob) throws {
    let analyzer = try SNAudioFileAnalyzer(url: URL(fileURLWithPath: job.path))
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    request.windowDuration = CMTime(seconds: job.window, preferredTimescale: 1000)
    request.overlapFactor = job.overlap
    let windows = SoundWindows(top: job.top, minConf: job.min_conf)
    try analyzer.add(request, withObserver: windows)
    analyzer.analyze()
    if let failure = windows.failure { throw failure }
}

let input = FileHandle.standardInput.readDataToEndOfFile()
do {
    switch CommandLine.arguments.dropFirst().first {
    case "images": try runImages(JSONDecoder().decode(ImagesJob.self, from: input))
    case "video": try await runVideo(JSONDecoder().decode(VideoJob.self, from: input))
    case "sounds": try runSounds(JSONDecoder().decode(SoundsJob.self, from: input))
    default: throw Failure(description: "usage: tlv-vision images|video|sounds < job.json")
    }
} catch {
    FileHandle.standardError.write(Data("tlv-vision: \(error)\n".utf8))
    exit(1)
}
