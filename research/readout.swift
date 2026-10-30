// Metal qualification harness for the current fixed-corpus readout.
// CPU adjoints use rounded GPU primal centroids after independent centroid checks.
import Foundation
import Metal
import CryptoKit

let vocabulary = 262144
let offsets = [0, 8, 72, 584, 4680]
let nodes = 37448
let scale = Double(Float(2.82842712))

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ text: String) { description = text }
}

struct WaveShape {
    var blocks: UInt32
    var tokens: UInt32
    var proposal_start: UInt32 = 0
    var intra: UInt32 = 0
    var vocabulary: UInt32 = 262144
    var shards: UInt32 = 2048
    var temperature: Float = 0
    var score: UInt32 = 1
}

let header = """
#include <metal_stdlib>
using namespace metal;
struct WaveShape {
    uint blocks, tokens, proposal_start, intra, vocabulary, shards;
    float temperature;
    uint score;
};
static_assert(sizeof(WaveShape) == 32, "WaveShape size");
static_assert(__builtin_offsetof(WaveShape, score) == 28, "WaveShape score");
"""

func values(_ count: Int, _ salt: UInt32, _ divisor: Float) -> [Float16] {
    (0..<count).map { index in
        var x = UInt32(truncatingIfNeeded: index) &+ salt
        x = (x ^ (x >> 16)) &* 0x7feb352d
        x = (x ^ (x >> 15)) &* 0x846ca68b
        x ^= x >> 16
        return Float16(Float(Int(x & 31) - 16) / divisor)
    }
}

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    var pipelines: [String: MTLComputePipelineState] = [:]

    init(_ source: String) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw Failure("Metal device/queue unavailable") }
        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.fastMathEnabled = false
        library = try device.makeLibrary(source: source, options: options)
    }

    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        let result = values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
        guard let result else { throw Failure("Metal input allocation failed") }
        return result
    }

    func output(_ count: Int, _ stride: Int = 4) throws -> MTLBuffer {
        guard let result = device.makeBuffer(length: count * stride, options: .storageModeShared)
        else { throw Failure("Metal output allocation failed") }
        memset(result.contents(), 0, result.length)
        return result
    }

    func dispatch(_ name: String, _ buffers: [MTLBuffer], _ shape: WaveShape,
                  _ groups: Int, _ width: Int) throws {
        if pipelines[name] == nil {
            guard let function = library.makeFunction(name: name)
            else { throw Failure("missing kernel \(name)") }
            pipelines[name] = try device.makeComputePipelineState(function: function)
        }
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
        else { throw Failure("Metal command allocation failed") }
        encoder.setComputePipelineState(pipelines[name]!)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var parameters = shape
        encoder.setBytes(&parameters, length: MemoryLayout<WaveShape>.stride, index: buffers.count)
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if command.status != .completed { throw Failure("\(name): \(String(describing: command.error))") }
    }
}

struct ErrorStats {
    var count = 0
    var maximum = 0.0
    mutating func check(_ actual: Float, _ expected: Double, _ label: String) throws {
        let error = abs(Double(actual) - expected)
        if !actual.isFinite || !expected.isFinite || error > 3e-6 * (1 + abs(expected)) {
            throw Failure("\(label): actual=\(actual) reference=\(expected) error=\(error)")
        }
        count += 1
        maximum = max(maximum, error)
    }
}

struct Reference {
    var query: [[Double]]
    var errors: [[Double]]
    var queryGrad: [[Double]]
    var losses: [Double]
    var centroidChecks = 0
    var centroidError = 0.0
}

func reference(_ hidden: [Float16], _ projection: [Float16], _ weights: [Float16],
               _ targets: [UInt32], _ centroid: MTLBuffer) throws -> Reference {
    let stored = centroid.contents().bindMemory(to: Float16.self, capacity: nodes * 64)
    var result = Reference(query: [], errors: [], queryGrad: [], losses: [])
    var checked = Set<Int>()
    for (row, target) in targets.enumerated() {
        var query = [Double](repeating: 0, count: 64)
        for dim in 0..<64 {
            var total: Float = 0
            for h in 0..<512 {
                total = total.addingProduct(Float(hidden[row * 512 + h]), Float(projection[h * 640 + dim]))
            }
            query[dim] = Double(Float16(total * Float(scale)))
        }
        var rowErrors: [Double] = []
        var gradient = [Double](repeating: 0, count: 64)
        var loss = 0.0
        for level in 0..<6 {
            let leaves = 1 << ((5 - level) * 3)
            let parent = Int(target) / (8 * leaves)
            var children = [[Double]]()
            for child in 0..<8 {
                let index = parent * 8 + child
                var vector = [Double](repeating: 0, count: 64)
                for dim in 0..<64 {
                    if level == 5 {
                        vector[dim] = Double(weights[dim * vocabulary + index])
                    } else {
                        let address = (offsets[level] + index) * 64 + dim
                        vector[dim] = Double(stored[address])
                        if checked.insert(address).inserted {
                            var sum = 0.0
                            for token in index * leaves..<(index + 1) * leaves {
                                sum += Double(weights[dim * vocabulary + token])
                            }
                            let expected = Float16(sum / sqrt(Double(leaves)))
                            let error = abs(vector[dim] - Double(expected))
                            let tolerance = max(Double(expected.ulp), Double(Float16.leastNonzeroMagnitude))
                            if error > tolerance { throw Failure("centroid \(address) error \(error)") }
                            result.centroidChecks += 1
                            result.centroidError = max(result.centroidError, error)
                        }
                    }
                }
                children.append(vector)
            }
            let logits = children.map { child in zip(query, child).reduce(0) { $0 + $1.0 * $1.1 } }
            let peak = logits.max()!
            let exponents = logits.map { exp($0 - peak) }
            let denominator = exponents.reduce(0, +)
            let selected = Int(target) / leaves % 8
            loss += peak + log(denominator) - logits[selected]
            for child in 0..<8 {
                let error = (exponents[child] / denominator - (child == selected ? 1 : 0)) / Double(targets.count)
                rowErrors.append(error)
                for dim in 0..<64 { gradient[dim] += error * children[child][dim] }
            }
        }
        result.query.append(query)
        result.errors.append(rowErrors)
        result.queryGrad.append(gradient)
        result.losses.append(loss)
    }
    return result
}

func checkLeaves(_ reference: Reference, _ targets: [UInt32], _ buffer: MTLBuffer,
                 _ stats: inout ErrorStats) throws {
    let actual = buffer.contents().bindMemory(to: Float.self, capacity: 64 * vocabulary)
    for dim in 0..<64 {
        var endpoints = [Double](repeating: 0, count: vocabulary + 1)
        for (row, target) in targets.enumerated() {
            for level in 0..<6 {
                let leaves = 1 << ((5 - level) * 3)
                let parent = Int(target) / (8 * leaves)
                for child in 0..<8 {
                    let first = (parent * 8 + child) * leaves
                    let contribution = reference.query[row][dim] * reference.errors[row][level * 8 + child] / sqrt(Double(leaves))
                    endpoints[first] += contribution
                    endpoints[first + leaves] -= contribution
                }
            }
        }
        var total = 0.0
        for token in 0..<vocabulary {
            total += endpoints[token]
            try stats.check(actual[dim * vocabulary + token], total, "leaf")
        }
    }
}

func checkProjection(_ reference: Reference, _ hidden: [Float16], _ projection: [Float16],
                     _ dh: MTLBuffer, _ dq: MTLBuffer, _ stats: inout ErrorStats) throws {
    let actualQ = dq.contents().bindMemory(to: Float.self, capacity: 512 * 640)
    let actualH = dh.contents().bindMemory(to: Float.self, capacity: hidden.count)
    for h in 0..<512 {
        for dim in 0..<640 {
            var expected = 0.0
            if dim < 64 {
                for row in 0..<reference.query.count {
                    expected += Double(hidden[row * 512 + h]) * reference.queryGrad[row][dim] * scale
                }
            }
            try stats.check(actualQ[h * 640 + dim], expected, "projection")
        }
        for row in 0..<reference.query.count {
            var expected = 0.0
            for dim in 0..<64 {
                expected += Double(projection[h * 640 + dim]) * reference.queryGrad[row][dim] * scale
            }
            try stats.check(actualH[row * 512 + h], expected, "hidden")
        }
    }
}

func qualify(_ gpu: GPU, _ weights: [Float16], _ targets: [UInt32], _ zero: Bool) throws {
    let hidden = values(targets.count * 512, 3, 32)
    let projection = zero ? [Float16](repeating: 0, count: 512 * 640) : values(512 * 640, 17, 128)
    let h = try gpu.buffer(hidden), q = try gpu.buffer(projection), w = try gpu.buffer(weights)
    let target = try gpu.buffer(targets), centroid = try gpu.output(nodes * 64, 2)
    let selected = try gpu.output(targets.count), losses = try gpu.output(targets.count)
    let queries = try gpu.output(targets.count * 64), errors = try gpu.output(targets.count * 48)
    let gquery = try gpu.output(targets.count * 64), gw = try gpu.output(vocabulary * 64)
    let gnodes = try gpu.output((nodes + vocabulary) * 64)
    let gh = try gpu.output(hidden.count), gq = try gpu.output(projection.count)
    let shape = WaveShape(blocks: UInt32(targets.count), tokens: UInt32(targets.count))
    try gpu.dispatch("mhc4_centroids", [w, centroid], shape, 584 * 64, 64)
    try gpu.dispatch("mhc4_readout", [h, q, w, centroid, target, selected, losses], shape, targets.count, 64)
    try gpu.dispatch("tree_adjoint", [h, q, w, centroid, target, queries, errors, gquery, gnodes], shape, targets.count, 1)
    try gpu.dispatch("leaf_adjoint", [gnodes, gw], shape, vocabulary * 64 / 256, 256)
    try gpu.dispatch("project_adjoint", [h, q, gquery, gh, gq], shape, max(512 * 640, hidden.count) / 256, 256)
    let expected = try reference(hidden, projection, weights, targets, centroid)
    var stats = ErrorStats()
    for row in targets.indices {
        try stats.check(losses.contents().load(fromByteOffset: row * 4, as: Float.self), expected.losses[row], "loss")
        for dim in 0..<64 {
            let offset = (row * 64 + dim) * 4
            try stats.check(queries.contents().load(fromByteOffset: offset, as: Float.self), expected.query[row][dim], "query")
            try stats.check(gquery.contents().load(fromByteOffset: offset, as: Float.self), expected.queryGrad[row][dim], "query gradient")
        }
        for child in 0..<48 {
            try stats.check(errors.contents().load(fromByteOffset: (row * 48 + child) * 4, as: Float.self), expected.errors[row][child], "coefficient")
        }
    }
    try checkLeaves(expected, targets, gw, &stats)
    try checkProjection(expected, hidden, projection, gh, gq, &stats)
    print("targets=\(targets) zero_query=\(zero) checked=\(stats.count) max_error=\(stats.maximum) centroid_checks=\(expected.centroidChecks) centroid_error=\(expected.centroidError)")
}

do {
    guard CommandLine.arguments.count == 3, MemoryLayout<WaveShape>.stride == 32
    else { throw Failure("expected production source and adjoint source; WaveShape must be 32 bytes") }
    let forward = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
    let adjoint = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8)
    let gpu = try GPU(header + "\n" + forward + "\n" + adjoint)
    print("device=\(gpu.device.name) forward_sha256=\(SHA256.hash(data: Data(forward.utf8)))")
    let weights = values(vocabulary * 64, 91, 512)
    try qualify(gpu, weights, [0, 262143], false)
    try qualify(gpu, weights, [12345, 12345], false)
    try qualify(gpu, weights, [7, 32768], true)
    try qualify(gpu, weights, [4095], false)
    try qualify(gpu, weights, [0, 4096, 262143], false)
    try qualify(gpu, weights, (0..<32).map { UInt32(($0 * 4099 + 17) % vocabulary) }, false)
    print("scope=readout-only cast_adjoint=identity full_model_parity=false")
} catch {
    fputs("readout qualification failed: \(error)\n", stderr)
    exit(1)
}
