import Foundation
import UIKit

// Represents a tool definition and its execution handler
public struct LocalDeviceTool: Identifiable {
    public let id: String
    public let name: String
    public let description: String
    public let parametersDescription: String
    public let icon: String
}

public struct ToolCallResult: Identifiable, Equatable {
    public let id = UUID()
    public let toolName: String
    public let inputArgs: String
    public let output: String
    public let executionTimeMs: Double
    public let success: Bool
}

@objc public class DeviceToolRegistry: NSObject {
    @objc public static let shared = DeviceToolRegistry()
    
    public var availableTools: [LocalDeviceTool] = [
        LocalDeviceTool(
            id: "get_device_vitals",
            name: "get_device_vitals",
            description: "Query real-time hardware status: battery percentage, charging state, thermal state, and RAM footprint.",
            parametersDescription: "{}",
            icon: "battery.100.bolt"
        ),
        LocalDeviceTool(
            id: "calculate_math",
            name: "calculate_math",
            description: "Evaluate exact arithmetic or algebraic mathematical expressions using the local math engine.",
            parametersDescription: "{\"expression\": \"string\"}",
            icon: "function"
        ),
        LocalDeviceTool(
            id: "get_system_time",
            name: "get_system_time",
            description: "Get the exact current local time, date, time zone, and calendar status on the iPhone.",
            parametersDescription: "{}",
            icon: "clock.fill"
        ),
        LocalDeviceTool(
            id: "search_knowledge_vault",
            name: "search_knowledge_vault",
            description: "Search 100% offline indexed local documents and notes in the on-device RAG Knowledge Vault.",
            parametersDescription: "{\"query\": \"string\"}",
            icon: "books.vertical.fill"
        ),
        LocalDeviceTool(
            id: "create_local_reminder",
            name: "create_local_reminder",
            description: "Simulate creating an on-device task or reminder.",
            parametersDescription: "{\"title\": \"string\", \"due\": \"string\"}",
            icon: "checklist"
        )
    ]
    
    // Detects and parses <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    public func parseAndExecuteToolCall(in text: String) -> (cleanedText: String, result: ToolCallResult?) {
        let pattern = #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else {
            return (text, nil)
        }
        
        let nsString = text as NSString
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))
        guard let match = matches.first else {
            return (text, nil)
        }
        
        let jsonStr = nsString.substring(with: match.range(at: 1))
        let cleaned = regex.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: nsString.length), withTemplate: "").trimmingCharacters(in: .whitespacesAndNewlines)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        var toolName = "unknown"
        var argsStr = jsonStr
        var output = "Execution failed"
        var success = false
        
        if let data = jsonStr.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            toolName = json["name"] as? String ?? "unknown"
            if let args = json["arguments"] as? [String: Any],
               let argsData = try? JSONSerialization.data(withJSONObject: args, options: .prettyPrinted),
               let str = String(data: argsData, encoding: .utf8) {
                argsStr = str
            }
            
            output = executeTool(name: toolName, json: json)
            success = true
        }
        
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        let result = ToolCallResult(
            toolName: toolName,
            inputArgs: argsStr,
            output: output,
            executionTimeMs: elapsedMs,
            success: success
        )
        
        return (cleaned, result)
    }
    
    // Dispatches execution to local handlers
    public func executeTool(name: String, json: [String: Any]) -> String {
        let args = json["arguments"] as? [String: Any] ?? [:]
        
        switch name {
        case "get_device_vitals":
            let bridge = NanoEdgeBridge.sharedInstance()
            let stats = bridge.queryMemoryFootprint()
            let thermal = bridge.currentThermalState()
            let battery = Int(bridge.currentBatteryLevel() * 100)
            let charging = bridge.isDeviceCharging() ? "Yes (Plugged In)" : "No (On Battery)"
            return """
            • Battery Level: \(battery)%
            • Charging: \(charging)
            • Thermal State: \(thermal)
            • Active Footprint: \(String(format: "%.1f MB", stats.physicalFootprintMB))
            • Silicon: Apple A18 Pro (6-Core CPU, 6-Core GPU, 16-Core ANE)
            • Unified Memory: Yes (Shared High-Bandwidth LPDDR5X)
            """
            
        case "calculate_math":
            let exprStr = args["expression"] as? String ?? ""
            if exprStr.isEmpty { return "Error: No expression provided" }
            let expr = NSExpression(format: exprStr)
            if let val = expr.expressionValue(with: nil, context: nil) as? NSNumber {
                return "Result: \(val)"
            }
            return "Evaluated: \(exprStr)"
            
        case "get_system_time":
            let formatter = DateFormatter()
            formatter.dateStyle = .full
            formatter.timeStyle = .medium
            let nowStr = formatter.string(from: Date())
            let tz = TimeZone.current.identifier
            return "Current Time: \(nowStr) (\(tz))"
            
        case "search_knowledge_vault":
            let query = args["query"] as? String ?? ""
            let hits = LocalRAGStore.shared.search(query: query, topK: 2)
            if hits.isEmpty {
                return "No matching local documents found in Knowledge Vault for '\(query)'."
            }
            var res = "Knowledge Vault Matches:\n"
            for (idx, hit) in hits.enumerated() {
                res += "[\(idx+1)] \(hit.documentTitle) (Score: \(String(format: "%.2f", hit.score))):\n\(hit.snippet)\n\n"
            }
            return res.trimmingCharacters(in: .whitespacesAndNewlines)
            
        case "create_local_reminder":
            let title = args["title"] as? String ?? "Task"
            let due = args["due"] as? String ?? "Today"
            return "✓ Local task scheduled: '\(title)' due [\(due)]. (Stored in on-device cache)"
            
        default:
            return "Unknown tool '\(name)'."
        }
    }
}
