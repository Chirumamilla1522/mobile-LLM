import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

#if canImport(NanoEdgeBridge)
import NanoEdgeBridge
#endif

@main
struct NanoEdgeApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { _ in
            #if canImport(NanoEdgeBridge)
            NanoEdgeBridge.sharedInstance().handleMemoryWarning()
            #endif
        }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .background {
                #if canImport(NanoEdgeBridge)
                NanoEdgeBridge.sharedInstance().handleMemoryWarning()
                #endif
            }
        }
    }
}
