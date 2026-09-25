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
        let navigation = UINavigationBarAppearance()
        navigation.configureWithOpaqueBackground()
        navigation.backgroundColor = UIColor(StudioTheme.canvas)
        navigation.shadowColor = UIColor(StudioTheme.border)
        if let heading = UIFont(name: "PlusJakartaSans-Regular", size: 32),
           let label = UIFont(name: "PlusJakartaSans-Regular", size: 17) {
            navigation.largeTitleTextAttributes = [.font: heading, .foregroundColor: UIColor.white]
            navigation.titleTextAttributes = [.font: label, .foregroundColor: UIColor.white]
        }
        UINavigationBar.appearance().standardAppearance = navigation
        UINavigationBar.appearance().scrollEdgeAppearance = navigation

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
