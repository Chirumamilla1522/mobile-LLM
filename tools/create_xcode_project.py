#!/usr/bin/env python3
"""
Generates a standalone NanoEdge.xcodeproj file for physical iPhone deployment.
All sources are unified directly in NanoEdgeApp.
"""

import os
import uuid

def gen_id():
    return uuid.uuid4().hex[:24].upper()

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ios_dir = os.path.join(root_dir, "mobile", "ios")
    proj_dir = os.path.join(ios_dir, "NanoEdge.xcodeproj")
    os.makedirs(proj_dir, exist_ok=True)
    
    # Generate unique 24-character hex IDs
    proj_id = gen_id()
    target_id = gen_id()
    sources_build_phase_id = gen_id()
    frameworks_build_phase_id = gen_id()
    resources_build_phase_id = gen_id()
    app_product_id = gen_id()
    main_group_id = gen_id()
    products_group_id = gen_id()
    sources_group_id = gen_id()
    resources_group_id = gen_id()
    
    proj_cfg_list_id = gen_id()
    proj_cfg_debug_id = gen_id()
    proj_cfg_release_id = gen_id()
    
    target_cfg_list_id = gen_id()
    target_cfg_debug_id = gen_id()
    target_cfg_release_id = gen_id()
    
    lucide_pkg_ref_id = gen_id()
    lucide_dep_id = gen_id()
    lucide_build_file_id = gen_id()
    
    # Source files inside NanoEdgeApp/
    files = [
        ("NanoEdgeApp.swift", "NanoEdgeApp/NanoEdgeApp.swift", "sourcecode.swift", True, False),
        ("ContentView.swift", "NanoEdgeApp/ContentView.swift", "sourcecode.swift", True, False),
        ("FeaturesView.swift", "NanoEdgeApp/FeaturesView.swift", "sourcecode.swift", True, False),
        ("StudioTheme.swift", "NanoEdgeApp/StudioTheme.swift", "sourcecode.swift", True, False),
        ("ChatView.swift", "NanoEdgeApp/ChatView.swift", "sourcecode.swift", True, False),
        ("AppsView.swift", "NanoEdgeApp/AppsView.swift", "sourcecode.swift", True, False),
        ("BenchmarkView.swift", "NanoEdgeApp/BenchmarkView.swift", "sourcecode.swift", True, False),
        ("SpeechManager.swift", "NanoEdgeApp/SpeechManager.swift", "sourcecode.swift", True, False),
        ("DocumentScannerView.swift", "NanoEdgeApp/DocumentScannerView.swift", "sourcecode.swift", True, False),
        ("DeviceToolRegistry.swift", "NanoEdgeApp/DeviceToolRegistry.swift", "sourcecode.swift", True, False),
        ("LocalRAGStore.swift", "NanoEdgeApp/LocalRAGStore.swift", "sourcecode.swift", True, False),
        ("SpeculativeDecoder.swift", "NanoEdgeApp/SpeculativeDecoder.swift", "sourcecode.swift", True, False),
        ("VoiceOrbView.swift", "NanoEdgeApp/VoiceOrbView.swift", "sourcecode.swift", True, False),
        ("LiveCameraVisionView.swift", "NanoEdgeApp/LiveCameraVisionView.swift", "sourcecode.swift", True, False),
        ("DynamicIslandHUD.swift", "NanoEdgeApp/DynamicIslandHUD.swift", "sourcecode.swift", True, False),
        ("NanoEdgeBridge.h", "NanoEdgeApp/NanoEdgeBridge.h", "sourcecode.c.h", False, False),
        ("nanoedge_rust.h", "NanoEdgeApp/nanoedge_rust.h", "sourcecode.c.h", False, False),
        ("NanoEdgeBridge.mm", "NanoEdgeApp/NanoEdgeBridge.mm", "sourcecode.cpp.objcpp", True, False),
        ("memory_mapped_model.cpp", "NanoEdgeApp/memory_mapped_model.cpp", "sourcecode.cpp.cpp", True, False),
        ("arena_allocator.cpp", "NanoEdgeApp/arena_allocator.cpp", "sourcecode.cpp.cpp", True, False),
        ("neon_gemv.cpp", "NanoEdgeApp/neon_gemv.cpp", "sourcecode.cpp.cpp", True, False),
        ("metal_backend.mm", "NanoEdgeApp/metal_backend.mm", "sourcecode.cpp.objcpp", True, False),
        ("Info.plist", "NanoEdgeApp/Info.plist", "text.plist.xml", False, False),
        ("NanoEdge.entitlements", "NanoEdgeApp/NanoEdge.entitlements", "text.plist.entitlements", False, False),
        ("mobile_demo_q4.mllm", "NanoEdgeApp/Resources/mobile_demo_q4.mllm", "file", False, True),
        ("smollm2_135m_q4.mllm", "NanoEdgeApp/Resources/smollm2_135m_q4.mllm", "file", False, True),
        ("smollm2_vocab.json", "NanoEdgeApp/Resources/smollm2_vocab.json", "file", False, True),
        ("llama3_2_1b_instruct_q4.mllm", "NanoEdgeApp/Resources/llama3_2_1b_instruct_q4.mllm", "file", False, True),
        ("llama3_2_3b_instruct_q4.mllm", "NanoEdgeApp/Resources/llama3_2_3b_instruct_q4.mllm", "file", False, True),
        ("llama3_vocab.json", "NanoEdgeApp/Resources/llama3_vocab.json", "file", False, True),
    ]
    
    file_entries = {}
    build_file_entries = {}
    
    for name, path, ftype, is_source, is_resource in files:
        f_id = gen_id()
        file_entries[name] = (f_id, path, ftype, is_source, is_resource)
        if is_source or is_resource:
            b_id = gen_id()
            build_file_entries[name] = (b_id, f_id)

    # Build sections dynamically
    build_file_lines = []
    file_ref_lines = []
    sources_phase_lines = []
    resources_phase_lines = []
    app_group_lines = []
    res_group_lines = []
    
    for name, (f_id, path, ftype, is_source, is_resource) in file_entries.items():
        if is_source:
            b_id = build_file_entries[name][0]
            build_file_lines.append(f"\t\t{b_id} /* {name} in Sources */ = {{isa = PBXBuildFile; fileRef = {f_id} /* {name} */; }};")
            sources_phase_lines.append(f"\t\t\t\t{b_id} /* {name} in Sources */,")
            app_group_lines.append(f"\t\t\t\t{f_id} /* {name} */,")
        elif is_resource:
            b_id = build_file_entries[name][0]
            build_file_lines.append(f"\t\t{b_id} /* {name} in Resources */ = {{isa = PBXBuildFile; fileRef = {f_id} /* {name} */; }};")
            resources_phase_lines.append(f"\t\t\t\t{b_id} /* {name} in Resources */,")
            res_group_lines.append(f"\t\t\t\t{f_id} /* {name} */,")
        else:
            app_group_lines.append(f"\t\t\t\t{f_id} /* {name} */,")
            
        file_ref_lines.append(f"\t\t{f_id} /* {name} */ = {{isa = PBXFileReference; lastKnownFileType = {ftype}; path = {os.path.basename(path)}; sourceTree = \"<group>\"; }};")

    build_files_str = "\n".join(build_file_lines) + f"\n\t\t{lucide_build_file_id} /* Lucide in Frameworks */ = {{isa = PBXBuildFile; productRef = {lucide_dep_id} /* Lucide */; }};"
    file_refs_str = "\n".join(file_ref_lines)
    sources_phase_str = "\n".join(sources_phase_lines)
    resources_phase_str = "\n".join(resources_phase_lines)
    app_group_str = "\n".join(app_group_lines)
    res_group_str = "\n".join(res_group_lines)
    
    pbxproj_content = f"""// !$*UTF8*$!
{{
	archiveVersion = 1;
	classes = {{
	}};
	objectVersion = 56;
	objects = {{

/* Begin PBXBuildFile section */
{build_files_str}
/* End PBXBuildFile section */

/* Begin PBXFileReference section */
		{app_product_id} /* NanoEdge.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = NanoEdge.app; sourceTree = BUILT_PRODUCTS_DIR; }};
{file_refs_str}
/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
		{frameworks_build_phase_id} /* Frameworks */ = {{
			isa = PBXFrameworksBuildPhase;
			buildActionMask = 2147483647;
			files = (
				{lucide_build_file_id} /* Lucide in Frameworks */,
			);
			runOnlyForDeploymentPostprocessing = 0;
		}};
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
		{main_group_id} = {{
			isa = PBXGroup;
			children = (
				{sources_group_id} /* NanoEdgeApp */,
				{resources_group_id} /* Resources */,
				{products_group_id} /* Products */,
			);
			sourceTree = "<group>";
		}};
		{products_group_id} /* Products */ = {{
			isa = PBXGroup;
			children = (
				{app_product_id} /* NanoEdge.app */,
			);
			name = Products;
			sourceTree = "<group>";
		}};
		{sources_group_id} /* NanoEdgeApp */ = {{
			isa = PBXGroup;
			children = (
{app_group_str}
			);
			name = NanoEdgeApp;
			path = NanoEdgeApp;
			sourceTree = "<group>";
		}};
		{resources_group_id} /* Resources */ = {{
			isa = PBXGroup;
			children = (
{res_group_str}
			);
			name = Resources;
			path = NanoEdgeApp/Resources;
			sourceTree = "<group>";
		}};
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
		{target_id} /* NanoEdge */ = {{
			isa = PBXNativeTarget;
			buildConfigurationList = {target_cfg_list_id} /* Build configuration list for PBXNativeTarget "NanoEdge" */;
			buildPhases = (
				{sources_build_phase_id} /* Sources */,
				{frameworks_build_phase_id} /* Frameworks */,
				{resources_build_phase_id} /* Resources */,
			);
			buildRules = (
			);
			dependencies = (
			);
			name = NanoEdge;
			packageProductDependencies = (
				{lucide_dep_id} /* Lucide */,
			);
			productName = NanoEdge;
			productReference = {app_product_id} /* NanoEdge.app */;
			productType = "com.apple.product-type.application";
		}};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
		{proj_id} /* Project object */ = {{
			isa = PBXProject;
			attributes = {{
				BuildIndependentTargetsInParallel = 1;
				LastSwiftUpdateCheck = 1500;
				LastUpgradeCheck = 1500;
				TargetAttributes = {{
					{target_id} = {{
						CreatedOnToolsVersion = 15.0;
						DevelopmentTeam = 5HAT35ND22;
						ProvisioningStyle = Automatic;
					}};
				}};
			}};
			buildConfigurationList = {proj_cfg_list_id} /* Build configuration list for PBXProject "NanoEdge" */;
			compatibilityVersion = "Xcode 14.0";
			developmentRegion = en;
			hasScannedForEncodings = 0;
			knownRegions = (
				en,
				Base,
			);
			mainGroup = {main_group_id};
			packageReferences = (
				{lucide_pkg_ref_id} /* XCLocalSwiftPackageReference "Packages/Lucide" */,
			);
			productRefGroup = {products_group_id} /* Products */;
			projectDirPath = "";
			projectRoot = "";
			targets = (
				{target_id} /* NanoEdge */,
			);
		}};
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
		{resources_build_phase_id} /* Resources */ = {{
			isa = PBXResourcesBuildPhase;
			buildActionMask = 2147483647;
			files = (
{resources_phase_str}
			);
			runOnlyForDeploymentPostprocessing = 0;
		}};
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
		{sources_build_phase_id} /* Sources */ = {{
			isa = PBXSourcesBuildPhase;
			buildActionMask = 2147483647;
			files = (
{sources_phase_str}
			);
			runOnlyForDeploymentPostprocessing = 0;
		}};
/* End PBXSourcesBuildPhase section */

/* Begin XCLocalSwiftPackageReference section */
		{lucide_pkg_ref_id} /* XCLocalSwiftPackageReference "Packages/Lucide" */ = {{
			isa = XCLocalSwiftPackageReference;
			relativePath = Packages/Lucide;
		}};
/* End XCLocalSwiftPackageReference section */

/* Begin XCSwiftPackageProductDependency section */
		{lucide_dep_id} /* Lucide */ = {{
			isa = XCSwiftPackageProductDependency;
			package = {lucide_pkg_ref_id} /* XCLocalSwiftPackageReference "Packages/Lucide" */;
			productName = Lucide;
		}};
/* End XCSwiftPackageProductDependency section */

/* Begin XCBuildConfiguration section */
		{proj_cfg_debug_id} /* Debug */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				ALWAYS_SEARCH_USER_PATHS = NO;
				CLANG_ANALYZER_NONNULL = YES;
				CLANG_CXX_LANGUAGE_STANDARD = "c++20";
				CLANG_ENABLE_MODULES = YES;
				CLANG_ENABLE_OBJC_ARC = YES;
				ENABLE_TESTABILITY = YES;
				GCC_DYNAMIC_NO_PIC = NO;
				GCC_OPTIMIZATION_LEVEL = 0;
				GCC_PREPROCESSOR_DEFINITIONS = (
					"DEBUG=1",
					"$(inherited)",
				);
				IPHONEOS_DEPLOYMENT_TARGET = 17.0;
				MTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;
				MTL_FAST_MATH = YES;
				ONLY_ACTIVE_ARCH = YES;
				SDKROOT = iphoneos;
				SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG;
				SWIFT_OPTIMIZATION_LEVEL = "-Onone";
			}};
			name = Debug;
		}};
		{proj_cfg_release_id} /* Release */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				ALWAYS_SEARCH_USER_PATHS = NO;
				CLANG_ANALYZER_NONNULL = YES;
				CLANG_CXX_LANGUAGE_STANDARD = "c++20";
				CLANG_ENABLE_MODULES = YES;
				CLANG_ENABLE_OBJC_ARC = YES;
				COPY_PHASE_STRIP = NO;
				ENABLE_NS_ASSERTIONS = NO;
				GCC_OPTIMIZATION_LEVEL = s;
				IPHONEOS_DEPLOYMENT_TARGET = 17.0;
				MTL_ENABLE_DEBUG_INFO = NO;
				MTL_FAST_MATH = YES;
				SDKROOT = iphoneos;
				SWIFT_COMPILATION_MODE = wholemodule;
				SWIFT_OPTIMIZATION_LEVEL = "-O";
				VALIDATE_PRODUCT = YES;
			}};
			name = Release;
		}};
		{target_cfg_debug_id} /* Debug */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;
				CLANG_ENABLE_MODULES = YES;
				CODE_SIGN_ENTITLEMENTS = NanoEdgeApp/NanoEdge.entitlements;
				CODE_SIGN_STYLE = Automatic;
				DEVELOPMENT_TEAM = 5HAT35ND22;
				CURRENT_PROJECT_VERSION = 1;
				GENERATE_INFOPLIST_FILE = NO;
				HEADER_SEARCH_PATHS = (
					"$(SRCROOT)/NanoEdgeApp",
					"$(SRCROOT)/NanoEdgeApp/mllm",
					"$(SRCROOT)/../../include",
					"$(SRCROOT)/../../src",
				);
				INFOPLIST_FILE = NanoEdgeApp/Info.plist;
				LD_RUNPATH_SEARCH_PATHS = (
					"$(inherited)",
					"@executable_path/Frameworks",
				);
				LIBRARY_SEARCH_PATHS = (
					"$(SRCROOT)/NanoEdgeApp",
				);
				MARKETING_VERSION = 1.0;
				OTHER_LDFLAGS = (
					"-framework",
					"Metal",
					"-framework",
					"Foundation",
					"-framework",
					"UIKit",
					"-framework",
					"Vision",
					"-framework",
					"Speech",
					"-framework",
					"AVFoundation",
					"-lruntime_rust",
				);
				PRODUCT_BUNDLE_IDENTIFIER = com.nanoedge.NanoEdgeApp;
				PRODUCT_NAME = "$(TARGET_NAME)";
				SWIFT_EMIT_LOC_STRINGS = YES;
				SWIFT_OBJC_BRIDGING_HEADER = NanoEdgeApp/NanoEdgeBridge.h;
				SWIFT_VERSION = 5.0;
				TARGETED_DEVICE_FAMILY = "1,2";
			}};
			name = Debug;
		}};
		{target_cfg_release_id} /* Release */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;
				CLANG_ENABLE_MODULES = YES;
				CODE_SIGN_ENTITLEMENTS = NanoEdgeApp/NanoEdge.entitlements;
				CODE_SIGN_STYLE = Automatic;
				DEVELOPMENT_TEAM = 5HAT35ND22;
				CURRENT_PROJECT_VERSION = 1;
				GENERATE_INFOPLIST_FILE = NO;
				HEADER_SEARCH_PATHS = (
					"$(SRCROOT)/NanoEdgeApp",
					"$(SRCROOT)/NanoEdgeApp/mllm",
					"$(SRCROOT)/../../include",
					"$(SRCROOT)/../../src",
				);
				INFOPLIST_FILE = NanoEdgeApp/Info.plist;
				LD_RUNPATH_SEARCH_PATHS = (
					"$(inherited)",
					"@executable_path/Frameworks",
				);
				LIBRARY_SEARCH_PATHS = (
					"$(SRCROOT)/NanoEdgeApp",
				);
				MARKETING_VERSION = 1.0;
				OTHER_LDFLAGS = (
					"-framework",
					"Metal",
					"-framework",
					"Foundation",
					"-framework",
					"UIKit",
					"-framework",
					"Vision",
					"-framework",
					"Speech",
					"-framework",
					"AVFoundation",
					"-lruntime_rust",
				);
				PRODUCT_BUNDLE_IDENTIFIER = com.nanoedge.NanoEdgeApp;
				PRODUCT_NAME = "$(TARGET_NAME)";
				SWIFT_EMIT_LOC_STRINGS = YES;
				SWIFT_OBJC_BRIDGING_HEADER = NanoEdgeApp/NanoEdgeBridge.h;
				SWIFT_VERSION = 5.0;
				TARGETED_DEVICE_FAMILY = "1,2";
			}};
			name = Release;
		}};
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
		{proj_cfg_list_id} /* Build configuration list for PBXProject "NanoEdge" */ = {{
			isa = XCConfigurationList;
			buildConfigurations = (
				{proj_cfg_debug_id} /* Debug */,
				{proj_cfg_release_id} /* Release */,
			);
			defaultConfigurationIsVisible = 0;
			defaultConfigurationName = Release;
		}};
		{target_cfg_list_id} /* Build configuration list for PBXNativeTarget "NanoEdge" */ = {{
			isa = XCConfigurationList;
			buildConfigurations = (
				{target_cfg_debug_id} /* Debug */,
				{target_cfg_release_id} /* Release */,
			);
			defaultConfigurationIsVisible = 0;
			defaultConfigurationName = Release;
		}};
/* End XCConfigurationList section */

	}};
	rootObject = {proj_id} /* Project object */;
}}
"""
    
    pbxproj_path = os.path.join(proj_dir, "project.pbxproj")
    with open(pbxproj_path, "w") as f:
        f.write(pbxproj_content)
        
    print(f"Generated updated Xcode project at: {proj_dir}")

if __name__ == "__main__":
    main()
