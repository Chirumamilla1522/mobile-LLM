#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, NanoEdgeExecutionEngine) {
    NanoEdgeExecutionEngineMetalGPUTiled = 0,
    NanoEdgeExecutionEngineMetalGPUBaseline = 1,
    NanoEdgeExecutionEngineNeonCPUMultiCore = 2,
    NanoEdgeExecutionEngineNeonCPUSingleCore = 3,
    NanoEdgeExecutionEngineAppleNeuralEngine = 4
};

@interface NanoEdgeMemoryStats : NSObject
@property (nonatomic, assign) double virtualSizeMB;
@property (nonatomic, assign) double residentSizeMB;
@property (nonatomic, assign) double physicalFootprintMB;
@end

@interface NanoEdgeBenchmarkResult : NSObject
@property (nonatomic, copy) NSString* tensorName;
@property (nonatomic, copy) NSString* quantTypeName;
@property (nonatomic, copy) NSString* engineName;
@property (nonatomic, assign) uint32_t rows;
@property (nonatomic, assign) uint32_t cols;
@property (nonatomic, assign) double weightSizeMB;

// Latency distribution metrics
@property (nonatomic, assign) double minLatencyUs;
@property (nonatomic, assign) double meanLatencyUs;
@property (nonatomic, assign) double medianLatencyUs; // P50
@property (nonatomic, assign) double p90LatencyUs;
@property (nonatomic, assign) double p99LatencyUs;
@property (nonatomic, assign) double jitterUs;

// Legacy properties for compatibility
@property (nonatomic, assign) double cpuLatencyUs;
@property (nonatomic, assign) double gpuMinLatencyUs;
@property (nonatomic, assign) double gpuMedianLatencyUs;

// Throughput and silicon metrics
@property (nonatomic, assign) double effectiveBandwidthGBs;
@property (nonatomic, assign) double bandwidthUtilizationPct;
@property (nonatomic, assign) double gflops;
@property (nonatomic, assign) double tokensPerSecCeiling;
@property (nonatomic, copy) NSString* thermalStateName;
@property (nonatomic, assign) double memoryFootprintDeltaMB;
@property (nonatomic, assign) BOOL validationPassed;

// Architecture and kernel diagnostics
@property (nonatomic, assign) double arithmeticIntensity; // FLOPs/byte
@property (nonatomic, assign) double activationMemoryTrafficMB;
@property (nonatomic, assign) double memoryTrafficSavedMB;
@property (nonatomic, assign) double speedupVsSingleCore;

// Silicon Energy & Thermal Profiler (Apple A18 Pro Hardware Telemetry)
@property (nonatomic, assign) double activeWatts;
@property (nonatomic, assign) double energyPerTokenMilliJoules;
@property (nonatomic, assign) double batteryLifeRemainingHours;
@property (nonatomic, assign) NSInteger thermalStateLevel; // 0: Nominal, 1: Fair, 2: Serious, 3: Critical
@property (nonatomic, assign) double aneSpeedupVsCpu;
@property (nonatomic, assign) double aneEnergyEfficiencyVsGpu;

// A/B Comparison: With vs. Without Optimizations
@property (nonatomic, assign) double baselineLatencyUs;         // Latency WITHOUT optimizations (µs)
@property (nonatomic, assign) double baselineTokensPerSec;      // Speed WITHOUT optimizations (tok/s)
@property (nonatomic, assign) double optimizedLatencyUs;        // Latency WITH optimizations (µs)
@property (nonatomic, assign) double optimizedTokensPerSec;     // Speed WITH optimizations (tok/s)
@property (nonatomic, assign) double speedupFactor;             // e.g. 1.85x
@property (nonatomic, assign) double latencyReductionPct;       // e.g. 45.2% faster
@property (nonatomic, copy) NSString* baselineName;             // Description of baseline configuration
@property (nonatomic, copy) NSString* optimizedName;            // Description of optimized configuration

// Series data points for SwiftUI Charts (Optimized and Baseline)
@property (nonatomic, strong) NSArray<NSNumber*>* latencyDataPoints;
@property (nonatomic, strong) NSArray<NSNumber*>* throughputDataPoints;
@property (nonatomic, strong) NSArray<NSNumber*>* baselineLatencyDataPoints;
@property (nonatomic, strong) NSArray<NSNumber*>* baselineThroughputDataPoints;
@end

@interface NanoEdgeBridge : NSObject

@property (nonatomic, assign) NanoEdgeExecutionEngine executionEngine;
@property (nonatomic, assign) BOOL useQuantizedKVCache;

+ (instancetype)sharedInstance;

- (NSString*)deviceName;
- (BOOL)hasUnifiedMemory;
- (NanoEdgeMemoryStats*)queryMemoryFootprint;
- (NSString*)currentThermalState;
- (NSInteger)currentThermalStateLevel;
- (float)currentBatteryLevel;
- (BOOL)isDeviceCharging;

- (BOOL)loadModelFromPath:(NSString*)filePath error:(NSError**)error;
- (BOOL)isModelLoaded;
- (NSString*)loadedModelInfo;

- (NanoEdgeBenchmarkResult*)runDecodeBenchmark:(NSInteger)iterations;
- (NanoEdgeBenchmarkResult*)runDecodeBenchmark:(NSInteger)iterations engine:(NanoEdgeExecutionEngine)engine;
- (double)executeSingleDecodeStepWithEngine:(NanoEdgeExecutionEngine)engine;
- (void)handleMemoryWarning;

- (double)quantizedKVCacheMemorySavedMBForContext:(NSInteger)contextLength;

- (void)generateStreamingWithPrompt:(NSString*)prompt
                       systemPrompt:(NSString* _Nullable)systemPrompt
                          maxTokens:(NSInteger)maxTokens
                        temperature:(float)temperature
                            onToken:(void(^)(NSString* token, double tokensPerSec))tokenCallback
                         onComplete:(void(^)(NSString* fullText, double totalTimeSec, double avgTokPerSec, double ttftMs))completeCallback;

- (void)cancelGeneration;
- (BOOL)isGenerating;
- (BOOL)isRustEngineActive;

@end

NS_ASSUME_NONNULL_END
