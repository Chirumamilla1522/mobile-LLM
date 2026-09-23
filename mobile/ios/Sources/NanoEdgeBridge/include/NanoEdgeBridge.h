#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NanoEdgeMemoryStats : NSObject
@property (nonatomic, assign) double virtualSizeMB;
@property (nonatomic, assign) double residentSizeMB;
@property (nonatomic, assign) double physicalFootprintMB;
@end

@interface NanoEdgeBenchmarkResult : NSObject
@property (nonatomic, copy) NSString* tensorName;
@property (nonatomic, assign) uint32_t rows;
@property (nonatomic, assign) uint32_t cols;
@property (nonatomic, assign) double weightSizeMB;
@property (nonatomic, assign) double cpuLatencyUs;
@property (nonatomic, assign) double gpuMinLatencyUs;
@property (nonatomic, assign) double gpuMedianLatencyUs;
@property (nonatomic, assign) double effectiveBandwidthGBs;
@property (nonatomic, assign) double tokensPerSecCeiling;
@property (nonatomic, assign) BOOL validationPassed;
@end

@interface NanoEdgeBridge : NSObject

+ (instancetype)sharedInstance;

- (NSString*)deviceName;
- (BOOL)hasUnifiedMemory;
- (NanoEdgeMemoryStats*)queryMemoryFootprint;

- (BOOL)loadModelFromPath:(NSString*)filePath error:(NSError**)error;
- (BOOL)isModelLoaded;
- (NSString*)loadedModelInfo;

- (NanoEdgeBenchmarkResult*)runDecodeBenchmark:(NSInteger)iterations;
- (void)handleMemoryWarning;

@end

NS_ASSUME_NONNULL_END
