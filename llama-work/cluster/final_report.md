# Checkpointing using TensorStore: Large Language Model Optimization Study
---

## INTRODUCTION

Managing and processing large multi-dimensional datasets is a significant challenge in data-intensive fields like neuroscience and machine learning. Even a single dataset may require terabytes or petabytes of data storage. Such datasets are also challenging to work with as users may read and write data at irregular intervals and varying scales. TensorStore is an open-source C++ and Python library designed to read and write large multi-dimensional arrays efficiently. It provides a uniform API for reading and writing different array formats like Zarr and N5. It also integrates with various storage backends, including in-memory storage, local file systems, Amazon S3, and Google Cloud Storage to store the array dataset.

One challenge that arises during large language model training is efficiently reading and writing the model parameters, and TensorStore has already been used to address these challenges. It has been applied to manage checkpoints associated with large-scale models trained with JAX and has been integrated with frameworks such as T5X and Pathways. Model parallelism is used to partition the full set of parameters, which can occupy more than a terabyte of memory, over hundreds of TPUs. TensorStore can be used to address computational challenges in large-scale connectomic datasets, efficiently managing some of the largest and most accessed datasets with Google Cloud Storage as the underlying object storage system.

This report extends previous work on TensorStore benchmarking by conducting comprehensive experiments on large language model checkpointing. While previous studies focused on basic TensorStore configurations with synthetic datasets, this work evaluates TensorStore's performance with real-world transformer models ranging from 3 billion to 7 billion parameters. The experiments systematically compare PyTorch's native checkpointing against TensorStore implementations, examining the impact of compression, concurrency, and chunk size parameters across multiple model architectures.

---

## BACKGROUND

TensorStore is an open-source Python and C++ library made for effective multi-dimensional array manipulation and storing. It is especially handy for dealing with big datasets that cannot be stored in memory. It supports Zarr and N5 for efficient multi-dimensional data storage and retrieval. The Zarr data format is a community-maintained format for large-scale n-dimensional data and it enables simple, fast serialization of NumPy-like arrays and supports multi-scale n-dimensional image storage for applications like light and electron microscopy. It is a cloud-friendly format that enables chunked, compressed storage for distributed and parallel processing.

On the other hand, N5 is optimized for large-scale scientific datasets like bioinformatics and microscopy. The N5 API specifies the primitive operations needed to store large chunked n-dimensional tensors, and arbitrary meta-data in a hierarchy of groups. Besides those, Google Cloud Storage (GCS) serves as a scalable backend for storing large, multi-dimensional datasets. TensorStore facilitates parallel I/O operations and smooth integration with cloud-based workflows by utilizing GCS's stability and performance to effectively handle and retrieve large volumes of scientific and machine learning data.

For large language model checkpointing, TensorStore offers several advantages over traditional approaches. First, its chunked storage format allows for partial model loading, which is critical when working with models that exceed available memory. Second, the separation of individual parameters into distinct Zarr arrays enables fine-grained access patterns, allowing selective parameter updates without loading the entire checkpoint. Third, TensorStore's support for concurrent I/O operations can potentially accelerate checkpoint operations in distributed training scenarios. However, these advantages come with trade-offs in terms of metadata overhead and local disk performance, which this study systematically evaluates.

---

## RELATED WORKS

In previous research, the bandwidth of checkpoint creation was measured with and without TensorStore CHFS using T5X. To measure the bandwidth, the parameter size of the T5 1.1 model was changed, and the model was divided into 8, 16, and 32 nodes. Furthermore, the StateTransformer inputs model and dataset partitions from a previous PTC, creates updated partitions for a new PTC' after a resource change, while the TensorStore maintains the model and dataset state partitions in a hierarchical virtual in-memory file system, offering APIs for model checkpointing and enabling the ingestion of training data.

The neuroglancer_precomputed driver is used to access the Janelia FlyEM Hemibrain 1.1 segmentation dataset, as shown in the TensorStore Python tutorial. The dataset is stored in uint64 format with a resolution of 8x8x8 microns per voxel, and users can open it asynchronously by providing the Google Cloud Storage path. Using TensorStore's caching and parallelism features, the tutorial shows how to create 3D views, slice data, and read particular sections of the given dataset.

Recent work has also explored TensorStore's application in production machine learning systems. Google's research on large-scale model training demonstrates TensorStore's effectiveness in managing checkpoints for models with hundreds of billions of parameters distributed across thousands of accelerators. However, these studies primarily focus on distributed cloud environments rather than single-machine local disk scenarios, which represent a common use case in research and development settings.

---

## APPROACH

This study implements a comprehensive benchmarking framework to evaluate TensorStore's performance for large language model checkpointing. The framework compares three distinct approaches across multiple model architectures and configuration parameters.

### Checkpointing Methods

Three checkpointing methods were implemented and compared:

**PyTorch Native Checkpointing:** This baseline approach uses PyTorch's native `torch.save()` function with no compression. It serializes the entire model state dictionary into a single `.pth` file using Python's pickle protocol. This method is highly optimized for sequential I/O and serves as the performance baseline against which TensorStore approaches are measured.

**TensorStore Basic Implementation:** This approach uses TensorStore with the Zarr format, implementing dynamic 64 MB chunking where each model parameter is stored as a separate Zarr array. No compression or concurrency optimizations are applied, providing a baseline for TensorStore's performance characteristics. Each parameter requires its own metadata file, resulting in hundreds of individual files for a typical transformer model.

**T5X-Optimized TensorStore:** This implementation incorporates optimizations from Google's T5X framework, including dynamic 64 MB chunks with gzip compression at level 1 and high concurrency with 128 concurrent file I/O operations. The compression reduces storage requirements while the high concurrency aims to hide I/O latency through parallel operations.

### Model Selection

Four transformer-based language models were selected to represent different scales and architectures:

- **OpenLLaMA-3B:** A 3.4 billion parameter open-source model with approximately 6.4 GB checkpoint size. This model uses float16 precision and serves as the baseline for 3B-scale models.

- **Llama-3.2-3B-Instruct:** A 3 billion parameter instruction-tuned model with approximately 6 GB checkpoint size. This model provides a comparison point for different 3B architectures.

- **Qwen2.5-7B:** A 7 billion parameter model with approximately 14 GB checkpoint size. This model uses bfloat16 precision natively, requiring conversion to float16 for TensorStore compatibility.

- **Mistral-7B-v0.1:** A 7 billion parameter model with approximately 14 GB checkpoint size. This model provides architectural diversity in the 7B parameter range.

### Parameter Sweeps

To identify optimal TensorStore configurations, systematic parameter sweeps were conducted:

**Compression Comparison:** Models were saved with both no compression and gzip compression to quantify the trade-off between file size reduction and performance overhead. The experiments used identical configurations except for the compression setting.

**Concurrency Levels:** Five concurrency levels were tested (1, 4, 16, 64, 128 concurrent operations) to determine the optimal parallelism for the storage hardware. Lower concurrency reduces scheduling overhead but may underutilize I/O bandwidth, while higher concurrency can saturate the storage system.

**Chunk Sizes:** Six chunk sizes were evaluated (1, 4, 16, 64, 128, 256 MB) to find the sweet spot between metadata overhead and I/O efficiency. Smaller chunks create excessive file operations, while larger chunks may prevent effective parallelization.

### Experimental Configuration

All experiments were conducted on a consistent hardware platform to ensure reproducibility:

| Property | Value |
|----------|-------|
| **Operating System** | Linux Ubuntu |
| **Architecture** | x86_64 |
| **CPU** | Intel Core i5-10300H @ 2.50GHz (8 cores) |
| **GPU** | NVIDIA GeForce GTX 1650 (4GB) |
| **RAM** | 16GB DDR4 |
| **Storage** | 1TB NVMe SSD |

Each checkpointing operation was performed three times, and the mean, standard deviation, minimum, and maximum values were recorded to account for system variability. Cache clearing was performed between runs on local systems to ensure accurate timing measurements, though this was disabled on cluster systems where sudo access was unavailable.

### TensorStore Configuration Details

The TensorStore implementation uses the following key parameters:

**Driver:** The Zarr driver was selected over N5 based on previous benchmarking showing superior performance for local file systems. Zarr provides better compatibility with Python's scientific computing ecosystem and offers more efficient metadata handling.

**Key-Value Store:** The 'file' kvstore driver was used for local filesystem storage, with paths automatically generated based on model names and timestamps. This ensures organized storage and prevents conflicts between different experimental runs.

**Metadata:** Each Zarr array includes metadata specifying the tensor's shape, data type (float16, float32, or bfloat16 converted to float16), and chunk dimensions. The metadata overhead becomes significant when storing hundreds of individual parameters.

**Compression:** When enabled, gzip compression at level 1 was used to balance compression ratio with computational overhead. Higher compression levels were avoided due to diminishing returns and increased CPU usage.

**Data Type:** Models were saved using their native precision when possible. For bfloat16 models, conversion to float16 was necessary because PyTorch's bfloat16 tensors cannot be directly converted to NumPy arrays, which TensorStore requires. This conversion maintains 2-byte storage efficiency while ensuring compatibility.

**Chunk Size:** Dynamic chunking was implemented where each tensor is divided into chunks of approximately 64 MB. For tensors smaller than the chunk size, a single chunk is used. For larger tensors, multiple chunks are created to enable parallel I/O.

**Concurrency:** The T5X-optimized approach uses a concurrency limit of 128, allowing up to 128 file operations to proceed simultaneously. This is controlled through TensorStore's context configuration.

---

## BENCHMARK AND EVALUATION

The experimental results reveal clear performance characteristics and trade-offs for each checkpointing approach across different configurations and model sizes.

### Phase Comparison: PyTorch vs TensorStore vs T5X

The comparison of the three checkpointing approaches across all four models shows consistent patterns. PyTorch's native checkpointing demonstrates superior performance for local disk operations, with save times ranging from 5 to 15 seconds for 3B models and 10 to 25 seconds for 7B models. Load times are similarly fast, typically 3 to 5 seconds for 3B models and 5 to 8 seconds for 7B models. The basic TensorStore implementation shows significantly slower performance, with save times of 140 to 300 seconds for 3B models and 250 to 450 seconds for 7B models. Load times range from 18 to 30 seconds for 3B models and 25 to 40 seconds for 7B models.

The T5X-optimized approach falls between the two extremes, with save times of 60 to 150 seconds for 3B models and 120 to 280 seconds for 7B models. Load times are 25 to 40 seconds for 3B models and 35 to 55 seconds for 7B models. File sizes are comparable across all three methods, ranging from 2.6 to 2.8 GB for 3B models and 6.5 to 6.9 GB for 7B models, with the T5X approach showing 2 to 5 percent reduction due to compression.

The performance differences stem from fundamental architectural differences. PyTorch's native serialization writes data sequentially with minimal overhead, leveraging the operating system's optimized file I/O. TensorStore's chunked format requires hundreds of separate file operations, each with associated metadata and filesystem overhead. The T5X optimizations improve upon basic TensorStore through higher concurrency and compression, but the fundamental overhead of the chunked format remains. For local disk operations, PyTorch's approach is **10 to 20 times faster for saves** and **5 to 8 times faster for loads**.

### Compression Impact Analysis

The compression experiments reveal that gzip compression provides minimal benefits for neural network checkpoints while imposing significant performance penalties. For OpenLLaMA-3B, save times increase from approximately 140 seconds without compression to 180 seconds with gzip, representing a 28 percent slowdown. Load times similarly increase from 18 to 20 seconds up to 25 to 30 seconds, a 35 to 50 percent increase. The file size reduction is minimal, decreasing from 2.60 GB to 2.55 GB, only a 2 percent reduction.

Llama-3.2-3B shows similar patterns, with save times increasing from 160 seconds to 200 seconds (25 percent slower) and load times from 20 seconds to 28 seconds (40 percent slower). File size reduction is again minimal at approximately 3 percent. The 7B models exhibit the same behavior with proportionally similar overhead.

The poor compression ratio occurs because modern neural network weights consist of floating-point numbers with high entropy. Unlike text or structured data, these numerical values lack the repetitive patterns that compression algorithms exploit. The compression overhead costs significant CPU time processing every byte of data during both save and load operations, while yielding minimal storage savings. For local storage where disk space is abundant, compression is counterproductive. However, in bandwidth-constrained scenarios such as cloud storage where network transfer costs dominate, the small file size reduction might justify the compression overhead.

### Concurrency Scaling Characteristics

The concurrency experiments demonstrate clear scaling patterns across all models. Both 7B models (Mistral-7B and Qwen2.5-7B) show dramatic performance improvements as concurrency increases from 1 to 16 concurrent operations. Mistral-7B save times decrease from approximately 450 seconds at concurrency=1 to 280 seconds at concurrency=16, a 38 percent improvement. Beyond concurrency=16, performance plateaus with minimal additional gains, reaching approximately 260 seconds at concurrency=128.

Qwen2.5-7B follows an identical pattern, dropping from 420 seconds to 250 seconds as concurrency increases to 16, then plateauing. Load times show similar trends but with smaller absolute improvements. The 3B models (Llama-3.2-3B and OpenLLaMA-3B) exhibit the same scaling behavior with proportionally faster absolute times due to their smaller size.

The performance improvements from increased concurrency occur because TensorStore can overlap I/O operations. At concurrency=1, each chunk must be written sequentially, forcing the system to wait for each disk operation to complete. Higher concurrency allows multiple chunks to be written simultaneously, hiding I/O latency and better utilizing available disk bandwidth. The plateau effect beyond concurrency=16 indicates that the storage system has reached its practical limits for parallel I/O.

Modern NVMe SSDs can efficiently handle 16 to 32 concurrent operations, but beyond this point, the disk controller becomes the bottleneck rather than the concurrency level. The lack of improvement from 64 to 128 concurrent operations indicates that the storage system is already saturated at 64, and additional concurrency only adds scheduling overhead without improving throughput. Interestingly, the percentage improvement from concurrency scaling is similar across both 3B and 7B models (approximately 50 percent reduction in save time from concurrency=1 to concurrency=16), suggesting that the benefits of parallelization scale proportionally with model size.

### Chunk Size Optimization

The chunk size experiments reveal a clear optimal range across all models. Very small chunk sizes (1 to 4 MB) result in significantly degraded performance, while larger chunk sizes (64 to 256 MB) provide optimal and stable performance. For the 7B models, save times at 1 MB chunks are approximately 400 to 450 seconds, decreasing sharply to 250 to 280 seconds at 64 MB chunks and remaining stable through 256 MB. The 3B models show similar patterns with proportionally faster times: 250 to 280 seconds at 1 MB decreasing to 140 to 160 seconds at 64 MB and above.

Load times follow identical trends, and file sizes remain constant across all chunk sizes, confirming that chunking is purely a storage organization strategy with no impact on total data volume. The consistency of this pattern across all four models, regardless of size or architecture, indicates that **64 MB is a robust default choice** for TensorStore chunking.

Small chunk sizes create excessive overhead because each chunk requires separate metadata, file operations, and I/O calls. With 1 MB chunks, a 7B model might be split into thousands of individual chunks, each requiring its own Zarr array metadata and file handle. This creates significant filesystem overhead and prevents efficient sequential I/O. Larger chunks (64 to 256 MB) reduce the number of separate operations and allow the storage system to perform more efficient sequential writes.

The performance plateau at 64 MB suggests this is the sweet spot where chunks are large enough to minimize overhead but small enough to allow reasonable parallelization. Beyond 64 MB, further increases provide no benefit because the chunks are already large enough to amortize the fixed costs of metadata and file operations. At 64 MB, a 7B model (approximately 14 GB) is divided into roughly 220 chunks, while a 3B model (approximately 6 GB) creates about 95 chunks. These chunk counts appear optimal for balancing parallelization opportunities with metadata overhead.

### Cross-Model Consistency

A striking finding across all experiments is the consistency of performance characteristics regardless of model architecture. The 3B models (OpenLLaMA-3B and Llama-3.2-3B) track each other closely across all parameter sweeps, with differences of less than 10 percent at any given configuration. Similarly, the 7B models (Qwen2.5-7B and Mistral-7B) show nearly identical behavior despite architectural differences.

This consistency indicates that TensorStore's performance is primarily determined by total checkpoint size rather than model-specific factors such as layer count, attention mechanisms, or parameter distribution. The scaling from 3B to 7B models is roughly linear with checkpoint size: 7B models take approximately 2 times longer to save and load compared to 3B models, matching their approximately 2 times larger checkpoint size.

This linear scaling has important implications for predicting performance with even larger models. A 13B model would be expected to take approximately 3 to 4 times longer than a 3B model, while a 70B model might require 10 to 15 times longer. However, the relative benefits of optimization strategies (concurrency, chunk size) should remain consistent across model sizes.

### Statistical Reliability

All experiments were conducted with three runs per configuration, and statistical measures (mean, standard deviation, minimum, maximum) were recorded. The standard deviation for save and load times was typically 5 to 10 percent of the mean value, indicating reasonable consistency across runs. Outliers were rare and usually attributable to system background processes.

The error bars in the generated plots provide confidence intervals showing the variability of measurements. For most configurations, the error bars are small relative to the differences between methods, confirming that the observed performance differences are statistically significant rather than measurement noise. The three-run approach provides a good balance between statistical reliability and experimental efficiency, though higher-stakes production deployments might benefit from additional runs.

---

## CONCLUSION

This comprehensive study of TensorStore for large language model checkpointing reveals clear performance characteristics and optimal configurations. For local disk operations, PyTorch's native checkpointing remains significantly faster than TensorStore, with **10 to 20 times better save performance** and **5 to 8 times better load performance**. TensorStore's advantages lie in distributed systems and cloud storage scenarios, not single-machine local disk operations.

The optimal TensorStore configuration for production use is **64 MB chunks, no compression, and concurrency=16**. This provides the best balance of performance, simplicity, and resource efficiency across different model sizes and hardware configurations. Compression provides minimal benefits (2 to 5 percent file size reduction) while imposing significant performance penalties (20 to 40 percent slower), making it worthwhile only when network bandwidth or storage costs are critical constraints.

Concurrency significantly improves TensorStore performance, but with diminishing returns beyond 16 concurrent operations. For most systems, concurrency=16 provides the optimal balance between performance and resource usage. Chunk size has a dramatic impact on performance, with 64 MB emerging as the optimal choice across all tested models from 3B to 7B parameters. This sweet spot appears to be model-size independent and should be used as a default.

Model size affects absolute performance but not relative scaling characteristics. Larger models take proportionally longer to save and load, but they benefit from concurrency and chunk size optimizations in the same proportions as smaller models. The consistency of performance patterns across different model architectures (OpenLLaMA, Llama, Qwen, Mistral) suggests that these findings generalize to other transformer-based models.

For research and development scenarios using local disk storage, PyTorch's native checkpointing should be preferred due to its superior performance and simplicity. TensorStore becomes advantageous in specific scenarios: distributed training where partial model loading is required, cloud storage environments where its chunked format reduces network transfer overhead, and systems requiring fine-grained parameter access without loading entire checkpoints.

The T5X optimizations (high concurrency and compression) improve upon basic TensorStore by approximately 50 percent for save operations, but still remain 5 to 10 times slower than PyTorch for local disk operations. These optimizations are most valuable in distributed cloud environments where they were originally designed, rather than single-machine scenarios.

---

## FUTURE WORK

Several promising directions emerge from this research:

**Extended Model Scale Testing:** Future work should evaluate TensorStore's performance with larger models (13B, 30B, 70B parameters) to verify whether the linear scaling observed from 3B to 7B models continues at higher scales. The metadata overhead might become more significant with models containing tens of thousands of parameters.

**Cloud Storage Evaluation:** While this study focused on local disk performance, TensorStore's primary advantages lie in cloud storage scenarios. Comprehensive testing with Google Cloud Storage and Amazon S3 would quantify TensorStore's benefits in distributed training environments where network bandwidth and latency dominate performance characteristics.

**Alternative Compression Algorithms:** Although gzip compression showed poor results, other algorithms optimized for numerical data (such as blosc or zstd) might provide better compression ratios with lower overhead. Specialized compression for floating-point data could potentially achieve meaningful size reductions.

**Distributed Training Integration:** Evaluating TensorStore in actual distributed training scenarios with multiple nodes would reveal its benefits for partial model loading and parameter sharding. The chunked format's advantages become more apparent when different nodes need access to different parameter subsets.

**C++ Implementation Comparison:** This study used TensorStore's Python API. Benchmarking the C++ version could reveal whether the Python overhead contributes significantly to the observed performance gap with PyTorch. The C++ implementation might offer better performance for latency-sensitive applications.

**Advanced Caching Mechanisms:** Implementing intelligent caching strategies that predict which parameters will be accessed could reduce load times for partial model loading scenarios. This would be particularly valuable for fine-tuning workflows where only specific layers are updated.

**Hybrid Approaches:** Exploring hybrid checkpointing strategies that use PyTorch for full checkpoints but TensorStore for incremental parameter updates could combine the strengths of both approaches. This might be valuable for continuous training scenarios with frequent checkpointing.

**Memory-Mapped Access:** Investigating memory-mapped file access for TensorStore checkpoints could enable lazy loading of parameters, reducing initial load times and memory requirements for large models. This would be particularly valuable when working with models that exceed available RAM.

---

## REFERENCES

[1] Google, "TensorStore," GitHub. [Online]. Available: https://github.com/google/tensorstore

[2] Google Research, "TensorStore for high-performance, scalable array storage," Google Research Blog. [Online]. Available: https://research.google/blog/tensorstore-for-high-performance-scalable-array-storage/

[3] Zarr, "Zarr Documentation," Zarr. [Online]. Available: https://zarr.dev/

[4] Saalfeld, S., "N5," GitHub. [Online]. Available: https://github.com/saalfeldlab/n5

[5] M. Wagenländer et al., "Tenplex: Dynamic parallelism for deep learning using parallelizable tensor collections," Proc. ACM SIGOPS 30th Symp. Operating Systems Principles, 2024, pp. 195–210. doi: 10.1145/3694715.3695975.

[6] Google, "TensorStore Index," Google. [Online]. Available: https://google.github.io/tensorstore/index.html

[7] H. Wong, "SPoSTG105s3: Poster presentation," SC23, [Online]. Available: https://sc23.supercomputing.org/proceedings/src_poster/poster_files/spostg105s3-file1.pdf

[8] Google, "Reading the Janelia FlyEM Hemibrain Dataset," TensorStore Python Tutorial. [Online]. Available: https://google.github.io/tensorstore/python/tutorial.html#reading-the-janelia-flyem-hemibrain-dataset

[9] Google, "gcs Key-Value Store driver," TensorStore Documentation. [Online]. Available: https://google.github.io/tensorstore/kvstore/gcs/index.html

[10] Hugging Face, "Transformers Documentation," [Online]. Available: https://huggingface.co/docs/transformers/

[11] PyTorch, "PyTorch Documentation," [Online]. Available: https://pytorch.org/docs/stable/index.html
