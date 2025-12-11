Checkpointing using Tensorstore

     	     Zia Uddin Chowdhury							PhD Lead – Jie Ye  
[zchowdhury1@hawk.iit.edu](mailto:zchowdhury1@hawk.iit.edu)					          	           [jye20@hawk.iit.edu](mailto:jye20@hawk.iit.edu) 

INTRODUCTION   
Managing and processing large multi-dimensional datasets is a significant challenge in data-intensive fields like neuroscience and machine learning. Even a single dataset may require terabytes or petabytes of data storage. Such datasets are also challenging to work with as users may read and write data at irregular intervals and varying scales. TensorStore \[1\] is an open-source C++ and Python library designed to read and write large multi-dimensional arrays efficiently. It provides a uniform API for reading and writing different array formats like Zarr and N5. It also integrates with various storage backends, including in-memory storage, local file systems, Amazon S3, and Google Cloud Storage to store the array dataset. One challenge that arises during this training process is efficiently reading and writing the model parameters and TensorStore has already been used to address these challenges. It has been applied to manage checkpoints associated with large-scale models trained with JAX and has been integrated with frameworks such as T5X and Pathways. Model parallelism is used to partition the full set of parameters, which can occupy more than a terabyte of memory, over hundreds of TPUs. TensorStore can be used to address computational challenges in large-scale connectomic datasets, efficiently managing some of the largest and most accessed datasets with Google Cloud Storage as the underlying object storage system \[2\]

BACKGROUND   
TensorStore is an open-source Python and C++ library made for effective multi-dimensional array manipulation and storing. It is especially handy for dealing with big datasets that cannot be stored in memory. It supports Zarr and N5 for efficient multi-dimensional data storage and retrieval. The Zarr data format is a community-maintained format for large-scale n-dimensional data and it enables simple, fast serialization of NumPy-like arrays and supports multi-scale n-dimensional image storage for applications like light and electron microscopy \[3\] It is a cloud-friendly format that enables chunked, compressed storage for distributed and parallel processing. On the other hand, N5 is optimized for large-scale scientific datasets like bioinformatics and microscopy. The N5 API specifies the primitive operations needed to store large chunked n-dimensional tensors, and arbitrary meta-data in a hierarchy of groups \[4\] Besides those, Google Cloud Storage (GCS) serves as a scalable backend for storing large, multi-dimensional datasets. TensorStore facilitates parallel I/O operations and smooth integration with cloud-based workflows by utilizing GCS's stability and performance to effectively handle and retrieve large volumes of scientific and machine learning data. 

RELATED WORKS   
In this paper, the bandwidth of checkpoint creation was measured with and without TensorStore CHFS using T5X. To measure the bandwidth, the parameter size of the T5 1.1 model was changed, and the model was divided into 8, 16, and 32 nodes \[5\] Furthermore, the StateTransformer inputs model and dataset partitions from a previous PTC, creates updated partitions for a new PTC' after a resource change, while the TensorStore maintains the model and dataset state partitions in a hierarchical virtual in-memory file system, offering APIs for model checkpointing and enabling the ingestion of training data \[6\] The neuroglancer\_precomputed driver is used to access the Janelia FlyEM Hemibrain 1.1 segmentation dataset, as shown in the TensorStore Python tutorial.  The dataset is stored in uint64 format with a resolution of 8x8x8 microns per voxel, and users can open it asynchronously by providing the Google Cloud Storage path (gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation/).  Using TensorStore's caching and parallelism features, the tutorial shows how to create 3D views, slice data, and read particular sections of the given dataset \[8\]

APPROACH   
Detailed explanation of the components:

* driver: The driver determines the storage format used to structure and manage the dataset. Here I have given N5 as the driver, which means it would follow N5 format, a format for storing large data. The N5 format is widely used for scientific and high-dimensional data due to its hierarchical structure for efficient retrieval and chunked storage for faster access. Besides that, Zarr format would also be used as the data storage, where the driver would just be labelled as ‘zarr’ instead of ‘n5’.   
* kvstore: kvstore represents the key-value store, which manages how data is stored and accessed. The layer facilitates communication between TensorStore and various storage backends, such as cloud storage and local file systems. The types of drivers that can be used are ‘file’, ‘gcs’ and ‘s3’ as described below:  
* ‘file’: saves data in a directory on the local filesystem, and the ‘path’ would be the local path in which the data would be stored.    
* ‘gcs’: uses Google Cloud Storage for remote storage. For these cloud storages, it would also have an additional property \- ‘bucket’, which would represent the name of the bucket in Google Cloud Storage. The detailed steps to configure GCS for TensorStore are outlined here: [https://google.github.io/tensorstore/kvstore/gcs/index.html](https://google.github.io/tensorstore/kvstore/gcs/index.html) \[9\]  
* ‘s3’: it uses cloud storage of AWS and has a similar property of bucket, as that of Google Cloud Storage. The ‘path’ would be the virtual directory to store the dataset on the cloud.   
* metadata: It defines key properties of the dataset, such as its shape, data type, how it is structured and how it is divided into smaller chunks for storage optimization.    
* ‘compression’: Compression is the process of reducing the size of stored data by encoding it in a more efficient way. In TensorStore, compression is crucial for optimizing storage space, improving read/write performance, and reducing I/O overhead. It ensures reduced storage size and optimized data retrieval performance. One of the compression type is ‘gzip’ which is one of the widely used algorithms, as well as it provides a good balance between compression ratio and speed. Either the compression type can be none or there are others such as ‘zlib’ and ‘blosc’. Different compression methods offer trade-offs between speed, storage efficiency, and computational overhead.   
* ‘dataType’: The dataType parameter in TensorStore defines the type of values stored in the dataset. Choosing the right data type impacts memory usage, computational efficiency, and precision. For example, int8, int16, int32, and int64 can be used for storing signed data whereas uint32 and uint64 can be used for storing unsigned data, float16, float32 can be used for storage of decimal values and bool can be used for the storage of boolean values.    
* ‘dimension’: This represents the shape of the tensor, creating a table-like structure for the dataset. For example, a dimension of \[1000, 2000\] represents a dataset with 1000 rows and 2000 columns. So, this allows the usage of multi-dimensional dataset.    
* ‘blockSize’: It refers to the smaller chunks of data, as the dataset is divided into smaller blocks (chunks) for efficient storage and retrieval. Chunking is crucial for handling large datasets, as reading/writing smaller chunks improves performance over working with the entire dataset at once. So a blockSize of \[10, 10\] would mean that each small chunk of data consists of 10 rows and 10 columns.    
* fill\_value: When a new dataset is created, instead of leaving empty spaces, all elements are automatically initialized to zero, reducing the hassle of manual insertion of data.    
* create: If this is set to true, a new dataset is created if it doesn’t already exist.   
* delete\_existing: Setting delete\_existing to True ensures that any existing dataset at the same path is deleted before creating a new one. This ensures that old data doesn’t interfere with new one.

Test call for n5 baseline:  
total\_size\_mb \= 1024  
block\_sizes\_mb \= \[4, 16, 64, 256, 1024\]    
data\_type \= 'int32'  
driver \= 'n5'  
file \= 'file'  
driver\_path \= 'dataset/improved/n5'  
compression \= 'none'  
file\_io\_limit \= 'shared'  
file\_io\_sync \= True

\# call the following in a for loop for the varying block size:  
dims \= calculate\_dimensions(total\_size\_mb, block\_mb, data\_type)  
write\_tensor(driver, file, driver\_path, data\_type, \*dims, compression, file\_io\_limit, file\_io\_sync)  
read\_tensor(driver, file, driver\_path, dims\[1\], file\_io\_limit, file\_io\_sync)

It’s optimal to call the write\_tensor() and read\_tensor() from separate ‘for’ loops for varying quantity accordingly, and even nested loops where appropriate, instead of calling them together from the same ‘for’ loop.

All of the tests are carried out in the PC with the following configuration:

| Property | Value |
| :---: | :---: |
| OS | Linux Ubuntu |
| Architecture | x86\_64 |
| CPU op-mode(s) | 32-bit, 64-bit |
| CPU(s) | 8 |
| On-line CPU(s) list | 0-7 |
| Model name | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| Graphics Card | NVIDIA GeForce GTX 1650 |
| RAM | 16GB |
| Storage | NVMe SSD |
| Storage Capacity | 1 TB |

BENCHMARK AND EVALUATION 

CONCLUSION   
In conclusion, the performance benchmarking of TensorStore was conducted using a consistent baseline configuration: a dataset size of 1GB, block sizes of 4, 16, 64, 256, and 1024, data type set to int32, driver as n5, with file\_io\_sync set to true and file\_io\_concurrency set to shared. Various parameters were then systematically altered to identify the optimal configuration. For local file system usage, the best performance was observed when using the ‘zarr’ as the driver, zstd compression algorithm, float64 as the data type, file\_io\_sync set to true, and file\_io\_concurrency as shared. This combination consistently yielded the fastest and most reliable results. Similar tests with the change of parameters were conducted using Zarr, following the same approach as with N5, and the results consistently showed that Zarr significantly outperformed N5. Although using Google Cloud Storage under the same settings with N5 resulted in roughly 20 times longer execution times, it offers the notable advantage of offloading local storage requirements, making it a practical alternative whereas storage is limited. Additionally, while GCS introduces latency, its scalability and accessibility makes it suitable for distributed or cloud-native applications, eliminating the need for a local file storage system. 

FUTURE WORK

1) Expanding Feature Set: Future work could involve incorporating advanced caching mechanisms and support for more cloud storage providers, including Amazon S3, to enhance performance and scalability in TensorStore.

2) Testing the C++ Version: A potential future direction is to explore and benchmark the C++ version of TensorStore to evaluate its performance and compare it with the Python version, along with using all the parameters as used here.

3) Comprehensive Testing with Zarr: Could include comprehensive testing with Zarr, ensuring full compatibility and performance optimization for large-scale data storage and retrieval, as well as implementing the other possible combinations of parameters available.  
     
4) Integrating with S3: Additionally integrating TensorStore with Amazon S3, enabling efficient data storage and access across scalable cloud environments as well as comparing the results with Google Cloud Storage (GCS) to figure out the better cloud based storage in regards to usage of TensorStore.

REFERENCES  
\[1\] Google, "TensorStore," GitHub. \[Online\]. Available: [https://github.com/google/tensorstore](https://github.com/google/tensorstore)   
\[2\] Google Research, "TensorStore for high-performance, scalable array storage," Google Research Blog. \[Online\]. Available: [https://research.google/blog/tensorstore-for-high-performance-scalable-array-storage/](https://research.google/blog/tensorstore-for-high-performance-scalable-array-storage/)   
\[3\] Zarr, "Zarr Documentation," Zarr. \[Online\]. Available: [https://zarr.dev/](https://zarr.dev/)   
\[4\] Saalfeld, S., "N5," GitHub. \[Online\]. Available: [https://github.com/saalfeldlab/n5](https://github.com/saalfeldlab/n5)   
\[5\] M. Wagenländer et al., "Tenplex: Dynamic parallelism for deep learning using parallelizable tensor collections," Proc. ACM SIGOPS 30th Symp. Operating Systems Principles, 2024, pp. 195–210. doi: 10.1145/3694715.3695975.  
\[6\] Google, "TensorStore Index," Google. \[Online\]. Available: [https://google.github.io/tensorstore/index.html](https://google.github.io/tensorstore/index.html)   
\[7\] H. Wong, "SPoSTG105s3: Poster presentation," SC23, \[Online\]. Available: [https://sc23.supercomputing.org/proceedings/src\_poster/poster\_files/spostg105s3-file1.pdf](https://sc23.supercomputing.org/proceedings/src_poster/poster_files/spostg105s3-file1.pdf)   
\[8\] Google, "Reading the Janelia FlyEM Hemibrain Dataset," TensorStore Python Tutorial. \[Online\]. Available: [https://google.github.io/tensorstore/python/tutorial.html\#reading-the-janelia-flyem-hemibrain-dataset](https://google.github.io/tensorstore/python/tutorial.html#reading-the-janelia-flyem-hemibrain-dataset)  
\[9\] Google, "gcs Key-Value Store driver," TensorStore Documentation. \[Online\]. Available: [https://google.github.io/tensorstore/kvstore/gcs/index.html](https://google.github.io/tensorstore/kvstore/gcs/index.html) 

