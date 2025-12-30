# Parallel Image Processing Pipeline  
**Serial – OpenMP – MPI Comparison**

**78.6% SPEEDUP COMPARED TO SERIAL THROUGH PARALLELIZATION**


This project implements the **same image processing pipeline** using three different execution models in order to analyze and compare:

- performance,
- scalability,
- memory models,
- and programming complexity.

The project is designed as a **step-by-step transition** from sequential execution to shared-memory parallelism and finally to distributed-memory parallelism.

---

## 🎯 Project Motivation

Modern image processing workloads are computationally expensive and naturally parallelizable.  
This project aims to answer the following questions:

- How does performance change when moving from serial execution to parallel execution?
- What are the conceptual and practical differences between **threads** and **processes**?
- How does explicit communication in MPI differ from implicit sharing in OpenMP?

To ensure a fair comparison, **the same algorithmic pipeline** is used in all implementations.

---

## 🧠 Image Processing Pipeline

Each input image is processed through three well-defined stages:

### 1️⃣ Preprocessing
- Input: RGB image (`CV_8UC3`)
- Conversion to grayscale using weighted RGB coefficients
- Normalization to floating-point values in the range `[0, 1]`

### 2️⃣ Processing
- A 3×3 convolution kernel is applied
- Kernel highlights intensity changes and emphasizes edges
- This stage is computationally the most expensive and is the main target for parallelization

### 3️⃣ Postprocessing
- Thresholding is applied to suppress weak responses
- Output becomes a binary edge map (`0` or `1`)
- Final result is converted back to 8-bit format for visualization

---

![araba](https://github.com/user-attachments/assets/34f0233f-98ef-47bc-a9da-ea53ee454283)


<img width="4897" height="3264" alt="sonuc_openmp" src="https://github.com/user-attachments/assets/f4eab18d-5c11-428b-b4fc-4bf81e406ea9" />

---

## ⚙️ Execution Models

### 🔹 Serial Implementation
- Single-threaded execution
- Processes one image at a time
- Serves as a **baseline** for correctness and performance comparison

---

### 🔹 OpenMP Implementation (Shared Memory)
- Uses multi-threading on a single machine
- Threads share the same address space
- Parallelization is achieved using `#pragma omp parallel for`
- Different scheduling strategies are explored

**Key characteristic:**  
Data sharing is implicit and managed by the runtime.

---

### 🔹 MPI Implementation (Distributed Memory)
- Each process has its **own private memory**
- No data is shared implicitly
- All communication is explicit and programmer-controlled

MPI collective operations used in this project:

- `MPI_Bcast`  
  Broadcasts global pipeline parameters (e.g. threshold value)

- `MPI_Scatter`  
  Distributes image indices among processes for workload partitioning

- `MPI_Reduce (MPI_MAX)`  
  Computes the total parallel execution time based on the slowest process

**Key characteristic:**  
Parallelism is achieved through **explicit message passing**, making memory ownership and data movement fully visible.

---

## 🛠️ Build System

The project uses **CMake** to ensure portability and clean builds.

### Requirements
- C++17 compatible compiler
- OpenCV
- OpenMP
- MPI (OpenMPI or MPICH)
- CMake ≥ 3.16


---



## 📈 Observations

- OpenMP offers fast development and low overhead for shared-memory systems.
- MPI introduces additional complexity due to explicit communication.
- MPI overhead dominates for small workloads.
- For larger workloads, MPI provides superior scalability potential.


---



### Build
```bash
rm -rf build
cmake -S . -B build
cmake --build build -j




