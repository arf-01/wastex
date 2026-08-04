# WasteX Engineering Practice & Learning Roadmap

This document outlines a structured learning and implementation roadmap for your work on [WasteX](file:///c:/WASTE/wastex). Because you have moved to **Version 2** (where the central node is purely a data collector and the complex training logic is decoupled), you have a massive opportunity to make this a **standout portfolio project**.

To impress engineering teams, a portfolio project needs to show that you care about **constraints (memory/CPU)**, **reliability (networking drops)**, and **real-world edge conditions**.

---

## 🌟 How to Make This "Portfolio-Worthy"

Most beginners just train a standard heavy model, wrap it in Flask, and call it a day. To show true **Software Engineering** and **MLOps** capabilities, you should focus on the following pillars:

1. **Edge ML Optimization (TinyML)**: Prove you can deploy ML on constrained hardware (Raspberry Pi) without melting the CPU.
2. **Network Resilience**: Prove your system doesn't crash or lose data when the Wi-Fi drops.
3. **Observability**: Prove you can measure your system's performance in production.

---

## 🚀 Module 1: Edge ML & Model Optimization

Running standard machine learning models in raw TensorFlow on a Raspberry Pi is slow and consumes too much RAM.

### 1. Quantization & TFLite / ONNX Export
- **Why it matters**: ML Engineers rarely deploy raw `.keras` or `.h5` files to production. They optimize them.
- **Your Task**:
  - Write a script to convert your lightweight model to **TensorFlow Lite (.tflite)**.
  - Apply **INT8 Quantization** (Post-training quantization) to reduce the model size by 4x.
  - Modify `pi/scripts/image_watcher.py` to use `tflite_runtime` instead of standard `tensorflow`. This avoids installing the massive 500MB TensorFlow package on the Pi.

### 2. Hardware Profiling (The "Wow" Factor for Resumes)
- **Why it matters**: Metrics prove engineering capability.
- **Your Task**:
  - Create a benchmark script on the Pi.
  - Measure **Inference Latency (ms)**, **Memory Usage (MB)**, and **CPU Temperature (°C)** before and after your optimizations. 
  - *Resume Bullet Point: "Reduced edge inference latency by X% and memory footprint by Y% by optimizing a MobileNetV3 model with INT8 quantization and TFLite conversion."*

---

## 🔄 Module 2: Distributed Systems & Network Resilience

Your Master node is now an OOD (Out of Distribution) collector. But what happens if the Raspberry Pi loses internet connection for 3 hours?

### 1. Local Write-Ahead Log (WAL) / Queue on Edge
- **Why it matters**: "It works on my local Wi-Fi" is not a production guarantee. Edge networks are notoriously flaky.
- **Your Task**:
  - Update the Pi sync script so that when it detects an OOD image, it saves the image to a local directory and writes a record to a local **SQLite database** first.
  - A separate background thread should continuously read from this local SQLite DB and attempt to push to the Master node. 
  - If the push succeeds, delete the local record and image. If it fails, keep it.

### 2. Exponential Backoff & Jitter
- **Why it matters**: If 50 smart bins go offline and suddenly reconnect, they shouldn't all hammer the server at the exact same millisecond (Thundering Herd problem).
- **Your Task**:
  - Implement retry logic in your sync script. If the Master is down, wait 2 seconds, then 4 seconds, then 8 seconds, etc. (Exponential Backoff), adding a tiny random delay (Jitter) so multiple bins don't sync simultaneously.

---

## 🌐 Module 3: Backend Engineering (The Master Collector)

The Master node is responsible for receiving images and displaying them to admins. 

### 1. Asynchronous Webhooks & High-Performance Endpoints
- **Why it matters**: If a Pi uploads an image, the Django view shouldn't block while processing or verifying the image.
- **Your Task**:
  - Use **Django Ninja** or **Django REST Framework** to build a clean, typed API endpoint for the Pi to push images to.
  - Ensure the OOD Gallery implements pagination (so loading the page doesn't crash if there are 10,000 OOD images).

### 2. Data Lifecycle Management
- **Why it matters**: Storage isn't infinite.
- **Your Task**:
  - Write a Django Management command (`clean_stale_ood.py`) that deletes OOD images older than 30 days that an admin hasn't labeled.
  - Hook this up to a simple cron job.

---

## 🔍 Module 4: Observability (Standing out as a Pro)

### 1. Edge Heartbeats & Logging
- **Why it matters**: How do you know if a bin is online?
- **Your Task**:
  - Make the Pi send a lightweight "Heartbeat" ping to the Master node every 60 seconds (containing CPU temp and free disk space).
  - Show a "Bin Status" dashboard on the Master node (Online/Offline, Last Seen, Storage Full Warning).

### 2. Proper Python Logging
- **Why it matters**: `print()` is for scripts. `logging` is for software.
- **Your Task**:
  - Implement Python's `logging` module across the Pi scripts and Django backend. 
  - Log to rotating files so the Pi's SD card doesn't fill up with text logs over time.

---

## 📆 Recommended Execution Plan

| Phase | Focus Area | Goal |
| :--- | :--- | :--- |
| **Phase 1** | **Edge Optimization** | Optimize your model using quantization and TFLite. Measure the performance gains. |
| **Phase 2** | **Resiliency** | Build the local SQLite queue on the Pi for offline image caching and exponential backoff sync. |
| **Phase 3** | **Observability** | Add heartbeats and robust rotating logs. Build the "Bin Status" view on the Master node. |
| **Phase 4** | **Documentation** | Write a killer README. Add architecture diagrams. Document your performance benchmarks (Before vs After). |
