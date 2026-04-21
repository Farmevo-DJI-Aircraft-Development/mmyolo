# DJI Developer Support — AI Inside Bounding Box Access via MSDK v5

## Summary

We are using the DJI AI Inside program to deploy a custom YOLOv8 object detection model (trained via MMYOLO) onto a **DJI Matrice 4E**. The model is trained, quantized via DJI's portal, and deployed to the drone's on-board NPU.

Our goal is to **receive the bounding box coordinates (class, confidence, x, y, width, height) from the on-device inference results inside a custom Android app built with MSDK v5.**

---

## What We Have

- Custom YOLOv8-s model trained using the DJI AI Inside patch (`0001-NEW-ai-inside-init.patch`, authored by `developer@dji.com`)
- Model quantized and deployed to the Matrice 4E via DJI's AI Inside portal
- Inference running on-device and results visible in **DJI Pilot 2**

---

## The Problem

MSDK v5 does not appear to expose a public API for accessing AI Inside inference results (bounding boxes, class labels, confidence scores) in a custom mobile application.

We investigated the following MSDK v5 components and found none of them suitable:

| Component | Why it doesn't work |
|---|---|
| `IPerceptionManager` | Obstacle avoidance sensor data only — no AI inference results |
| `ICameraStreamManager` | Provides raw video frames for client-side inference only — no on-device results |
| `IIntelligentBoxManager` | Manifold 3 hardware metrics (CPU/GPU/power) and app management only — no model outputs |
| `IIntelligentFlightManager` | DJI's built-in subject tracking only — not custom model outputs |

**DJI Pilot 2 is able to display bounding boxes from on-device inference**, which confirms the inference results are accessible at the firmware level — but this appears to use internal APIs not exposed in the public MSDK v5.

---

## Questions for DJI Support

1. **Is there a public MSDK v5 API** that exposes AI Inside inference results (bounding boxes, class labels, confidence scores) directly to a custom mobile app?

2. **Is the internal protocol used by DJI Pilot 2** to display bounding boxes accessible to third-party MSDK developers in any form?

3. **Is a dedicated AI Inside result listener or callback planned** for a future MSDK v5 release?

4. **What is the recommended approach** for a custom MSDK v5 app to receive real-time bounding box detections from an on-device AI Inside model on the Matrice 4E?

---

## Our Target Workflow

```
Matrice 4E NPU
    (runs quantized YOLOv8-s model)
            ↓
    Bounding box detections
            ↓
    Custom Android App (MSDK v5)
            ↓
    Application logic (alerts, logging, display)
```

We want to avoid running inference client-side on the mobile device, as the model is already deployed and running on the drone's NPU.

---

## Environment

- **Drone**: DJI Matrice 4E
- **SDK**: MSDK v5 (Android)
- **Model**: YOLOv8-s (custom trained, AI Inside quantized)
- **Training framework**: MMYOLO with DJI AI Inside patch
