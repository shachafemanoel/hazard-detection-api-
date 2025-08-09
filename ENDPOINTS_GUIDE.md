# Image and Plot Endpoints Guide

This guide explains how to retrieve images and model response plots from a live detection session.

For general API usage and sending detection requests, see the [Client Guide](CLIENT_GUIDE.md).

## Prerequisites
1. Start a session using `POST /session/start`.
2. Run detections with `POST /detect/{session_id}` to generate reports.

Each detection that meets the confidence threshold can create a report containing the original frame. Use the endpoints below to fetch these images or an annotated plot.

## Get Original Report Image
**GET `/session/{session_id}/report/{report_id}/image`**

Returns the original image that triggered the report.

```bash
# Save the image to disk
curl -o report.jpg "https://YOUR_API/session/SESSION_ID/report/REPORT_ID/image"
```

Response: JPEG image stream (`content-type: image/jpeg`).

## Get Annotated Model Response Plot
**GET `/session/{session_id}/report/{report_id}/plot`**

Returns the image with detection bounding box and label drawn on top.

```bash
# Download annotated plot
curl -o plot.jpg "https://YOUR_API/session/SESSION_ID/report/REPORT_ID/plot"
```

Response: JPEG image stream (`content-type: image/jpeg`).

These endpoints make it easy to review detections visually during a live session.
