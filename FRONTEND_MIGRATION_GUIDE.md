# Frontend Migration Guide: Report Management API

This guide explains how to update the frontend to use the new centralized report management API.

For general client integration and response handling, see the [Client Guide](CLIENT_GUIDE.md).

## Overview

The report management system has been migrated from the Node.js frontend to the FastAPI backend for:
- Real-time report creation during detection
- Better performance and scalability
- Centralized data management
- Enhanced filtering and analytics

## API Endpoints

### Base URL
```
Production: https://hazard-api-production-production.up.railway.app
Development: http://localhost:8080
```

### Available Endpoints

#### 1. Create Report
```http
POST /reports
Content-Type: application/json

{
  "detection": {
    "class_id": 0,
    "class_name": "Pothole",
    "confidence": 0.85,
    "bbox": [100.0, 50.0, 200.0, 150.0],
    "area": 10000.0,
    "center_x": 150.0,
    "center_y": 100.0
  },
  "image_data": "data:image/jpeg;base64,/9j/4AAQ...",
  "description": "Pothole on Main Street",
  "severity": "high",
  "tags": ["road-damage", "urgent"]
}
```

#### 2. List Reports
```http
GET /reports?status=pending&limit=20&page=1&sort_by=created_at&sort_order=desc
```

#### 3. Get Report
```http
GET /reports/{report_id}
```

#### 4. Update Report
```http
PATCH /reports/{report_id}
Content-Type: application/json

{
  "status": "confirmed",
  "description": "Updated description",
  "severity": "critical"
}
```

#### 5. Delete Report
```http
DELETE /reports/{report_id}
```

#### 6. Confirm Report
```http
POST /reports/{report_id}/confirm
Content-Type: application/json

{
  "notes": "Confirmed after inspection"
}
```

#### 7. Dismiss Report
```http
POST /reports/{report_id}/dismiss
Content-Type: application/json

{
  "reason": "False positive"
}
```

#### 8. Get Statistics
```http
GET /reports/stats
```

#### 9. File Upload
```http
POST /reports/upload
Content-Type: multipart/form-data

file: [image file]
detection_data: [JSON string]
metadata: [JSON string]
```

## Frontend Code Updates

### 1. Update reports-api.js

Replace the existing reports-api.js with API calls to the new endpoints:

```javascript
const API_BASE_URL = 'https://hazard-api-production-production.up.railway.app';

// Create report
export async function createReport(reportData) {
  const response = await fetch(`${API_BASE_URL}/reports`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(reportData)
  });
  
  if (!response.ok) {
    throw new Error(`Failed to create report: ${response.statusText}`);
  }
  
  return await response.json();
}

// Fetch reports with filters
export async function fetchReports(filters = {}) {
  const params = new URLSearchParams();
  
  // Add filters to params
  if (filters.status) params.append('status', filters.status);
  if (filters.session_id) params.append('session_id', filters.session_id);
  if (filters.min_confidence) params.append('min_confidence', filters.min_confidence);
  if (filters.page) params.append('page', filters.page);
  if (filters.limit) params.append('limit', filters.limit);
  
  const response = await fetch(`${API_BASE_URL}/reports?${params.toString()}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch reports: ${response.statusText}`);
  }
  
  const data = await response.json();
  return {
    reports: data.reports,
    pagination: data.pagination,
    metrics: {
      total: data.total_count
    }
  };
}

// Update report
export async function updateReport(reportId, updateData) {
  const response = await fetch(`${API_BASE_URL}/reports/${reportId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(updateData)
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update report: ${response.statusText}`);
  }
  
  return await response.json();
}

// Delete report
export async function deleteReportById(reportId) {
  const response = await fetch(`${API_BASE_URL}/reports/${reportId}`, {
    method: 'DELETE'
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete report: ${response.statusText}`);
  }
  
  return await response.json();
}

// Get report statistics
export async function getReportStats() {
  const response = await fetch(`${API_BASE_URL}/reports/stats`);
  
  if (!response.ok) {
    throw new Error(`Failed to get stats: ${response.statusText}`);
  }
  
  return await response.json();
}

// Confirm report
export async function confirmReport(reportId, notes = '') {
  const response = await fetch(`${API_BASE_URL}/reports/${reportId}/confirm`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ notes })
  });
  
  if (!response.ok) {
    throw new Error(`Failed to confirm report: ${response.statusText}`);
  }
  
  return await response.json();
}
```

### 2. Update Data Formats

The new API uses different field names and structures:

#### Old Format:
```javascript
{
  id: "report-123",
  location: [lat, lon],  // Array format
  image: {
    url: "cloudinary-url"
  },
  // ... other fields
}
```

#### New Format:
```javascript
{
  id: "report-123",
  location: {
    latitude: lat,
    longitude: lon,
    bbox: [x1, y1, x2, y2],
    center: [x, y]
  },
  image: {
    url: "cloudinary-url",
    public_id: "cloudinary-id",
    width: 640,
    height: 480,
    thumbnail_url: "thumbnail-url"
  },
  detection: {
    class_id: 0,
    class_name: "Pothole",
    confidence: 0.85,
    bbox: [x1, y1, x2, y2],
    area: 10000.0,
    center_x: 150.0,
    center_y: 100.0
  },
  status: "pending",  // pending, confirmed, dismissed
  severity: "high",   // low, medium, high, critical
  tags: [],
  metadata: {
    session_id: "session-123",
    source: "detection_pipeline"
  }
}
```

### 3. Update Status Management

The new system uses different status values:
- `pending` → Report awaiting review
- `confirmed` → Report confirmed and valid
- `dismissed` → Report dismissed as invalid
- `archived` → Report archived (future use)

### 4. Update Filtering

New filtering options available:
```javascript
const filters = {
  status: ['pending', 'confirmed'],  // Array of statuses
  severity: ['high', 'critical'],    // Array of severities
  class_ids: [0, 1, 2],              // Array of detection class IDs
  min_confidence: 0.7,               // Minimum confidence threshold
  max_confidence: 1.0,               // Maximum confidence threshold
  session_id: 'session-123',         // Filter by session
  source: 'detection_pipeline',      // Filter by source
  date_from: '2024-01-01T00:00:00',  // ISO date string
  date_to: '2024-12-31T23:59:59',    // ISO date string
  page: 1,                           // Page number
  limit: 20,                         // Items per page
  sort_by: 'created_at',             // Sort field
  sort_order: 'desc'                 // asc or desc
};
```

## Migration Steps

1. **Update API Base URL**: Change all API calls to use the new FastAPI endpoint
2. **Update Data Models**: Modify frontend data handling to work with new response formats
3. **Update Filtering**: Implement new filtering options and parameters
4. **Add Status Management**: Update UI to handle new status workflow
5. **Update Error Handling**: Handle new error response formats
6. **Test Integration**: Thoroughly test all report operations
7. **Remove Old Code**: Remove old Node.js report upload service files

## Benefits of Migration

- ✅ **Real-time Reports**: Automatic report creation during detection
- ✅ **Better Performance**: Direct database operations, no intermediate API calls
- ✅ **Enhanced Filtering**: More flexible and powerful filtering options
- ✅ **Centralized Logic**: All report logic in one place
- ✅ **Better Analytics**: Rich statistics and aggregation capabilities
- ✅ **Improved Reliability**: Robust error handling and validation
- ✅ **Scalability**: Built for high-volume report processing

## Testing

Use the provided example script to test the new API:

```bash
python examples/report_management_example.py
```

This will demonstrate all major report operations and help verify the migration is successful.