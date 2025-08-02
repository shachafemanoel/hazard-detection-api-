// sessionExample.js - Advanced session-based hazard tracking
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const API_URL = 'https://hazard-api-production-production.up.railway.app';

class HazardSession {
  constructor() {
    this.sessionId = null;
    this.detectedHazards = [];
    this.reports = [];
  }

  async start() {
    try {
      const response = await axios.post(`${API_URL}/session/start`);
      this.sessionId = response.data.session_id;
      console.log('🔄 Session started:', this.sessionId);
      return this.sessionId;
    } catch (error) {
      throw new Error(`Failed to start session: ${error.message}`);
    }
  }

  async detectInImage(imagePath) {
    if (!this.sessionId) {
      throw new Error('Session not started. Call start() first.');
    }

    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imagePath));

      const response = await axios.post(
        `${API_URL}/detect/${this.sessionId}`, 
        formData, 
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      const result = response.data;
      
      // Track detections
      this.detectedHazards.push(...result.detections);
      this.reports.push(...result.new_reports);

      console.log(`📸 Processed: ${path.basename(imagePath)}`);
      console.log(`  Found: ${result.detections.length} detections`);
      console.log(`  New reports: ${result.new_reports.length}`);
      console.log(`  Processing time: ${result.processing_time_ms}ms`);

      return result;
    } catch (error) {
      throw new Error(`Detection failed: ${error.message}`);
    }
  }

  async getSummary() {
    if (!this.sessionId) {
      throw new Error('Session not started');
    }

    try {
      const response = await axios.get(`${API_URL}/session/${this.sessionId}/summary`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get summary: ${error.message}`);
    }
  }

  async confirmReport(reportId) {
    if (!this.sessionId) {
      throw new Error('Session not started');
    }

    try {
      const response = await axios.post(
        `${API_URL}/session/${this.sessionId}/report/${reportId}/confirm`
      );
      console.log(`✅ Report confirmed: ${reportId}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to confirm report: ${error.message}`);
    }
  }

  async dismissReport(reportId) {
    if (!this.sessionId) {
      throw new Error('Session not started');
    }

    try {
      const response = await axios.post(
        `${API_URL}/session/${this.sessionId}/report/${reportId}/dismiss`
      );
      console.log(`❌ Report dismissed: ${reportId}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to dismiss report: ${error.message}`);
    }
  }

  async end() {
    if (!this.sessionId) {
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/session/${this.sessionId}/end`);
      console.log('🏁 Session ended successfully');
      return response.data;
    } catch (error) {
      console.error('⚠️ Error ending session:', error.message);
    } finally {
      this.sessionId = null;
    }
  }

  printStatistics() {
    const hazardTypes = {};
    this.detectedHazards.forEach(hazard => {
      hazardTypes[hazard.class_name] = (hazardTypes[hazard.class_name] || 0) + 1;
    });

    console.log('\n📊 Session Statistics:');
    console.log(`Total detections: ${this.detectedHazards.length}`);
    console.log(`Total reports: ${this.reports.length}`);
    console.log('\nHazard breakdown:');
    Object.entries(hazardTypes).forEach(([type, count]) => {
      console.log(`  ${type}: ${count}`);
    });

    const avgConfidence = this.detectedHazards.length > 0 
      ? this.detectedHazards.reduce((sum, h) => sum + h.confidence, 0) / this.detectedHazards.length
      : 0;
    console.log(`Average confidence: ${(avgConfidence * 100).toFixed(1)}%`);
  }
}

async function runSessionExample() {
  const session = new HazardSession();
  
  try {
    console.log('🚀 Session-based Hazard Detection Example');
    console.log('==========================================\n');

    // Start session
    await session.start();

    // Example images to process (replace with your images)
    const imagePaths = [
      './images/road1.jpg',
      './images/road2.jpg',
      './images/road3.jpg'
    ];

    // Filter to only existing images
    const existingImages = imagePaths.filter(fs.existsSync);
    
    if (existingImages.length === 0) {
      console.log('⚠️ No test images found. Please add images to ./images/ directory');
      console.log('Expected files:', imagePaths.map(p => path.basename(p)).join(', '));
      return;
    }

    console.log(`Processing ${existingImages.length} images...\n`);

    // Process each image
    for (const imagePath of existingImages) {
      await session.detectInImage(imagePath);
    }

    // Get session summary
    console.log('\n📋 Getting session summary...');
    const summary = await session.getSummary();
    
    console.log(`\nSession Summary:`);
    console.log(`Session ID: ${summary.id}`);
    console.log(`Start time: ${summary.start_time}`);
    console.log(`Total detections: ${summary.detection_count}`);
    console.log(`Unique hazards: ${summary.unique_hazards}`);

    // Review pending reports
    const pendingReports = summary.reports.filter(r => r.status === 'pending');
    console.log(`\n📝 Pending reports: ${pendingReports.length}`);

    // Auto-confirm high confidence reports
    for (const report of pendingReports) {
      const confidence = report.detection.confidence;
      const hazardType = report.detection.class_name;
      
      console.log(`\n🔍 Reviewing: ${hazardType} (${(confidence * 100).toFixed(1)}% confidence)`);
      
      if (confidence > 0.8) {
        await session.confirmReport(report.report_id);
      } else if (confidence < 0.4) {
        await session.dismissReport(report.report_id);
      } else {
        console.log(`⏳ Requires manual review: ${report.report_id}`);
      }
    }

    // Print statistics
    session.printStatistics();

  } catch (error) {
    console.error('❌ Session error:', error.message);
  } finally {
    // Always end the session
    await session.end();
  }
}

// Run the example
runSessionExample();