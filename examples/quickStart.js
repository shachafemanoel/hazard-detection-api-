// quickStart.js - Simple example to get started quickly
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_URL = 'https://hazard-api-production-production.up.railway.app';

async function quickDetection() {
  try {
    console.log('🚀 Quick Hazard Detection Example');
    console.log('===================================\n');

    // 1. Check if service is available
    console.log('1️⃣ Checking service health...');
    const healthResponse = await axios.get(`${API_URL}/health`);
    console.log('✅ Service Status:', healthResponse.data.status);
    console.log('📊 Model Backend:', healthResponse.data.backend_type);
    console.log('⚡ Performance Mode:', healthResponse.data.device_info?.performance_mode);

    // 2. Detect hazards in a single image
    console.log('\n2️⃣ Detecting hazards in image...');
    
    // Replace with your image path
    const imagePath = './test-image.jpg';
    
    if (!fs.existsSync(imagePath)) {
      console.log('⚠️ Please add a test image file as "test-image.jpg"');
      return;
    }

    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    const detectionResponse = await axios.post(`${API_URL}/detect`, formData, {
      headers: formData.getHeaders(),
      timeout: 30000 // 30 seconds
    });

    const result = detectionResponse.data;
    
    // 3. Display results
    console.log('\n📊 Detection Results:');
    console.log(`Found ${result.detections.length} hazards`);
    console.log(`Processing time: ${result.processing_time_ms}ms`);
    console.log(`Model: ${result.model_info.backend}`);

    // Show each detected hazard
    result.detections.forEach((detection, index) => {
      console.log(`\n🚨 Hazard ${index + 1}:`);
      console.log(`  Type: ${detection.class_name}`);
      console.log(`  Confidence: ${(detection.confidence * 100).toFixed(1)}%`);
      console.log(`  Location: x=${Math.round(detection.center_x)}, y=${Math.round(detection.center_y)}`);
      console.log(`  Size: ${Math.round(detection.width)} x ${Math.round(detection.height)} pixels`);
    });

    if (result.detections.length === 0) {
      console.log('✅ No hazards detected in this image');
    }

  } catch (error) {
    console.error('❌ Error:', error.response?.data?.detail || error.message);
    
    if (error.code === 'ECONNREFUSED') {
      console.log('💡 Make sure the API service is running at:', API_URL);
    }
  }
}

// Run the example
quickDetection();