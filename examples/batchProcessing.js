// batchProcessing.js - Efficient batch processing of multiple images
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const API_URL = 'https://hazard-api-production-production.up.railway.app';

class BatchProcessor {
  constructor(options = {}) {
    this.batchSize = options.batchSize || 5; // Process 5 images at once
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000; // 1 second
    this.results = [];
    this.errors = [];
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async processWithRetry(operation, retries = this.maxRetries) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        if (attempt === retries) throw error;
        
        console.log(`⚠️ Attempt ${attempt} failed, retrying in ${this.retryDelay}ms...`);
        await this.sleep(this.retryDelay * attempt);
      }
    }
  }

  async batchDetect(imagePaths) {
    try {
      const formData = new FormData();
      
      // Add all images to form data
      imagePaths.forEach((imagePath) => {
        if (fs.existsSync(imagePath)) {
          formData.append('files', fs.createReadStream(imagePath));
        } else {
          console.log(`⚠️ Skipping missing file: ${imagePath}`);
        }
      });

      const response = await this.processWithRetry(async () => {
        return await axios.post(`${API_URL}/detect-batch`, formData, {
          headers: formData.getHeaders(),
          timeout: 60000 // 60 seconds for batch processing
        });
      });

      return response.data;
    } catch (error) {
      throw new Error(`Batch detection failed: ${error.message}`);
    }
  }

  async processSingleImage(imagePath) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imagePath));

      const response = await this.processWithRetry(async () => {
        return await axios.post(`${API_URL}/detect`, formData, {
          headers: formData.getHeaders(),
          timeout: 30000
        });
      });

      return {
        success: true,
        filename: path.basename(imagePath),
        detections: response.data.detections,
        processing_time: response.data.processing_time_ms
      };
    } catch (error) {
      return {
        success: false,
        filename: path.basename(imagePath),
        error: error.message
      };
    }
  }

  async processDirectory(directoryPath) {
    if (!fs.existsSync(directoryPath)) {
      throw new Error(`Directory not found: ${directoryPath}`);
    }

    // Get all image files
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp'];
    const files = fs.readdirSync(directoryPath)
      .filter(file => imageExtensions.includes(path.extname(file).toLowerCase()))
      .map(file => path.join(directoryPath, file));

    if (files.length === 0) {
      console.log('⚠️ No image files found in directory');
      return { results: [], summary: { total: 0, successful: 0, failed: 0 } };
    }

    console.log(`📁 Found ${files.length} images in ${directoryPath}`);
    
    return await this.processImageList(files);
  }

  async processImageList(imagePaths) {
    const startTime = Date.now();
    const allResults = [];
    let totalDetections = 0;

    console.log(`🚀 Processing ${imagePaths.length} images in batches of ${this.batchSize}...`);

    // Process in batches
    for (let i = 0; i < imagePaths.length; i += this.batchSize) {
      const batch = imagePaths.slice(i, i + this.batchSize);
      const batchNumber = Math.floor(i / this.batchSize) + 1;
      const totalBatches = Math.ceil(imagePaths.length / this.batchSize);

      console.log(`\n📦 Processing batch ${batchNumber}/${totalBatches} (${batch.length} images)`);

      try {
        // Use batch API if available and batch size > 1
        if (batch.length > 1) {
          const batchResult = await this.batchDetect(batch);
          allResults.push(...batchResult.results);
          
          // Log batch results
          batchResult.results.forEach(result => {
            if (result.success) {
              console.log(`  ✅ ${result.filename}: ${result.detections.length} detections`);
              totalDetections += result.detections.length;
            } else {
              console.log(`  ❌ ${result.filename}: ${result.error}`);
            }
          });
        } else {
          // Process single image
          const result = await this.processSingleImage(batch[0]);
          allResults.push(result);
          
          if (result.success) {
            console.log(`  ✅ ${result.filename}: ${result.detections.length} detections`);
            totalDetections += result.detections.length;
          } else {
            console.log(`  ❌ ${result.filename}: ${result.error}`);
          }
        }

        // Small delay between batches to be respectful
        if (i + this.batchSize < imagePaths.length) {
          await this.sleep(500); // 500ms between batches
        }

      } catch (error) {
        console.error(`❌ Batch ${batchNumber} failed:`, error.message);
        
        // Add error results for all images in failed batch
        batch.forEach(imagePath => {
          allResults.push({
            success: false,
            filename: path.basename(imagePath),
            error: `Batch processing failed: ${error.message}`
          });
        });
      }
    }

    const totalTime = Date.now() - startTime;
    const successful = allResults.filter(r => r.success).length;
    const failed = allResults.filter(r => !r.success).length;

    // Generate summary
    const summary = {
      total: allResults.length,
      successful,
      failed,
      totalDetections,
      processingTimeMs: totalTime,
      averageTimePerImage: Math.round(totalTime / allResults.length),
      successRate: Math.round((successful / allResults.length) * 100)
    };

    return { results: allResults, summary };
  }

  generateReport(results, summary) {
    console.log('\n📊 BATCH PROCESSING REPORT');
    console.log('==========================');
    console.log(`Total images: ${summary.total}`);
    console.log(`Successful: ${summary.successful} (${summary.successRate}%)`);
    console.log(`Failed: ${summary.failed}`);
    console.log(`Total hazards detected: ${summary.totalDetections}`);
    console.log(`Total processing time: ${(summary.processingTimeMs / 1000).toFixed(1)}s`);
    console.log(`Average time per image: ${summary.averageTimePerImage}ms`);

    // Hazard type breakdown
    const hazardTypes = {};
    results.forEach(result => {
      if (result.success && result.detections) {
        result.detections.forEach(detection => {
          hazardTypes[detection.class_name] = (hazardTypes[detection.class_name] || 0) + 1;
        });
      }
    });

    if (Object.keys(hazardTypes).length > 0) {
      console.log('\n🚨 Hazard Types Detected:');
      Object.entries(hazardTypes)
        .sort(([,a], [,b]) => b - a)
        .forEach(([type, count]) => {
          console.log(`  ${type}: ${count}`);
        });
    }

    // Show failed images if any
    const failedResults = results.filter(r => !r.success);
    if (failedResults.length > 0) {
      console.log('\n❌ Failed Images:');
      failedResults.forEach(result => {
        console.log(`  ${result.filename}: ${result.error}`);
      });
    }
  }

  async saveResults(results, outputPath) {
    const report = {
      timestamp: new Date().toISOString(),
      results,
      summary: {
        total: results.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      }
    };

    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`📄 Results saved to: ${outputPath}`);
  }
}

async function runBatchExample() {
  console.log('🚀 Batch Processing Example');
  console.log('============================\n');

  const processor = new BatchProcessor({
    batchSize: 3, // Process 3 images at a time
    maxRetries: 2
  });

  try {
    // Option 1: Process a directory
    const directoryPath = './test-images';
    
    if (fs.existsSync(directoryPath)) {
      console.log(`📁 Processing directory: ${directoryPath}`);
      const { results, summary } = await processor.processDirectory(directoryPath);
      
      processor.generateReport(results, summary);
      
      // Save results to file
      await processor.saveResults(results, './batch-results.json');
      
    } else {
      // Option 2: Process specific files
      const testImages = [
        './image1.jpg',
        './image2.jpg', 
        './image3.jpg'
      ];
      
      // Filter existing images
      const existingImages = testImages.filter(fs.existsSync);
      
      if (existingImages.length === 0) {
        console.log('⚠️ No test images found.');
        console.log('Please create a "./test-images" directory with images, or');
        console.log('add individual image files:', testImages.join(', '));
        return;
      }

      console.log(`📋 Processing ${existingImages.length} specific images`);
      const { results, summary } = await processor.processImageList(existingImages);
      
      processor.generateReport(results, summary);
    }

  } catch (error) {
    console.error('❌ Batch processing error:', error.message);
  }
}

// Advanced usage example
async function advancedBatchExample() {
  console.log('\n🔬 Advanced Batch Processing');
  console.log('=============================\n');

  const processor = new BatchProcessor({ batchSize: 5 });

  try {
    // Check service health first
    const healthResponse = await axios.get(`${API_URL}/health`);
    console.log('✅ Service healthy:', healthResponse.data.status);
    console.log('📊 Backend:', healthResponse.data.backend_type);

    // Process with performance monitoring
    const startTime = Date.now();
    
    // You can implement custom logic here
    const imagePaths = ['./sample1.jpg', './sample2.jpg']; // Add your images
    const existingImages = imagePaths.filter(fs.existsSync);
    
    if (existingImages.length > 0) {
      const { results, summary } = await processor.processImageList(existingImages);
      
      // Custom analysis
      const highConfidenceDetections = results
        .filter(r => r.success)
        .flatMap(r => r.detections)
        .filter(d => d.confidence > 0.8);

      console.log(`\n🎯 High confidence detections: ${highConfidenceDetections.length}`);
      
      processor.generateReport(results, summary);
    } else {
      console.log('⚠️ No images found for advanced processing');
    }

  } catch (error) {
    console.error('❌ Advanced processing error:', error.message);
  }
}

// Export for use in other modules
module.exports = BatchProcessor;

// Run examples if this file is executed directly
if (require.main === module) {
  runBatchExample().then(() => {
    console.log('\n✅ Batch processing completed');
  });
}