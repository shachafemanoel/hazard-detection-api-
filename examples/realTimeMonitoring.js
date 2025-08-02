// realTimeMonitoring.js - Real-time hazard monitoring system
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const chokidar = require('chokidar'); // npm install chokidar

const API_URL = 'https://hazard-api-production-production.up.railway.app';

class RealTimeHazardMonitor {
  constructor(options = {}) {
    this.watchDirectory = options.watchDirectory || './incoming-images';
    this.outputDirectory = options.outputDirectory || './processed-results';
    this.alertThreshold = options.alertThreshold || 0.8; // Alert on 80%+ confidence
    this.sessionDuration = options.sessionDuration || 10 * 60 * 1000; // 10 minutes
    this.onHazardDetected = options.onHazardDetected || this.defaultHazardHandler;
    
    this.currentSession = null;
    this.sessionTimeout = null;
    this.stats = {
      imagesProcessed: 0,
      hazardsDetected: 0,
      alerts: 0,
      startTime: Date.now()
    };

    this.setupDirectories();
  }

  setupDirectories() {
    [this.watchDirectory, this.outputDirectory].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`📁 Created directory: ${dir}`);
      }
    });
  }

  async ensureSession() {
    if (!this.currentSession) {
      try {
        const response = await axios.post(`${API_URL}/session/start`);
        this.currentSession = response.data.session_id;
        console.log(`🔄 New session started: ${this.currentSession}`);
        
        // Auto-renew session
        this.sessionTimeout = setTimeout(() => {
          this.renewSession();
        }, this.sessionDuration);
        
      } catch (error) {
        console.error('❌ Failed to start session:', error.message);
        throw error;
      }
    }
    return this.currentSession;
  }

  async renewSession() {
    console.log('🔄 Renewing session...');
    
    if (this.currentSession) {
      try {
        await axios.post(`${API_URL}/session/${this.currentSession}/end`);
      } catch (error) {
        console.error('⚠️ Error ending old session:', error.message);
      }
    }
    
    this.currentSession = null;
    if (this.sessionTimeout) {
      clearTimeout(this.sessionTimeout);
    }
    
    await this.ensureSession();
  }

  async processImage(imagePath) {
    try {
      const sessionId = await this.ensureSession();
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imagePath));

      const response = await axios.post(
        `${API_URL}/detect/${sessionId}`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 30000
        }
      );

      const result = response.data;
      this.stats.imagesProcessed++;

      // Log processing result
      const filename = path.basename(imagePath);
      console.log(`📸 Processed: ${filename} - ${result.detections.length} detections in ${result.processing_time_ms}ms`);

      // Handle detections
      for (const detection of result.detections) {
        this.stats.hazardsDetected++;
        
        if (detection.confidence >= this.alertThreshold) {
          this.stats.alerts++;
          await this.onHazardDetected(detection, filename, imagePath);
        }
      }

      // Save results
      await this.saveProcessingResult(imagePath, result);

      return result;

    } catch (error) {
      console.error(`❌ Error processing ${imagePath}:`, error.message);
      return { error: error.message };
    }
  }

  async defaultHazardHandler(detection, filename, imagePath) {
    const alert = {
      timestamp: new Date().toISOString(),
      filename,
      hazardType: detection.class_name,
      confidence: (detection.confidence * 100).toFixed(1),
      location: {
        x: Math.round(detection.center_x),
        y: Math.round(detection.center_y),
        width: Math.round(detection.width),
        height: Math.round(detection.height)
      },
      bbox: detection.bbox
    };

    console.log(`🚨 HIGH CONFIDENCE ALERT:`, alert);

    // Save alert to file
    const alertFile = path.join(this.outputDirectory, 'alerts.jsonl');
    fs.appendFileSync(alertFile, JSON.stringify(alert) + '\n');

    // You could also:
    // - Send webhook notification
    // - Send email alert
    // - Push to external monitoring system
    // - Trigger automated response
    
    return alert;
  }

  async saveProcessingResult(imagePath, result) {
    const filename = path.basename(imagePath, path.extname(imagePath));
    const resultFile = path.join(this.outputDirectory, `${filename}_result.json`);
    
    const processedResult = {
      timestamp: new Date().toISOString(),
      originalImage: imagePath,
      detections: result.detections,
      processingTime: result.processing_time_ms,
      sessionStats: result.session_stats
    };

    fs.writeFileSync(resultFile, JSON.stringify(processedResult, null, 2));
  }

  startMonitoring() {
    console.log(`👁️ Starting real-time monitoring of: ${this.watchDirectory}`);
    console.log(`📊 Alert threshold: ${(this.alertThreshold * 100)}%`);
    console.log(`📁 Results will be saved to: ${this.outputDirectory}`);

    const watcher = chokidar.watch(this.watchDirectory, {
      ignored: /(^|[\/\\])\../, // ignore dotfiles
      persistent: true,
      ignoreInitial: false // Process existing files
    });

    watcher
      .on('add', (imagePath) => {
        const ext = path.extname(imagePath).toLowerCase();
        if (['.jpg', '.jpeg', '.png', '.bmp'].includes(ext)) {
          console.log(`\n📥 New image detected: ${path.basename(imagePath)}`);
          this.processImage(imagePath);
        }
      })
      .on('error', (error) => {
        console.error('❌ Watcher error:', error);
      });

    // Print stats every 30 seconds
    setInterval(() => {
      this.printStats();
    }, 30000);

    console.log('\n✅ Real-time monitoring started. Add images to the watch directory to process them.');
    console.log('Press Ctrl+C to stop monitoring.\n');

    // Graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🛑 Stopping monitoring...');
      watcher.close();
      
      if (this.currentSession) {
        try {
          await axios.post(`${API_URL}/session/${this.currentSession}/end`);
          console.log('✅ Session ended successfully');
        } catch (error) {
          console.error('⚠️ Error ending session:', error.message);
        }
      }
      
      this.printFinalStats();
      process.exit(0);
    });

    return watcher;
  }

  printStats() {
    const uptime = Math.round((Date.now() - this.stats.startTime) / 1000);
    const rate = this.stats.imagesProcessed > 0 ? 
      Math.round(uptime / this.stats.imagesProcessed) : 0;

    console.log('\n📊 MONITORING STATS');
    console.log('==================');
    console.log(`Uptime: ${uptime}s`);
    console.log(`Images processed: ${this.stats.imagesProcessed}`);
    console.log(`Hazards detected: ${this.stats.hazardsDetected}`);
    console.log(`High-confidence alerts: ${this.stats.alerts}`);
    console.log(`Average processing rate: ${rate}s/image`);
    console.log(`Current session: ${this.currentSession ? 'Active' : 'None'}`);
  }

  printFinalStats() {
    console.log('\n📋 FINAL MONITORING REPORT');
    console.log('==========================');
    const totalTime = Math.round((Date.now() - this.stats.startTime) / 1000);
    console.log(`Total monitoring time: ${totalTime}s`);
    console.log(`Images processed: ${this.stats.imagesProcessed}`);
    console.log(`Total hazards detected: ${this.stats.hazardsDetected}`);
    console.log(`High-confidence alerts: ${this.stats.alerts}`);
    
    if (this.stats.imagesProcessed > 0) {
      console.log(`Average processing rate: ${Math.round(totalTime / this.stats.imagesProcessed)}s/image`);
      console.log(`Hazard detection rate: ${Math.round((this.stats.hazardsDetected / this.stats.imagesProcessed) * 100)}%`);
    }
  }

  // Manual processing of existing images
  async processExistingImages() {
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp'];
    const files = fs.readdirSync(this.watchDirectory)
      .filter(file => imageExtensions.includes(path.extname(file).toLowerCase()))
      .map(file => path.join(this.watchDirectory, file));

    if (files.length === 0) {
      console.log('📁 No existing images found in watch directory');
      return;
    }

    console.log(`📋 Processing ${files.length} existing images...`);

    for (const imagePath of files) {
      await this.processImage(imagePath);
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second delay
    }

    console.log('✅ Finished processing existing images');
  }
}

// Custom alert handler example
function customAlertHandler(detection, filename, imagePath) {
  // Example: Send webhook notification
  const webhookUrl = 'https://your-webhook-url.com/alerts';
  
  const payload = {
    alert_type: 'hazard_detected',
    timestamp: new Date().toISOString(),
    image: filename,
    hazard: {
      type: detection.class_name,
      confidence: detection.confidence,
      location: detection.bbox
    }
  };

  // Send webhook (replace with your actual webhook logic)
  console.log('🔔 Would send webhook:', payload);
  
  // You could use axios.post(webhookUrl, payload) here
}

// Usage example
async function startRealTimeMonitoring() {
  console.log('🚀 Real-Time Hazard Monitoring System');
  console.log('=====================================\n');

  // Check API health first
  try {
    const health = await axios.get(`${API_URL}/health`);
    console.log('✅ API Health:', health.data.status);
    console.log('📊 Model Backend:', health.data.backend_type);
  } catch (error) {
    console.error('❌ API not available:', error.message);
    return;
  }

  const monitor = new RealTimeHazardMonitor({
    watchDirectory: './incoming-images',
    outputDirectory: './monitoring-results',
    alertThreshold: 0.7, // 70% confidence threshold
    onHazardDetected: customAlertHandler // Custom alert handler
  });

  // Process any existing images first
  await monitor.processExistingImages();

  // Start real-time monitoring
  monitor.startMonitoring();
}

// Simulation mode for testing
async function simulationMode() {
  console.log('🎭 Simulation Mode - Testing with sample images');
  
  const monitor = new RealTimeHazardMonitor({
    watchDirectory: './test-simulation',
    outputDirectory: './simulation-results'
  });

  // Create test directory
  if (!fs.existsSync('./test-simulation')) {
    fs.mkdirSync('./test-simulation');
    console.log('📁 Created ./test-simulation directory');
    console.log('💡 Add some test images to this directory to see the monitoring in action');
  }

  monitor.startMonitoring();
}

module.exports = RealTimeHazardMonitor;

// Run if executed directly
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.includes('--simulate')) {
    simulationMode();
  } else {
    startRealTimeMonitoring();
  }
}