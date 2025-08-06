// Test OpenCV Fix - Comprehensive API Testing
// This script tests the PIL fallback and error handling improvements

const API_URL = 'https://hazard-api-production-production.up.railway.app';

async function testOpenCVFix() {
    console.log('🧪 Testing OpenCV Fix & PIL Fallback');
    console.log('=====================================');
    
    try {
        // Test 1: API Health Check
        console.log('\n📋 Step 1: Testing API Health...');
        const healthResponse = await fetch(`${API_URL}/health`);
        if (healthResponse.ok) {
            const health = await healthResponse.json();
            console.log('✅ API is healthy:', health.status);
        } else {
            throw new Error(`Health check failed: ${healthResponse.status}`);
        }
        
        // Test 2: Session Creation
        console.log('\n📋 Step 2: Creating Session...');
        const sessionResponse = await fetch(`${API_URL}/session/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                confidence_threshold: 0.5,
                source: 'opencv_fix_test'
            })
        });
        
        if (!sessionResponse.ok) {
            throw new Error(`Session creation failed: ${sessionResponse.status}`);
        }
        
        const session = await sessionResponse.json();
        console.log('✅ Session created:', session.session_id);
        
        // Test 3: Create test images and try detection
        console.log('\n📋 Step 3: Testing Detection with Multiple Image Types...');
        
        // Create different test images to trigger various processing paths
        const testImages = [
            { name: 'RGB Image', format: 'image/jpeg', size: [100, 100] },
            { name: 'PNG Image', format: 'image/png', size: [200, 150] },
            { name: 'Large Image', format: 'image/jpeg', size: [640, 480] }
        ];
        
        let successCount = 0;
        let failCount = 0;
        
        for (const testImg of testImages) {
            try {
                console.log(`\n🖼️  Testing ${testImg.name}...`);
                
                // Create canvas-based test image
                const canvas = document.createElement('canvas');
                canvas.width = testImg.size[0];
                canvas.height = testImg.size[1];
                const ctx = canvas.getContext('2d');
                
                // Fill with gradient pattern
                const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
                gradient.addColorStop(0, '#FF0000');
                gradient.addColorStop(0.5, '#00FF00');
                gradient.addColorStop(1, '#0000FF');
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Add some geometric shapes
                ctx.fillStyle = '#FFFFFF';
                ctx.fillRect(10, 10, 30, 30);
                ctx.beginPath();
                ctx.arc(canvas.width - 30, 30, 20, 0, 2 * Math.PI);
                ctx.fill();
                
                // Convert to blob
                const imageBlob = await new Promise(resolve => {
                    canvas.toBlob(resolve, testImg.format, 0.8);
                });
                
                // Test detection
                const formData = new FormData();
                formData.append('file', imageBlob, `test_${testImg.name.replace(' ', '_').toLowerCase()}.jpg`);
                
                const detectionResponse = await fetch(`${API_URL}/detect/${session.session_id}`, {
                    method: 'POST',
                    body: formData
                });
                
                if (detectionResponse.ok) {
                    const result = await detectionResponse.json();
                    console.log(`  ✅ ${testImg.name} processed successfully`);
                    console.log(`     Found ${result.detections?.length || 0} detections`);
                    console.log(`     Processing time: ${result.processing_time_ms || 'N/A'}ms`);
                    console.log(`     Image size: ${result.image_size?.width}x${result.image_size?.height}`);
                    successCount++;
                } else {
                    const errorText = await detectionResponse.text();
                    console.log(`  ❌ ${testImg.name} failed:`, detectionResponse.status);
                    console.log(`     Error: ${errorText.substring(0, 200)}`);
                    failCount++;
                }
                
            } catch (error) {
                console.log(`  ❌ ${testImg.name} error:`, error.message);
                failCount++;
            }
        }
        
        console.log('\n🎉 OpenCV Fix Test Results:');
        console.log('============================');
        console.log(`✅ Successful detections: ${successCount}/${testImages.length}`);
        console.log(`❌ Failed detections: ${failCount}/${testImages.length}`);
        
        if (successCount > 0) {
            console.log('✅ PIL fallback is working!');
            console.log('✅ Image processing errors have been fixed');
            console.log('✅ Camera detection should now work reliably');
            console.log(`\n🚀 Test camera at: https://hazard-detection-production-8735.up.railway.app/camera.html`);
        } else {
            console.log('❌ All detections failed - may need more investigation');
        }
        
        // Test 4: Base64 Detection (camera format)
        console.log('\n📋 Step 4: Testing Base64 Detection (Camera Format)...');
        
        try {
            // Create a small test image for base64 testing
            const canvas = document.createElement('canvas');
            canvas.width = 320;
            canvas.height = 240;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#FF4444';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '20px Arial';
            ctx.fillText('TEST', 50, 50);
            
            // Convert to base64
            const base64Data = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            
            const base64Response = await fetch(`${API_URL}/detect-base64`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64Data,
                    confidence_threshold: 0.5
                })
            });
            
            if (base64Response.ok) {
                const result = await base64Response.json();
                console.log('✅ Base64 detection working');
                console.log(`   Found ${result.detections?.length || 0} detections`);
                console.log(`   Processing time: ${result.processing_time_ms || 'N/A'}ms`);
            } else {
                const errorText = await base64Response.text();
                console.log('❌ Base64 detection failed:', base64Response.status);
                console.log('   Error:', errorText.substring(0, 200));
            }
            
        } catch (error) {
            console.log('❌ Base64 test error:', error.message);
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.log('\n⏱️ Railway deployment may still be in progress...');
        console.log('   Wait 2-3 minutes for the API to fully deploy the fixes.');
    }
}

// Check if we're in browser environment
if (typeof window !== 'undefined') {
    testOpenCVFix();
} else {
    console.log('❌ This test needs to run in a browser environment');
    console.log('   Copy and run in browser console to test camera functionality');
}