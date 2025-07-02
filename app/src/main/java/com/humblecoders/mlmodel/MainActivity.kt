package com.humblecoders.mlmodel

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            setContent { ImageClassifierApp() }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            setContent { ImageClassifierApp() }
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }
}

@Composable
fun ImageClassifierApp() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var prediction by remember { mutableStateOf("Point camera at objects") }

    val classifier = remember { ImageClassifier(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Prediction Display
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Text(
                text = prediction,
                fontSize = 20.sp,
                modifier = Modifier.padding(16.dp)
            )
        }

        // Camera Preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    setupCamera(this, lifecycleOwner, cameraExecutor) { bitmap ->
                        prediction = classifier.classify(bitmap)
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )
    }
}

fun setupCamera(
    previewView: PreviewView,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    cameraExecutor: ExecutorService,
    onImageAnalyzed: (Bitmap) -> Unit
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(previewView.context)

    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    try {
                        // Only process every 30th frame
                        if (imageProxy.imageInfo.timestamp % 30 == 0L) {
                            val bitmap = imageProxy.toBitmap()
                            onImageAnalyzed(bitmap)
                        }
                    } catch (e: Exception) {
                        // Ignore errors
                    } finally {
                        imageProxy.close()
                    }
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (e: Exception) {
            // Handle camera binding error
        }
    }, ContextCompat.getMainExecutor(previewView.context))
}

class ImageClassifier(private val context: android.content.Context) {
    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    init {
        loadModel()
        loadLabels()
    }

    private fun loadModel() {
        try {
            val assetFileDescriptor = context.assets.openFd("mobilenet_v1_1.0_224_quant.tflite")
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer: MappedByteBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY, startOffset, declaredLength
            )
            interpreter = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadLabels() {
        try {
            context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                labels.addAll(lines.toList())
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): String {
        return try {
            if (interpreter == null || labels.isEmpty()) {
                return "Model not loaded"
            }

            // Resize to 224x224
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

            // Prepare input: [1, 224, 224, 3] as UByte
            val input = Array(1) { Array(224) { Array(224) { ByteArray(3) } } }

            // Fill input with pixel values [0-255]
            for (y in 0 until 224) {
                for (x in 0 until 224) {
                    val pixel = resizedBitmap.getPixel(x, y)
                    input[0][y][x][0] = Color.red(pixel).toByte()
                    input[0][y][x][1] = Color.green(pixel).toByte()
                    input[0][y][x][2] = Color.blue(pixel).toByte()
                }
            }

            // Output array for quantized model
            val output = Array(1) { ByteArray(labels.size) }

            // Run inference
            interpreter!!.run(input, output)

            // Find best prediction
            val predictions = output[0]
            var maxIndex = 0
            var maxValue = predictions[0].toInt() and 0xFF

            for (i in predictions.indices) {
                val value = predictions[i].toInt() and 0xFF
                if (value > maxValue) {
                    maxValue = value
                    maxIndex = i
                }
            }

            // Return result with confidence
            val confidence = (maxValue / 255.0 * 100).toInt()
            if (confidence > 20) {
                "${labels[maxIndex]} ($confidence%)"
            } else {
                "Uncertain"
            }
        } catch (e: Exception) {
            "Error: ${e.message}"
        }
    }
}