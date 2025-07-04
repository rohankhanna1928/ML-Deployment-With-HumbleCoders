package com.humblecoders.mlmodel

import android.Manifest
import android.content.Context
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
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

    Column(modifier = Modifier.fillMaxSize()) {
        // Show prediction result
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = prediction,
                fontSize = 20.sp,
                modifier = Modifier.padding(16.dp)
            )
        }

        // Camera preview
        AndroidView(
            factory = { PreviewView(it) },
            modifier = Modifier.fillMaxSize()
        ) { previewView ->
            startCamera(previewView, lifecycleOwner) { bitmap ->
                prediction = classifier.classify(bitmap)
            }
        }
    }
}

fun startCamera(
    previewView: PreviewView,
    lifecycleOwner: LifecycleOwner,
    onResult: (Bitmap) -> Unit
) {
    val cameraProvider = ProcessCameraProvider.getInstance(previewView.context)
    val executor = Executors.newSingleThreadExecutor()

    cameraProvider.addListener({
        val provider = cameraProvider.get()

        // Camera preview
        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        // Image analysis
        val imageAnalysis = ImageAnalysis.Builder().build()
        imageAnalysis.setAnalyzer(executor) { imageProxy ->
            val bitmap = imageProxy.toBitmap()
            onResult(bitmap)
            imageProxy.close()
        }

        // Start camera
        provider.unbindAll()
        provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis
        )

    }, ContextCompat.getMainExecutor(previewView.context))
}

class ImageClassifier(context: Context) {
    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    init {
        loadModel(context)
        loadLabels(context)
    }

    private fun loadModel(context: Context) {
        val assetFileDescriptor = context.assets.openFd("mobilenet_v1_1.0_224_quant.tflite")
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val modelBuffer: MappedByteBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
        interpreter = Interpreter(modelBuffer)
    }

    private fun loadLabels(context: Context) {
        context.assets.open("labels.txt").bufferedReader().useLines { lines ->
            labels.addAll(lines.toList())
        }

    }

    fun classify(bitmap: Bitmap): String {
        if (interpreter == null) return "Model not loaded"

        // Resize image to 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Prepare input data
        val input = Array(1) { Array(224) { Array(224) { ByteArray(3) } } }

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resizedBitmap.getPixel(x, y)
                input[0][y][x][0] = Color.red(pixel).toByte()
                input[0][y][x][1] = Color.green(pixel).toByte()
                input[0][y][x][2] = Color.blue(pixel).toByte()
            }
        }

        // Run prediction
        val output = Array(1) { ByteArray(labels.size) }
        interpreter!!.run(input, output)

        // Find best result
        val predictions = output[0]
        var bestIndex = 0
        var bestScore = predictions[0].toInt() and 0xFF

        for (i in predictions.indices) {

            val score = predictions[i].toInt() and 0xFF
            if (score > bestScore) {
                bestScore = score
                bestIndex = i
            }
        }

        val confidence = (bestScore / 255.0 * 100).toInt()
        return if (confidence > 20) "${labels[bestIndex]} ($confidence%)" else "Uncertain"
    }
}