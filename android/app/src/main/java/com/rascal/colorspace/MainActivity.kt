package com.rascal.colorspace

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import androidx.core.graphics.toColorInt
import androidx.core.net.toUri
import android.location.Location
import android.location.LocationManager
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: GLRenderer
    private lateinit var seekBarRotX: SeekBar
    private lateinit var seekBarRotY: SeekBar
    private lateinit var seekBarRotZ: SeekBar
    private lateinit var seekBarPushFactor: SeekBar
    private lateinit var seekBarVarRestore: SeekBar
    private lateinit var labelPushFactor: android.widget.TextView
    private lateinit var labelVarRestore: android.widget.TextView
    private lateinit var labelRotX: android.widget.TextView
    private lateinit var labelRotY: android.widget.TextView
    private lateinit var labelRotZ: android.widget.TextView
    private lateinit var btnLoadImage: Button
    private lateinit var btnReset: Button
    private lateinit var btnSplit: Button
    private lateinit var btnSave: Button
    private lateinit var btnChannelC: Button
    private lateinit var btnChannelR: Button
    private lateinit var btnChannelG: Button
    private lateinit var btnChannelB: Button
    private lateinit var btnROI: Button
    private lateinit var btnICA: Button
    private lateinit var btnInfo: Button
    private lateinit var btnRandom: Button
    private var isRandomRotating = false
    private var randRotHandler: android.os.Handler? = null
    private var randRotRunnable: Runnable? = null
    private lateinit var btnPreset: Button
    private lateinit var btnPresetClear: Button
    private val presetNumberButtons = mutableListOf<Button>()
    private lateinit var splashImage: ImageView
    private lateinit var topOverlay: LinearLayout
    private lateinit var bottomOverlay: LinearLayout
    private lateinit var roiHintText: android.widget.TextView
    private lateinit var roiOverlay: View
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    private lateinit var gestureDetector: GestureDetector
    
    private enum class ChannelMode {
        COLOR, RED, GREEN, BLUE
    }
    
    private var currentChannelMode = ChannelMode.COLOR
    private var roiMode = false

    private class Preset(
        val rotX: Int,
        val rotY: Int,
        val rotZ: Int,
        val pushFactor: Int,
        val varRestore: Int,
        val channelMode: ChannelMode,
        val roiActive: Boolean,
        val roiBitmap: Bitmap?,
        val roiMu: FloatArray?,
        val roiW: FloatArray?,
        val roiWInv: FloatArray?,
        val rotationMatrix: FloatArray? = null
    )

    private val presets = mutableListOf<Preset>()
    private val maxPresets = 5

    private var roiActive = false  // Indique si la ZCA est calculée avec ROI
    @Volatile private var roiComputationRunning = false  // Empêche deux calculs ROI en parallèle
    @Volatile private var roiImageMask: BooleanArray? = null  // Masque ROI aux dimensions du bitmap image (sauvegardé après ZCA ROI)
    private var roiPaintOverlay: Bitmap? = null
    private var roiCanvas: android.graphics.Canvas? = null
    private var roiPaint: android.graphics.Paint? = null
    
    private var lastTouchX: Float = 0f
    private var lastTouchY: Float = 0f
    private var splitDragActive: Boolean = false
    private var lastRotationAngle: Float = 0f
    private var lastMidX: Float = 0f
    private var lastMidY: Float = 0f
    private var activePointerId: Int = MotionEvent.INVALID_POINTER_ID
    private var isRotating: Boolean = false
    private var isPanning: Boolean = false
    private var floatAngleX: Float = 0f
    private var floatAngleY: Float = 0f
    private var floatAngleZ: Float = 0f
    
    companion object {
        private const val TAG = "MainActivity"

        fun srgbToLin(c: Float): Float =
            if (c <= 0.04045f) c / 12.92f
            else ((c + 0.055f) / 1.055f).pow(2.4f)

        fun linToSrgb(c: Float): Float {
            val cc = c.coerceIn(0f, 1f)
            return if (cc <= 0.0031308f) 12.92f * cc
            else (1.055f * cc.pow(1f / 2.4f) - 0.055f).coerceIn(0f, 1f)
        }
    }

    @Volatile private var currentImageUri: Uri? = null
    private var pendingAction: PendingAction = PendingAction.NONE
    private var lastKnownLocation: Location? = null

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) fetchLocationAndLaunchCamera()
        else { lastKnownLocation = null; launchCamera() }
    }
    
    private enum class PendingAction {
        NONE, CAMERA, GALLERY
    }

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        Log.d(TAG, "Image picker result: resultCode=${result.resultCode}")
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                Log.d(TAG, "Image URI selected: $uri")
                loadImageFromUri(uri)
            } ?: Log.e(TAG, "No URI in result data")
        } else {
            Log.e(TAG, "Image picker cancelled or failed")
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        Log.d(TAG, "Camera result: resultCode=${result.resultCode}")
        if (result.resultCode == RESULT_OK) {
            val uriStr = result.data?.getStringExtra(CameraActivity.EXTRA_PHOTO_URI)
            if (uriStr != null) {
                val uri = uriStr.toUri()
                Log.d(TAG, "Photo captured: $uri")
                loadImageFromUri(uri)
            } else {
                Log.e(TAG, "No photo URI in result")
            }
        } else {
            Log.e(TAG, "Photo capture cancelled or failed")
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            when (pendingAction) {
                PendingAction.CAMERA -> askGpsThenTakePhoto()
                PendingAction.GALLERY -> openImagePicker()
                PendingAction.NONE -> {}
            }
            pendingAction = PendingAction.NONE
        } else {
            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            pendingAction = PendingAction.NONE
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        enableFullscreenMode()
        
        setContentView(R.layout.activity_main)

        glSurfaceView = findViewById(R.id.glSurfaceView)
        seekBarRotX = findViewById(R.id.seekBarRotX)
        seekBarRotY = findViewById(R.id.seekBarRotY)
        seekBarRotZ = findViewById(R.id.seekBarRotZ)
        seekBarPushFactor = findViewById(R.id.seekBarPushFactor)
        seekBarVarRestore = findViewById(R.id.seekBarVarRestore)
        labelPushFactor = findViewById(R.id.labelPushFactor)
        labelVarRestore = findViewById(R.id.labelVarRestore)
        labelRotX = findViewById(R.id.labelRotX)
        labelRotY = findViewById(R.id.labelRotY)
        labelRotZ = findViewById(R.id.labelRotZ)
        btnLoadImage = findViewById(R.id.btnLoadImage)
        btnReset = findViewById(R.id.btnReset)
        btnSplit = findViewById(R.id.btnSplit)
        btnSave = findViewById(R.id.btnSave)
        btnChannelC = findViewById(R.id.btnChannelC)
        btnChannelR = findViewById(R.id.btnChannelR)
        btnChannelG = findViewById(R.id.btnChannelG)
        btnChannelB = findViewById(R.id.btnChannelB)
        btnROI = findViewById(R.id.btnROI)
        btnICA = findViewById(R.id.btnICA)
        btnInfo = findViewById(R.id.btnInfo)
        btnRandom = findViewById(R.id.btnRandom)
        btnPreset = findViewById(R.id.btnPreset)
        btnPresetClear = findViewById(R.id.btnPresetClear)
        presetNumberButtons.clear()
        presetNumberButtons.addAll(listOf(
            findViewById(R.id.btnPreset1),
            findViewById(R.id.btnPreset2),
            findViewById(R.id.btnPreset3),
            findViewById(R.id.btnPreset4),
            findViewById(R.id.btnPreset5)
        ))
        splashImage = findViewById(R.id.splashImage)
        topOverlay = findViewById(R.id.topOverlay)
        bottomOverlay = findViewById(R.id.bottomOverlay)
        roiHintText = findViewById(R.id.roiHintText)
        roiOverlay = findViewById(R.id.roiOverlay)
        roiOverlay.setOnTouchListener { _, event -> handleTouchGestures(event) }

        // Configurer l'overlay ROI pour dessiner le bitmap
        roiOverlay.background = object : android.graphics.drawable.Drawable() {
            override fun draw(canvas: android.graphics.Canvas) {
                roiPaintOverlay?.let { bitmap ->
                    canvas.drawBitmap(bitmap, 0f, 0f, null)
                }
            }
            override fun setAlpha(alpha: Int) {}
            override fun setColorFilter(colorFilter: android.graphics.ColorFilter?) {}
            @Deprecated("Deprecated in Java")
            override fun getOpacity(): Int = android.graphics.PixelFormat.TRANSLUCENT
        }

        glSurfaceView.setEGLContextClientVersion(2)
        glSurfaceView.setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        glSurfaceView.holder.setFormat(android.graphics.PixelFormat.RGBA_8888)
        renderer = GLRenderer(this)
        glSurfaceView.setRenderer(renderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

        setupSeekBars()
        setupButtons()
        setupGestureDetector()
        showSplashState()
        
        // Restore saved state AFTER views are initialized
        val savedZoom = savedInstanceState?.getFloat("zoomScale", 1f) ?: 1f
        
        savedInstanceState?.let {
            currentImageUri = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                it.getParcelable("currentImageUri", Uri::class.java)
            } else {
                @Suppress("DEPRECATION") it.getParcelable("currentImageUri")
            }
            seekBarRotX.progress = it.getInt("rotX", 180)
            seekBarRotY.progress = it.getInt("rotY", 180)
            seekBarRotZ.progress = it.getInt("rotZ", 180)
            seekBarPushFactor.progress = it.getInt("pushFactor", 100)
            
            // Apply the restored rotation and push factor values
            updateRotation()
            updatePushFactor()
        }
        
        // Reload image if we have a saved URI (silently, no toast messages)
        currentImageUri?.let { uri ->
            loadImageFromUri(uri, showToast = false, restoreZoom = savedZoom)
            showImageControls() // Afficher les contrôles si une image est déjà chargée
        }
    }
    
    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        currentImageUri?.let { outState.putParcelable("currentImageUri", it) }
        outState.putInt("rotX", seekBarRotX.progress)
        outState.putInt("rotY", seekBarRotY.progress)
        outState.putInt("rotZ", seekBarRotZ.progress)
        outState.putInt("pushFactor", seekBarPushFactor.progress)
        outState.putFloat("zoomScale", renderer.getZoomScale())
    }
    
    private fun enableFullscreenMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        
        val windowInsetsController = WindowCompat.getInsetsController(window, window.decorView)
        windowInsetsController.apply {
            hide(WindowInsetsCompat.Type.statusBars())
            hide(WindowInsetsCompat.Type.navigationBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
        
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
    
    override fun onPause() {
        super.onPause()
        stopRandomRotation()
        glSurfaceView.onPause()
    }

    override fun onResume() {
        super.onResume()
        glSurfaceView.onResume()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enableFullscreenMode()
        }
    }

    private fun setupSeekBars() {
        val seekBarListener = object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    stopRandomRotation()
                    updateRotation()
                }
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        }

        seekBarRotX.setOnSeekBarChangeListener(seekBarListener)
        seekBarRotY.setOnSeekBarChangeListener(seekBarListener)
        seekBarRotZ.setOnSeekBarChangeListener(seekBarListener)
        
        seekBarPushFactor.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    updatePushFactor()
                }
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        seekBarVarRestore.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    updateVarRestore()
                }
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        updatePushFactorLabel()
        updateVarRestoreLabel()
    }

    private fun setupButtons() {
        btnLoadImage.setOnClickListener {
            checkPermissionAndLoadImage()
        }

        btnReset.setOnClickListener {
            resetParameters(resetZCA = true)
        }

        btnSplit.setOnClickListener {
            toggleSplitMode()
        }

        btnSave.setOnClickListener {
            saveProcessedImage()
        }
        
        btnChannelC.setOnClickListener {
            setChannelMode(ChannelMode.COLOR)
        }
        
        btnChannelR.setOnClickListener {
            setChannelMode(ChannelMode.RED)
        }
        
        btnChannelG.setOnClickListener {
            setChannelMode(ChannelMode.GREEN)
        }
        
        btnChannelB.setOnClickListener {
            setChannelMode(ChannelMode.BLUE)
        }
        
        btnROI.setOnClickListener {
            toggleROIMode()
        }

        btnICA.setOnClickListener {
            applyICA()
        }

        btnInfo.setOnClickListener {
            showInfoDialog()
        }

        btnRandom.setOnClickListener {
            toggleRandomRotation()
        }

        btnPreset.setOnClickListener {
            savePreset()
        }

        for (i in presetNumberButtons.indices) {
            presetNumberButtons[i].setOnClickListener {
                recallPreset(i)
            }
        }

        btnPresetClear.setOnClickListener {
            clearExtraPresets()
        }
    }

    private fun updateRotation() {
        val angleX = (seekBarRotX.progress - 180).toFloat()
        val angleY = (seekBarRotY.progress - 180).toFloat()
        val angleZ = (seekBarRotZ.progress - 180).toFloat()

        btnICA.setTextColor(android.graphics.Color.WHITE)
        glSurfaceView.queueEvent {
            renderer.setRotationAngles(angleX, angleY, angleZ)
        }
        glSurfaceView.requestRender()
        labelRotX.text = getString(R.string.angle_degrees, angleX.toInt())
        labelRotY.text = getString(R.string.angle_degrees, angleY.toInt())
        labelRotZ.text = getString(R.string.angle_degrees, angleZ.toInt())
    }
    
    private fun updatePushFactor() {
        val factor = seekBarPushFactor.progress / 100f
        glSurfaceView.queueEvent {
            renderer.setPushFactor(factor)
        }
        glSurfaceView.requestRender()
        updatePushFactorLabel()
    }
    
    private fun updatePushFactorLabel() {
        val factor = seekBarPushFactor.progress / 100f
        labelPushFactor.text = String.format(Locale.US, "%.2f", factor)
    }

    private fun updateVarRestore() {
        val value = seekBarVarRestore.progress / 100f
        glSurfaceView.queueEvent {
            renderer.setVarRestore(value)
        }
        glSurfaceView.requestRender()
        updateVarRestoreLabel()
    }

    private fun updateVarRestoreLabel() {
        val value = seekBarVarRestore.progress / 100f
        labelVarRestore.text = String.format(Locale.US, "%.2f", value)
    }
    
    private fun resetParameters(resetZCA: Boolean = false) {
        stopRandomRotation()
        disableSplit()
        seekBarRotX.progress = 180
        seekBarRotY.progress = 180
        seekBarRotZ.progress = 180
        seekBarPushFactor.progress = 100
        seekBarVarRestore.progress = 100
        glSurfaceView.queueEvent { renderer.clearICABase() }
        updateRotation()
        updatePushFactor()
        updateVarRestore()

        btnICA.setTextColor(android.graphics.Color.WHITE)
        if (resetZCA) {
            roiActive = false
            roiImageMask = null
            btnROI.setTextColor(android.graphics.Color.WHITE)
            clearROIPaint()
            glSurfaceView.queueEvent {
                renderer.restoreOriginalBitmap()
                renderer.resetZoom()
                glSurfaceView.requestRender()
            }
        } else {
            glSurfaceView.queueEvent {
                renderer.resetZoom()
                glSurfaceView.requestRender()
            }
        }
    }

    private fun toggleSplitMode() {
        if (!renderer.isSplitActive()) {
            renderer.setSplitActive(true)
            val centerPos = renderer.getCenterSplitPos()
            renderer.setSplitPos(centerPos)
            btnSplit.setTextColor(android.graphics.Color.RED)
            glSurfaceView.requestRender()
        } else {
            disableSplit()
        }
    }

    private fun disableSplit() {
        if (!renderer.isSplitActive()) return
        renderer.setSplitActive(false)
        renderer.setSplitPos(0.5f)
        btnSplit.setTextColor(android.graphics.Color.WHITE)
        splitDragActive = false
        glSurfaceView.requestRender()
    }

    private fun checkPermissionAndLoadImage() {
        showImageSourceDialog()
    }

    private fun showImageSourceDialog() {
        val options = arrayOf("Take Photo", "Choose from Gallery")
        AlertDialog.Builder(this)
            .setTitle("Select Image Source")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> checkCameraPermissionAndTakePhoto()
                    1 -> checkGalleryPermissionAndPick()
                }
            }
            .show()
    }

    private fun checkGalleryPermissionAndPick() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            openImagePicker()
        } else {
            pendingAction = PendingAction.GALLERY
            permissionLauncher.launch(permission)
        }
    }

    private fun checkCameraPermissionAndTakePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            askGpsThenTakePhoto()
        } else {
            pendingAction = PendingAction.CAMERA
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun askGpsThenTakePhoto() {
        AlertDialog.Builder(this)
            .setTitle("GPS coordinates")
            .setMessage("Save GPS coordinates in captured photos?")
            .setPositiveButton("Yes") { _, _ ->
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                    == PackageManager.PERMISSION_GRANTED) {
                    fetchLocationAndLaunchCamera()
                } else {
                    locationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
                }
            }
            .setNegativeButton("No") { _, _ -> lastKnownLocation = null; launchCamera() }
            .show()
    }

    private fun fetchLocationAndLaunchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED) {
            val lm = getSystemService(LOCATION_SERVICE) as LocationManager
            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)
            for (provider in providers) {
                val loc = lm.getLastKnownLocation(provider)
                if (loc != null) {
                    lastKnownLocation = loc
                    Log.d(TAG, "Location obtained from $provider: ${loc.latitude}, ${loc.longitude}")
                    break
                }
            }
        }
        launchCamera()
    }

    private fun launchCamera() {
        val intent = Intent(this, CameraActivity::class.java)
        lastKnownLocation?.let { loc ->
            intent.putExtra(CameraActivity.EXTRA_GPS_LATITUDE, loc.latitude)
            intent.putExtra(CameraActivity.EXTRA_GPS_LONGITUDE, loc.longitude)
            intent.putExtra(CameraActivity.EXTRA_GPS_HAS_ALTITUDE, loc.hasAltitude())
            if (loc.hasAltitude()) intent.putExtra(CameraActivity.EXTRA_GPS_ALTITUDE, loc.altitude)
        }
        cameraLauncher.launch(intent)
    }

    private fun openImagePicker() {
        val intent = Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private fun loadImageFromUri(uri: Uri, showToast: Boolean = true, restoreZoom: Float? = null) {
        if (showToast) {
            Toast.makeText(this, "Loading image...", Toast.LENGTH_SHORT).show()
        }

        // Reset UI immédiatement sur le UI thread
        currentImageUri = uri
        disableSplit()
        if (restoreZoom == null) {
            seekBarRotX.progress = 180
            seekBarRotY.progress = 180
            seekBarRotZ.progress = 180
            seekBarPushFactor.progress = 100
            seekBarVarRestore.progress = 100
            updateRotation()
            updatePushFactor()
            updateVarRestore()
            setChannelMode(ChannelMode.COLOR)
            clearAllPresets()
            roiMode = false
            roiHintText.visibility = View.GONE
            roiOverlay.visibility = View.GONE
        }
        roiActive = false
        roiImageMask = null
        btnROI.setTextColor(android.graphics.Color.WHITE)
        btnICA.setTextColor(android.graphics.Color.WHITE)
        clearROIPaint()
        hideSplashState()
        showImageControls()

        val isNewLoad = (restoreZoom == null)
        Thread {
            try {
                // Tout le décodage sur le thread background
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                var inputStream = openUriInputStream(uri)
                BitmapFactory.decodeStream(inputStream, null, options)
                inputStream?.close()

                val maxDimension = 4096
                var inSampleSize = 1
                while ((options.outWidth / inSampleSize) > maxDimension * 2 || (options.outHeight / inSampleSize) > maxDimension * 2) {
                    inSampleSize *= 2
                }

                val loadOptions = BitmapFactory.Options().apply { this.inSampleSize = inSampleSize }
                inputStream = openUriInputStream(uri)
                var bitmap = BitmapFactory.decodeStream(inputStream, null, loadOptions)
                inputStream?.close()

                val origW = options.outWidth; val origH = options.outHeight
                if (bitmap != null) {
                    val bw = bitmap.width; val bh = bitmap.height
                    if (bw > maxDimension || bh > maxDimension) {
                        val scale = maxDimension.toFloat() / maxOf(bw, bh)
                        val newW = (bw * scale).toInt()
                        val newH = (bh * scale).toInt()
                        val scaled = bitmap.scale(newW, newH)
                        bitmap.recycle()
                        bitmap = scaled
                        if (showToast) runOnUiThread {
                            Toast.makeText(this, "Original: ${origW}x${origH} → downsampled to ${newW}x${newH}", Toast.LENGTH_LONG).show()
                        }
                    } else {
                        if (showToast) runOnUiThread {
                            Toast.makeText(this, "Image loaded: ${origW}x${origH}", Toast.LENGTH_SHORT).show()
                        }
                    }
                }

                if (bitmap != null) {
                    bitmap = correctExifOrientation(bitmap, uri)
                    val finalBitmap = bitmap
                    val (mu, w, wInv) = computeZCAFromBitmap(finalBitmap, null)
                    glSurfaceView.queueEvent {
                        renderer.loadBitmapWithPrecomputedZCA(finalBitmap, mu, w, wInv)
                        if (restoreZoom != null) renderer.setZoomScale(restoreZoom)
                        else renderer.resetZoom()
                        glSurfaceView.requestRender()
                        if (isNewLoad) {
                            runOnUiThread {
                                presets.add(Preset(180, 180, 180, 100, 100, ChannelMode.COLOR, false, null, null, null, null))
                                updatePresetUI()
                            }
                        }
                    }
                } else {
                    if (showToast) runOnUiThread {
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading image", e)
                if (showToast) runOnUiThread {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun showSplashState() {
        splashImage.visibility = View.VISIBLE
        glSurfaceView.visibility = View.GONE
        topOverlay.visibility = View.VISIBLE
        bottomOverlay.visibility = View.GONE
        btnPreset.visibility = View.GONE
        btnRandom.visibility = View.GONE
    }

    private fun hideSplashState() {
        splashImage.visibility = View.GONE
        glSurfaceView.visibility = View.VISIBLE
        topOverlay.visibility = View.VISIBLE
        bottomOverlay.visibility = View.VISIBLE
        btnRandom.visibility = View.VISIBLE
    }
    
    private fun showImageControls() {
        findViewById<LinearLayout>(R.id.imageControls).visibility = View.VISIBLE
        findViewById<LinearLayout>(R.id.varPushControls).visibility = View.VISIBLE
        btnPreset.visibility = View.VISIBLE
    }

    private fun setupGestureDetector() {
        scaleGestureDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                if (!roiMode) {
                    val scaleFactor = detector.scaleFactor
                    val w = glSurfaceView.width.toFloat().coerceAtLeast(1f)
                    val h = glSurfaceView.height.toFloat().coerceAtLeast(1f)
                    val focalX =  (detector.focusX / w) * 2f - 1f
                    val focalY = -((detector.focusY / h) * 2f - 1f)
                    glSurfaceView.queueEvent {
                        renderer.updateZoomWithFocalPoint(scaleFactor, focalX, focalY)
                        glSurfaceView.requestRender()
                    }
                }
                return true
            }
        })

        gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true
            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (roiMode) {
                    processROIZCA()
                } else {
                    resetParameters(resetZCA = false)
                }
                return true
            }
        })
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        return handleTouchGestures(event)
    }

    private fun isTouchOnImageArea(event: MotionEvent): Boolean {
        val topBottom = topOverlay.bottom
        val bottomTop = bottomOverlay.top
        for (i in 0 until event.pointerCount) {
            val y = event.getY(i)
            if (y < topBottom || y > bottomTop) return false
        }
        return true
    }

    private fun handleTouchGestures(event: MotionEvent): Boolean {
        if (roiMode) {
            handleROIPaint(event)
            return true
        }

        if (!isTouchOnImageArea(event)) return false

        if (splitDragActive || (renderer.isSplitActive() && event.pointerCount == 1 && isTouchNearSplitLine(event))) {
            handleSplitDrag(event)
            return true
        }

        scaleGestureDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)

        val pointerCount = event.pointerCount
        
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                stopRandomRotation()
                activePointerId = event.getPointerId(0)
                lastTouchX = event.x
                lastTouchY = event.y
                isRotating = false
                floatAngleX = (seekBarRotX.progress - 180).toFloat()
                floatAngleY = (seekBarRotY.progress - 180).toFloat()
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (pointerCount == 2) {
                    isRotating = true
                    isPanning = true
                    lastRotationAngle = getRotationAngle(event)
                    floatAngleZ = (seekBarRotZ.progress - 180).toFloat()
                    lastMidX = (event.getX(0) + event.getX(1)) / 2f
                    lastMidY = (event.getY(0) + event.getY(1)) / 2f
                }
            }
            MotionEvent.ACTION_MOVE -> {
                when {
                    pointerCount == 2 && (isRotating || isPanning) -> {
                        val midX = (event.getX(0) + event.getX(1)) / 2f
                        val midY = (event.getY(0) + event.getY(1)) / 2f

                        // Pan: midpoint displacement
                        if (isPanning) {
                            val dmx = midX - lastMidX
                            val dmy = midY - lastMidY
                            val w = glSurfaceView.width.toFloat().coerceAtLeast(1f)
                            val h = glSurfaceView.height.toFloat().coerceAtLeast(1f)
                            val ndcDx =  (dmx / w) * 2f
                            val ndcDy = -(dmy / h) * 2f
                            glSurfaceView.queueEvent {
                                renderer.panImage(ndcDx, ndcDy)
                            }
                        }
                        lastMidX = midX
                        lastMidY = midY

                        // Rotation Z: angle change between fingers
                        if (isRotating) {
                            val currentAngle = getRotationAngle(event)
                            var angleDelta = currentAngle - lastRotationAngle
                            while (angleDelta > 180f) angleDelta -= 360f
                            while (angleDelta < -180f) angleDelta += 360f

                            floatAngleZ += angleDelta
                            while (floatAngleZ > 180f) floatAngleZ -= 360f
                            while (floatAngleZ < -180f) floatAngleZ += 360f

                            seekBarRotZ.progress = (floatAngleZ + 180f).toInt().coerceIn(0, 360)
                            btnICA.setTextColor(android.graphics.Color.WHITE)
                            glSurfaceView.queueEvent {
                                renderer.setRotationAngles(floatAngleX, floatAngleY, floatAngleZ)
                            }
                            labelRotZ.text = getString(R.string.angle_degrees, floatAngleZ.toInt())
                            lastRotationAngle = currentAngle
                        }

                        glSurfaceView.requestRender()
                    }
                    pointerCount == 1 && !isRotating && !scaleGestureDetector.isInProgress -> {
                        val pointerIndex = event.findPointerIndex(activePointerId)
                        if (pointerIndex >= 0) {
                            val x = event.getX(pointerIndex)
                            val y = event.getY(pointerIndex)
                            val dx = x - lastTouchX
                            val dy = y - lastTouchY

                            val screenMin = minOf(glSurfaceView.width, glSurfaceView.height).toFloat().coerceAtLeast(1f)
                            floatAngleY += (dx / screenMin) * 360f
                            floatAngleX -= (dy / screenMin) * 360f

                            while (floatAngleX > 180f) floatAngleX -= 360f
                            while (floatAngleX < -180f) floatAngleX += 360f
                            while (floatAngleY > 180f) floatAngleY -= 360f
                            while (floatAngleY < -180f) floatAngleY += 360f

                            seekBarRotX.progress = (floatAngleX + 180f).toInt().coerceIn(0, 360)
                            seekBarRotY.progress = (floatAngleY + 180f).toInt().coerceIn(0, 360)

                            val angleZ = (seekBarRotZ.progress - 180).toFloat()
                            btnICA.setTextColor(android.graphics.Color.WHITE)
                            glSurfaceView.queueEvent {
                                renderer.setRotationAngles(floatAngleX, floatAngleY, angleZ)
                            }
                            glSurfaceView.requestRender()
                            labelRotX.text = getString(R.string.angle_degrees, floatAngleX.toInt())
                            labelRotY.text = getString(R.string.angle_degrees, floatAngleY.toInt())

                            lastTouchX = x
                            lastTouchY = y
                        }
                    }
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                activePointerId = MotionEvent.INVALID_POINTER_ID
                isRotating = false
                isPanning = false
            }
            MotionEvent.ACTION_POINTER_UP -> {
                if (pointerCount <= 2) {
                    isRotating = false
                    isPanning = false
                }
            }
        }
        return true
    }

    private fun getRotationAngle(event: MotionEvent): Float {
        val dx = event.getX(0) - event.getX(1)
        val dy = event.getY(0) - event.getY(1)
        return (atan2(dy, dx) * (180.0 / PI)).toFloat()
    }

    private fun isTouchNearSplitLine(event: MotionEvent): Boolean {
        if (event.pointerCount != 1) return false
        val imageRect = renderer.getImageRectScreen() ?: return false
        val lineX = imageRect.left + renderer.getSplitPos() * imageRect.width()
        val grab = (0.04f * imageRect.width() / renderer.getZoomScale().coerceAtLeast(0.01f)).coerceAtLeast(40f)
        return abs(event.x - lineX) < grab
    }

    private fun updateSplitPosFromTouch(x: Float) {
        val imageRect = renderer.getImageRectScreen() ?: return
        val pos = (x - imageRect.left) / imageRect.width()
        renderer.setSplitPos(pos)
        glSurfaceView.requestRender()
    }

    private fun handleSplitDrag(event: MotionEvent) {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                splitDragActive = true
                stopRandomRotation()
                updateSplitPosFromTouch(event.x)
            }
            MotionEvent.ACTION_MOVE -> {
                if (splitDragActive && event.pointerCount == 1) {
                    updateSplitPosFromTouch(event.getX(0))
                }
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                splitDragActive = false
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                splitDragActive = false
            }
        }
    }

    private fun setChannelMode(mode: ChannelMode) {
        currentChannelMode = mode
        
        btnChannelC.setTextColor(if (mode == ChannelMode.COLOR) android.graphics.Color.RED else android.graphics.Color.WHITE)
        btnChannelR.setTextColor(if (mode == ChannelMode.RED) android.graphics.Color.RED else android.graphics.Color.WHITE)
        btnChannelG.setTextColor(if (mode == ChannelMode.GREEN) android.graphics.Color.RED else android.graphics.Color.WHITE)
        btnChannelB.setTextColor(if (mode == ChannelMode.BLUE) android.graphics.Color.RED else android.graphics.Color.WHITE)
        
        glSurfaceView.queueEvent {
            renderer.setChannelMode(mode.ordinal)
            glSurfaceView.requestRender()
        }
    }
    
    private fun showInfoDialog() {
        val message = android.text.SpannableString(
            "Colour Visualisation Tool\n\n" +
            "Version 1.8.7 — 2026\n\n" +
            "Fabrice Monna\n" +
            "Fabrice.Monna@ube.fr\n\n" +
            "All rights reserved."
        )
        val email = "Fabrice.Monna@ube.fr"
        val start = message.indexOf(email)
        if (start >= 0) {
            message.setSpan(
                object : android.text.style.ClickableSpan() {
                    override fun onClick(widget: View) {
                        val intent = Intent(Intent.ACTION_SENDTO).apply {
                            data = "mailto:$email".toUri()
                            putExtra(Intent.EXTRA_SUBJECT, "Rascal")
                        }
                        startActivity(Intent.createChooser(intent, "Send email"))
                    }
                },
                start, start + email.length,
                android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            message.setSpan(
                android.text.style.ForegroundColorSpan("#4FC3F7".toColorInt()),
                start, start + email.length,
                android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
            )
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle("Rascal")
            .setMessage(message)
            .setPositiveButton("OK", null)
            .create()
        dialog.show()
        (dialog.findViewById<android.widget.TextView>(android.R.id.message))?.movementMethod =
            android.text.method.LinkMovementMethod.getInstance()
    }

    private fun savePreset() {
        if (currentImageUri == null) {
            Toast.makeText(this, "No image loaded", Toast.LENGTH_SHORT).show()
            return
        }
        if (presets.size >= maxPresets) {
            Toast.makeText(this, "Maximum presets reached (5)", Toast.LENGTH_SHORT).show()
            return
        }

        val rotX = seekBarRotX.progress
        val rotY = seekBarRotY.progress
        val rotZ = seekBarRotZ.progress
        val push = seekBarPushFactor.progress
        val varRest = seekBarVarRestore.progress
        val channel = currentChannelMode
        val roiActiveNow = roiActive

        glSurfaceView.queueEvent {
            val rotMat = renderer.getRotationMatrixOverride()
            val bitmap = if (roiActiveNow) renderer.getCurrentBitmap() else null
            val zcaTriple = if (roiActiveNow) renderer.getCurrentZCAMatrices() else null
            runOnUiThread {
                if (presets.size >= maxPresets) {
                    bitmap?.recycle()
                    Toast.makeText(this, "Maximum presets reached (5)", Toast.LENGTH_SHORT).show()
                    return@runOnUiThread
                }
                if (roiActiveNow && zcaTriple != null) {
                    presets.add(Preset(rotX, rotY, rotZ, push, varRest, channel, true, bitmap, zcaTriple.first, zcaTriple.second, zcaTriple.third, rotMat))
                } else {
                    presets.add(Preset(rotX, rotY, rotZ, push, varRest, channel, false, null, null, null, null, rotMat))
                }
                updatePresetUI()
                Toast.makeText(this, "Preset ${presets.size} saved", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun recallPreset(index: Int) {
        val preset = presets.getOrNull(index) ?: return
        disableSplit()

        seekBarRotX.progress = preset.rotX
        seekBarRotY.progress = preset.rotY
        seekBarRotZ.progress = preset.rotZ
        seekBarPushFactor.progress = preset.pushFactor
        seekBarVarRestore.progress = preset.varRestore
        setChannelMode(preset.channelMode)
        updateRotation()
        updatePushFactor()
        updateVarRestore()

        if (preset.roiActive && preset.roiBitmap != null) {
            roiActive = true
            btnROI.setTextColor(android.graphics.Color.RED)
            clearROIPaint()
            glSurfaceView.queueEvent {
                renderer.restorePresetWithROI(preset.roiBitmap, preset.roiMu!!, preset.roiW!!, preset.roiWInv!!)
                if (preset.rotationMatrix != null) renderer.setRotationMatrixColumnMajor(preset.rotationMatrix)
                else renderer.clearICABase()
                glSurfaceView.requestRender()
            }
        } else {
            roiActive = false
            roiImageMask = null
            btnROI.setTextColor(android.graphics.Color.WHITE)
            clearROIPaint()
            glSurfaceView.queueEvent {
                renderer.restoreOriginalBitmap()
                if (preset.rotationMatrix != null) renderer.setRotationMatrixColumnMajor(preset.rotationMatrix)
                else renderer.clearICABase()
                glSurfaceView.requestRender()
            }
        }
        btnICA.setTextColor(if (preset.rotationMatrix != null) android.graphics.Color.YELLOW else android.graphics.Color.WHITE)
    }

    private fun clearAllPresets() {
        presets.forEach { it.roiBitmap?.recycle() }
        presets.clear()
        updatePresetUI()
    }

    private fun clearExtraPresets() {
        presets.drop(1).forEach { it.roiBitmap?.recycle() }
        if (presets.size > 1) presets.subList(1, presets.size).clear()
        updatePresetUI()
    }

    private fun updatePresetUI() {
        for (i in presetNumberButtons.indices) {
            presetNumberButtons[i].visibility = if (i < presets.size) View.VISIBLE else View.GONE
        }
        btnPresetClear.visibility = if (presets.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun toggleROIMode() {
        disableSplit()
        if (roiMode) {
            // Désactiver le mode peinture
            roiMode = false
            setROIPaintingMode(false)
            roiHintText.visibility = View.GONE
            roiOverlay.visibility = View.GONE
            btnROI.setTextColor(android.graphics.Color.WHITE)
            clearROIPaint()

            // Si ROI est actif, désactiver et restaurer l'image + ZCA initiales (déjà en mémoire)
            if (roiActive) {
                roiActive = false
                roiImageMask = null
                glSurfaceView.queueEvent {
                    renderer.restoreOriginalBitmap()
                    glSurfaceView.requestRender()
                }
            }
        } else if (roiActive) {
            // ROI est actif mais pas en mode peinture, désactiver et restaurer
            roiActive = false
            roiImageMask = null
            btnROI.setTextColor(android.graphics.Color.WHITE)
            roiOverlay.visibility = View.GONE
            clearROIPaint()
            glSurfaceView.queueEvent {
                renderer.restoreOriginalBitmap()
                glSurfaceView.requestRender()
            }
        } else {
            // Reset angles/push/varRestore et zoom/pan avant d'activer le ROI
            seekBarRotX.progress = 180
            seekBarRotY.progress = 180
            seekBarRotZ.progress = 180
            seekBarPushFactor.progress = 100
            seekBarVarRestore.progress = 100
            updateRotation()
            updatePushFactor()
            updateVarRestore()
            glSurfaceView.queueEvent {
                renderer.resetZoom()
                glSurfaceView.requestRender()
            }

            // Activer le mode ROI peinture
            roiMode = true
            setROIPaintingMode(true)
            btnROI.setTextColor(android.graphics.Color.RED)
            roiHintText.text = getString(R.string.roi_hint)
            roiHintText.visibility = View.VISIBLE
            roiOverlay.visibility = View.VISIBLE
            
            // Initialiser le bitmap de peinture, ou le recréer si les dimensions ont changé
            val existingOverlay = roiPaintOverlay
            if (existingOverlay == null ||
                existingOverlay.width != glSurfaceView.width ||
                existingOverlay.height != glSurfaceView.height) {
                existingOverlay?.recycle()
                roiPaintOverlay = null
                initializeROIPaint()
            }
            
            Toast.makeText(this, "ROI mode: Paint with finger, double tap to process", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun setROIPaintingMode(painting: Boolean) {
        val enable = !painting
        btnLoadImage.isEnabled = enable
        btnReset.isEnabled = enable
        btnSplit.isEnabled = enable
        btnSave.isEnabled = enable
        btnChannelC.isEnabled = enable
        btnChannelR.isEnabled = enable
        btnChannelG.isEnabled = enable
        btnChannelB.isEnabled = enable
        btnICA.isEnabled = enable
        btnInfo.isEnabled = enable
        btnPreset.isEnabled = enable
        btnPresetClear.isEnabled = enable
        for (b in presetNumberButtons) b.isEnabled = enable
        seekBarRotX.isEnabled = enable
        seekBarRotY.isEnabled = enable
        seekBarRotZ.isEnabled = enable
        seekBarPushFactor.isEnabled = enable
        seekBarVarRestore.isEnabled = enable
    }

    private fun initializeROIPaint() {
        val width = glSurfaceView.width
        val height = glSurfaceView.height
        if (width <= 0 || height <= 0) {
            glSurfaceView.post { initializeROIPaint() }
            return
        }
        roiPaintOverlay = createBitmap(width, height)
        roiCanvas = android.graphics.Canvas(roiPaintOverlay!!)
        roiPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.argb(127, 255, 255, 0)
            strokeWidth = 60f
            style = android.graphics.Paint.Style.STROKE
            strokeCap = android.graphics.Paint.Cap.ROUND
            strokeJoin = android.graphics.Paint.Join.ROUND
            isAntiAlias = true
        }
        Log.d(TAG, "ROI paint initialized: ${width}x${height}")
    }
    
    private fun handleROIPaint(event: MotionEvent) {
        // Ne passer au gestureDetector que DOWN/UP : MOVE annule la détection du double tap
        if (event.action == MotionEvent.ACTION_DOWN || event.action == MotionEvent.ACTION_UP) {
            gestureDetector.onTouchEvent(event)
        }

        if (roiCanvas == null || roiPaint == null) return

        val imageRect = renderer.getImageRectScreen()

        fun clamp(x: Float, y: Float): Pair<Float, Float> {
            if (imageRect == null) return Pair(x, y)
            return Pair(x.coerceIn(imageRect.left, imageRect.right),
                        y.coerceIn(imageRect.top,  imageRect.bottom))
        }

        val (x, y) = clamp(event.x, event.y)

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = x
                lastTouchY = y
            }
            MotionEvent.ACTION_MOVE -> {
                roiCanvas?.drawLine(lastTouchX, lastTouchY, x, y, roiPaint!!)
                lastTouchX = x
                lastTouchY = y
                roiOverlay.invalidate()
            }
        }
    }
    
    private fun clearROIPaint() {
        roiPaintOverlay?.eraseColor(android.graphics.Color.TRANSPARENT)
        roiOverlay.invalidate()
    }
    
    private fun processROIZCA() {
        if (roiPaintOverlay == null) {
            Toast.makeText(this, "No ROI painted. Please paint an area first.", Toast.LENGTH_LONG).show()
            Log.w(TAG, "processROIZCA: roiPaintOverlay is null (glSurfaceView=${glSurfaceView.width}x${glSurfaceView.height})")
            return
        }
        
        if (currentImageUri == null) {
            Toast.makeText(this, "No image loaded", Toast.LENGTH_SHORT).show()
            return
        }
        
        // Désactiver le mode peinture mais garder le bouton rouge
        roiMode = false
        setROIPaintingMode(false)
        roiHintText.visibility = View.GONE
        // Garder roiOverlay VISIBLE (transparent) pour bloquer zoom/pan sur glSurfaceView
        roiOverlay.visibility = View.VISIBLE

        roiActive = true
        btnROI.setTextColor(android.graphics.Color.RED)

        Toast.makeText(this, "Computing ZCA on ROI…", Toast.LENGTH_LONG).show()

        // Capturer imageRect sur le UI thread avant de lancer le thread background
        val imageRect = renderer.getImageRectScreen()
        val screenW = glSurfaceView.width
        val screenH = glSurfaceView.height

        // Copier le bitmap de peinture avant de l'effacer, puis tout faire en background
        val bitmapConfig: Bitmap.Config = roiPaintOverlay!!.config ?: Bitmap.Config.ARGB_8888
        val paintCopy = roiPaintOverlay!!.copy(bitmapConfig, false)
        clearROIPaint()

        recomputeZCAWithPaint(paintCopy, imageRect, screenW, screenH)
    }
    
    private fun openUriInputStream(uri: Uri): java.io.InputStream? {
        return if (uri.scheme == "file") {
            val path = uri.path ?: return null
            java.io.FileInputStream(path)
        } else {
            contentResolver.openInputStream(uri)
        }
    }

    private fun correctExifOrientation(bitmap: Bitmap, uri: Uri): Bitmap {
        val inputStream = openUriInputStream(uri) ?: return bitmap
        val exif = androidx.exifinterface.media.ExifInterface(inputStream)
        inputStream.close()
        val rotation = when (exif.getAttributeInt(
            androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
            androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
        )) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90  -> 90f
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> return bitmap
        }
        val matrix = android.graphics.Matrix().apply { postRotate(rotation) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    private fun extractROIMask(bitmap: Bitmap): BooleanArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val mask = BooleanArray(pixels.size) { (pixels[it] ushr 24) > 0 }
        Log.d(TAG, "extractROIMask: ${width}x${height}, ${mask.count { it }} pixels selected")
        return mask
    }
    
    private fun recomputeZCAWithPaint(paintBitmap: Bitmap, imageRect: android.graphics.RectF?, screenW: Int, screenH: Int) {
        if (currentImageUri == null) {
            Toast.makeText(this, "No image loaded", Toast.LENGTH_SHORT).show()
            return
        }
        if (roiComputationRunning) {
            Log.w(TAG, "recomputeZCAWithPaint: calcul ROI déjà en cours, ignoré")
            paintBitmap.recycle()
            return
        }
        roiComputationRunning = true
        Thread {
            try {
                val screenMask = extractROIMask(paintBitmap)
                paintBitmap.recycle()
                if (screenMask.count { it } < 10) {
                    roiComputationRunning = false
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "ROI too small, paint a larger area", Toast.LENGTH_SHORT).show()
                        roiActive = false
                        roiImageMask = null
                        btnROI.setTextColor(android.graphics.Color.WHITE)
                        roiOverlay.visibility = View.GONE
                        setROIPaintingMode(false)
                    }
                    return@Thread
                }
                recomputeZCABackground(screenMask, imageRect, screenW, screenH)
            } catch (e: Exception) {
                Log.e(TAG, "Error in recomputeZCAWithPaint", e)
                roiComputationRunning = false
                runOnUiThread {
                    roiOverlay.visibility = View.GONE
                    setROIPaintingMode(false)
                }
            }
        }.start()
    }

    private fun recomputeZCABackground(screenMask: BooleanArray?, imageRect: android.graphics.RectF?, screenW: Int, screenH: Int) {
        val snapshotUri = currentImageUri ?: return
        Thread {
            try {
                // Lire dimensions sans charger
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                var inputStream = openUriInputStream(snapshotUri)
                BitmapFactory.decodeStream(inputStream, null, options)
                inputStream?.close()

                // Même downsampling que loadImageFromUri
                val maxDimension = 4096
                var inSampleSize = 1
                val iw = options.outWidth; val ih = options.outHeight
                while ((iw / inSampleSize) > maxDimension * 2 || (ih / inSampleSize) > maxDimension * 2) {
                    inSampleSize *= 2
                }

                inputStream = openUriInputStream(snapshotUri)
                var bitmap = BitmapFactory.decodeStream(inputStream, null,
                    BitmapFactory.Options().apply { this.inSampleSize = inSampleSize })
                inputStream?.close()

                // Redimensionnement proportionnel exact si le grand côté dépasse 4096
                if (bitmap != null) {
                    val bw = bitmap.width; val bh = bitmap.height
                    if (bw > maxDimension || bh > maxDimension) {
                        val scale = maxDimension.toFloat() / maxOf(bw, bh)
                        val scaled = bitmap.scale((bw * scale).toInt(), (bh * scale).toInt())
                        bitmap.recycle()
                        bitmap = scaled
                    }
                }

                // Correction EXIF indispensable : le masque est peint sur l'image corrigée
                bitmap = bitmap?.let { correctExifOrientation(it, snapshotUri) }

                if (bitmap != null) {
                    // Mapper le masque écran → masque image via le rectangle réel de l'image
                    val imageMask = if (screenMask != null && imageRect != null) {
                        screenMaskToImageMask(screenMask, screenW, screenH, bitmap.width, bitmap.height, imageRect)
                    } else if (screenMask != null && roiPaintOverlay != null) {
                        resizeMask(screenMask, roiPaintOverlay!!.width, roiPaintOverlay!!.height, bitmap.width, bitmap.height)
                    } else null
                    // Sauvegarder le masque image pour l'ICA (l'overlay sera effacé)
                    roiImageMask = imageMask

                    val pixelCount = imageMask?.count { it } ?: (bitmap.width * bitmap.height)
                    Log.d(TAG, "ROI: $pixelCount pixels (bitmap ${bitmap.width}x${bitmap.height})")

                    // Calculer ZCA locale sur les pixels ROI uniquement
                    val (muNew, wNew, wInvNew) = computeZCAFromBitmap(bitmap, imageMask)
                    Log.d(TAG, "New ZCA after ROI (local): mu=[${muNew[0]},${muNew[1]},${muNew[2]}]")
                    Log.d(TAG, "W_roi: [${wNew[0]},${wNew[1]},${wNew[2]}] [${wNew[3]},${wNew[4]},${wNew[5]}] [${wNew[6]},${wNew[7]},${wNew[8]}]")

                    // Envoyer le bitmap original + matrices ROI au renderer (pas de transformation pixel)
                    glSurfaceView.queueEvent {
                        try {
                            renderer.loadBitmapAsROIWithMatrices(bitmap, muNew, wNew, wInvNew)
                            glSurfaceView.requestRender()
                        } finally {
                            bitmap.recycle()
                        }
                    }
                    roiComputationRunning = false
                    runOnUiThread {
                        roiOverlay.visibility = View.GONE
                        setROIPaintingMode(false)
                    }

                } else {
                    roiComputationRunning = false
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                        roiOverlay.visibility = View.GONE
                        setROIPaintingMode(false)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error recomputing ZCA", e)
                roiComputationRunning = false
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    roiOverlay.visibility = View.GONE
                    setROIPaintingMode(false)
                }
            }
        }.start()
    }
    

    @Suppress("UNUSED_PARAMETER", "LocalVariableName")
    private fun computeZCAFromBitmap(bitmap: Bitmap, mask: BooleanArray?): Triple<FloatArray, FloatArray, FloatArray> {
        val width = bitmap.width
        val height = bitmap.height
        val rowPixels = IntArray(width)

        // Pass 1: mean
        var n = 0
        val mu = FloatArray(3)
        for (y in 0 until height) {
            bitmap.getPixels(rowPixels, 0, width, 0, y, width, 1)
            for (x in 0 until width) {
                if (mask == null || mask[y * width + x]) {
                    val p = rowPixels[x]
                    mu[0] += srgbToLin(((p shr 16) and 0xFF) / 255f)
                    mu[1] += srgbToLin(((p shr 8)  and 0xFF) / 255f)
                    mu[2] += srgbToLin((p           and 0xFF) / 255f)
                    n++
                }
            }
        }
        if (n <= 1) return Triple(FloatArray(3), FloatArray(9) { if (it % 4 == 0) 1f else 0f }, FloatArray(9) { if (it % 4 == 0) 1f else 0f })
        mu[0] /= n.toFloat(); mu[1] /= n.toFloat(); mu[2] /= n.toFloat()

        // Pass 2: covariance (double precision)
        val cov = DoubleArray(9)
        for (y in 0 until height) {
            bitmap.getPixels(rowPixels, 0, width, 0, y, width, 1)
            for (x in 0 until width) {
                if (mask == null || mask[y * width + x]) {
                    val p = rowPixels[x]
                    val r = srgbToLin(((p shr 16) and 0xFF) / 255f).toDouble() - mu[0]
                    val g = srgbToLin(((p shr 8)  and 0xFF) / 255f).toDouble() - mu[1]
                    val b = srgbToLin((p           and 0xFF) / 255f).toDouble() - mu[2]
                    cov[0]+=r*r; cov[1]+=r*g; cov[2]+=r*b
                    cov[3]+=g*r; cov[4]+=g*g; cov[5]+=g*b
                    cov[6]+=b*r; cov[7]+=b*g; cov[8]+=b*b
                }
            }
        }
        for (i in 0..8) cov[i] /= (n - 1).toDouble()

        // Eigen decomposition (Jacobi) in double
        val A = cov.clone()
        val V = DoubleArray(9) { if (it % 4 == 0) 1.0 else 0.0 }
        for (iter in 0 until 50) {
            var p = 0; var q = 1; var maxV = abs(A[1])
            if (abs(A[2]) > maxV) { p=0; q=2; maxV=abs(A[2]) }
            if (abs(A[5]) > maxV) { p=1; q=2 }
            if (abs(A[p*3+q]) < 1e-12) break
            val theta = 0.5 * atan2(2.0*A[p*3+q], A[q*3+q]-A[p*3+p])
            val c = cos(theta); val s = sin(theta)
            val Anew = A.clone()
            val Vnew = V.clone()
            Anew[p*3+p] = c*c*A[p*3+p] - 2*c*s*A[p*3+q] + s*s*A[q*3+q]
            Anew[q*3+q] = s*s*A[p*3+p] + 2*c*s*A[p*3+q] + c*c*A[q*3+q]
            Anew[p*3+q] = 0.0; Anew[q*3+p] = 0.0
            val r3 = if (p==0 && q==1) 2 else if (p==0) 1 else 0
            Anew[r3*3+p] = c*A[r3*3+p] - s*A[r3*3+q]; Anew[p*3+r3] = Anew[r3*3+p]
            Anew[r3*3+q] = s*A[r3*3+p] + c*A[r3*3+q]; Anew[q*3+r3] = Anew[r3*3+q]
            for (i in 0..2) {
                Vnew[i*3+p] = c*V[i*3+p] - s*V[i*3+q]
                Vnew[i*3+q] = s*V[i*3+p] + c*V[i*3+q]
            }
            for (i in 0..8) { A[i]=Anew[i]; V[i]=Vnew[i] }
        }

        // Build W = V @ diag(1/sqrt(lambda)) @ V^T  and  Winv = V @ diag(sqrt(lambda)) @ V^T
        val wLin    = FloatArray(9)
        val wInvLin = FloatArray(9)
        for (i in 0..2) {
            val ev = A[i*3+i] + 1e-8
            val inv = 1.0 / sqrt(ev)
            val fwd = sqrt(ev)
            for (j in 0..2) {
                for (k in 0..2) {
                    // W[j,k] += V[j,i] * (1/sqrt(lambda_i)) * V[k,i]
                    wLin   [j*3+k] += (V[j*3+i] * inv * V[k*3+i]).toFloat()
                    wInvLin[j*3+k] += (V[j*3+i] * fwd * V[k*3+i]).toFloat()
                }
            }
        }

        Log.d(TAG, "computeZCAFromBitmap: n=$n mu=[${mu[0]},${mu[1]},${mu[2]}] W00=${wLin[0]} W11=${wLin[4]} W22=${wLin[8]}")
        return Triple(mu, wLin, wInvLin)
    }

    private fun screenMaskToImageMask(
        screenMask: BooleanArray,
        screenW: Int, screenH: Int,
        imageW: Int, imageH: Int,
        imageRect: android.graphics.RectF
    ): BooleanArray {
        val dst = BooleanArray(imageW * imageH)
        for (y in 0 until imageH) {
            val sy = (imageRect.top + (y + 0.5f) / imageH * imageRect.height())
                .toInt().coerceIn(0, screenH - 1)
            for (x in 0 until imageW) {
                val sx = (imageRect.left + (x + 0.5f) / imageW * imageRect.width())
                    .toInt().coerceIn(0, screenW - 1)
                dst[y * imageW + x] = screenMask[sy * screenW + sx]
            }
        }
        Log.d(TAG, "screenMaskToImageMask: ${imageW}x${imageH}, pixels: ${dst.count { it }}")
        return dst
    }

    private fun resizeMask(srcMask: BooleanArray, srcWidth: Int, srcHeight: Int, dstWidth: Int, dstHeight: Int): BooleanArray {
        val dstMask = BooleanArray(dstWidth * dstHeight)
        
        // Redimensionnement par échantillonnage du plus proche voisin
        for (y in 0 until dstHeight) {
            for (x in 0 until dstWidth) {
                val srcX = (x * srcWidth / dstWidth).coerceIn(0, srcWidth - 1)
                val srcY = (y * srcHeight / dstHeight).coerceIn(0, srcHeight - 1)
                dstMask[y * dstWidth + x] = srcMask[srcY * srcWidth + srcX]
            }
        }
        
        Log.d(TAG, "Mask resized from ${srcWidth}x${srcHeight} to ${dstWidth}x${dstHeight}, pixels: ${dstMask.count { it }}")
        return dstMask
    }
    
    
    private fun copyExifData(sourceUri: Uri, destUri: Uri): Boolean {
        var gpsCopied = false
        try {
            val sourceStream = contentResolver.openInputStream(sourceUri) ?: return false
            val srcExif = androidx.exifinterface.media.ExifInterface(sourceStream)
            sourceStream.close()

            val destStream = contentResolver.openFileDescriptor(destUri, "rw") ?: return false
            val dstExif = androidx.exifinterface.media.ExifInterface(destStream.fileDescriptor)

            val tags = listOf(
                androidx.exifinterface.media.ExifInterface.TAG_MAKE,
                androidx.exifinterface.media.ExifInterface.TAG_MODEL,
                androidx.exifinterface.media.ExifInterface.TAG_DATETIME,
                androidx.exifinterface.media.ExifInterface.TAG_DATETIME_ORIGINAL,
                androidx.exifinterface.media.ExifInterface.TAG_DATETIME_DIGITIZED,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE_REF,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE_REF,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE_REF,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_TIMESTAMP,
                androidx.exifinterface.media.ExifInterface.TAG_GPS_DATESTAMP,
                androidx.exifinterface.media.ExifInterface.TAG_FOCAL_LENGTH,
                androidx.exifinterface.media.ExifInterface.TAG_F_NUMBER,
                androidx.exifinterface.media.ExifInterface.TAG_EXPOSURE_TIME,
                androidx.exifinterface.media.ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY,
                androidx.exifinterface.media.ExifInterface.TAG_IMAGE_WIDTH,
                androidx.exifinterface.media.ExifInterface.TAG_IMAGE_LENGTH
            )
            for (tag in tags) {
                srcExif.getAttribute(tag)?.let { dstExif.setAttribute(tag, it) }
            }
            val lat = srcExif.getAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE)
            val lon = srcExif.getAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE)
            if (lat != null && lon != null) {
                gpsCopied = true
            } else {
                val loc = lastKnownLocation
                if (loc != null) {
                    fun toDms(deg: Double): String {
                        val d = deg.toInt()
                        val mFull = (deg - d) * 60.0
                        val m = mFull.toInt()
                        val s = (mFull - m) * 60.0
                        return "$d/1,$m/1,${(s * 1000).toInt()}/1000"
                    }
                    dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE, toDms(abs(loc.latitude)))
                    dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE_REF, if (loc.latitude >= 0) "N" else "S")
                    dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE, toDms(abs(loc.longitude)))
                    dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE_REF, if (loc.longitude >= 0) "E" else "W")
                    if (loc.hasAltitude()) {
                        val alt = abs(loc.altitude)
                        dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE, "${(alt * 100).toInt()}/100")
                        dstExif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE_REF, if (loc.altitude >= 0) "0" else "1")
                    }
                    gpsCopied = true
                    Log.d(TAG, "GPS written from device location: ${loc.latitude}, ${loc.longitude}")
                }
            }
            dstExif.saveAttributes()
            destStream.close()
            Log.d(TAG, "EXIF data copied from source to saved image (GPS: $gpsCopied)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy EXIF data", e)
        }
        return gpsCopied
    }

    private fun saveProcessedImage() {
        if (currentImageUri == null) {
            Toast.makeText(this, "No image loaded", Toast.LENGTH_SHORT).show()
            return
        }

        Toast.makeText(this, "Saving image...", Toast.LENGTH_SHORT).show()

        glSurfaceView.queueEvent {
            val state = renderer.getFullRenderState()
            Thread {
                try {
                    val src = state.bitmap
                    if (src == null) {
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "No bitmap available", Toast.LENGTH_SHORT).show()
                        }
                        return@Thread
                    }
                    val result = applyShaderToBitmap(src, state)
                    src.recycle()
                    runOnUiThread { saveBitmapToGallery(result, currentImageUri) }
                } catch (e: OutOfMemoryError) {
                    Log.e(TAG, "OOM in applyShaderToBitmap", e)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Out of memory - image too large", Toast.LENGTH_LONG).show()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error saving processed image", e)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()
        }
    }

    @Suppress("LocalVariableName")
    private fun applyShaderToBitmap(bitmap: Bitmap, state: GLRenderer.RenderState): Bitmap {
        val bw = bitmap.width
        val bh = bitmap.height
        val pixels = IntArray(bw * bh)
        bitmap.getPixels(pixels, 0, bw, 0, 0, bw, bh)

        val mu  = state.mu
        val wM  = state.w
        val wI  = state.wInv
        val R   = state.rotation
        val psh = state.pushFactor
        val vr  = state.varRestore
        val ch  = state.channelMode

        // GL passes Kotlin row-major arrays as column-major to GLSL.
        // GLSL mat3*vec3 with column-major data = R^T_kotlin * v for non-symmetric R.
        // W and Winv are symmetric so their transpose = themselves.
        // Shader: z = W*(x-mu); z2 = R^T*z*push; eff_Winv = I + varRestore*(Winv-I); xo = mu + eff_Winv*z2
        fun mul33(a: FloatArray, b: FloatArray): FloatArray {
            val r = FloatArray(9)
            for (i in 0..2) for (j in 0..2) for (k in 0..2) r[i*3+j] += a[i*3+k] * b[k*3+j]
            return r
        }
        // eff_Winv = I + varRestore * (Winv - I)
        val effWinv = FloatArray(9) { i ->
            val eye = if (i % 4 == 0) 1f else 0f
            eye + vr * (wI[i] - eye)
        }
        val Rt = floatArrayOf(R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8])
        val T0 = mul33(effWinv, Rt)
        val T1 = mul33(T0, wM)
        val T  = FloatArray(9) { T1[it] * psh }

        for (i in pixels.indices) {
            val p  = pixels[i]
            val rl = srgbToLin(((p shr 16) and 0xFF) / 255f)
            val gl = srgbToLin(((p shr 8)  and 0xFF) / 255f)
            val bl = srgbToLin((p           and 0xFF) / 255f)

            val cr = rl - mu[0]; val cg = gl - mu[1]; val cb = bl - mu[2]

            var or = (mu[0] + T[0]*cr + T[1]*cg + T[2]*cb).coerceIn(0f, 1f)
            var og = (mu[1] + T[3]*cr + T[4]*cg + T[5]*cb).coerceIn(0f, 1f)
            var ob = (mu[2] + T[6]*cr + T[7]*cg + T[8]*cb).coerceIn(0f, 1f)

            or = linToSrgb(or); og = linToSrgb(og); ob = linToSrgb(ob)

            when (ch) {
                1 -> { og = or; ob = or }
                2 -> { or = og; ob = og }
                3 -> { or = ob; og = ob }
            }

            pixels[i] = (0xFF shl 24) or
                        ((or * 255f + 0.5f).toInt().coerceIn(0, 255) shl 16) or
                        ((og * 255f + 0.5f).toInt().coerceIn(0, 255) shl 8) or
                        (ob * 255f + 0.5f).toInt().coerceIn(0, 255)
        }

        return Bitmap.createBitmap(pixels, bw, bh, Bitmap.Config.ARGB_8888)
    }
    
    // ── ICA ──────────────────────────────────────────────────────────────────

    /**
     * Calcule Rica (rotation ICA 3×3) à partir de la ZCA courante et l'applique
     * comme nouvelle rotation de base (composée avec la rotation seekbar actuelle).
     *
     * Pipeline :
     *   1. Lit le bitmap courant sur le thread GL
     *   2. Échantillonne N_ICA pixels aléatoires, les whiten via W_lin courant
     *   3. FastICA symétrique → Rica (matrice orthogonale 3×3)
     *   4. Envoie Rica au renderer : u_rotation = Rica * R_seekbar
     */
    private fun applyICA() {
        if (currentImageUri == null) {
            Toast.makeText(this, "No image loaded", Toast.LENGTH_SHORT).show()
            return
        }
        Toast.makeText(this, "Computing ICA…", Toast.LENGTH_SHORT).show()

        // Capturer le masque ROI sauvegardé (l'overlay est effacé après le ZCA ROI)
        val savedMask: BooleanArray? = if (roiActive) roiImageMask else null

        glSurfaceView.queueEvent {
            val bitmap = renderer.getCurrentBitmap()
            val (mu, w, _) = renderer.getCurrentZCAMatrices()
            Thread {
                try {
                    if (bitmap == null) return@Thread
                    val rica = computeICARotation(bitmap, mu, w, mask = savedMask)
                    bitmap.recycle()

                    // Extraction Euler ZYX standard de R = Rz*Ry*Rx (row-major : R[r,c] = rica[r*3+c])
                    // Identique à Rascal.py RotationState.R_to_euler_zyx et index.html R_to_euler_zyx :
                    // sy = sqrt(R00² + R10²) ; x = atan2(R21, R22) ; y = atan2(-R20, sy) ; z = atan2(R10, R00)
                    val cyVal = sqrt(rica[0].toDouble()*rica[0] + rica[3].toDouble()*rica[3])
                    val angX: Float
                    val angY: Float
                    val angZ: Float
                    if (cyVal > 1e-6) {
                        angX = (atan2( rica[7].toDouble(), rica[8].toDouble()) * (180.0 / PI)).toFloat()
                        angY = (atan2(-rica[6].toDouble(), cyVal)              * (180.0 / PI)).toFloat()
                        angZ = (atan2( rica[3].toDouble(), rica[0].toDouble()) * (180.0 / PI)).toFloat()
                    } else {
                        angX = (atan2(-rica[5].toDouble(), rica[4].toDouble()) * (180.0 / PI)).toFloat()
                        angY = (atan2(-rica[6].toDouble(), cyVal)              * (180.0 / PI)).toFloat()
                        angZ = 0f
                    }
                    val progX = (angX + 180f).coerceIn(0f, 360f).toInt()
                    val progY = (angY + 180f).coerceIn(0f, 360f).toInt()
                    val progZ = (angZ + 180f).coerceIn(0f, 360f).toInt()
                    // Rebuild R(angX,angY,angZ) row-major, identique à computeRotationMatrix(angX,angY,angZ)
                    // pour éviter tout saut quand le slider efface rotationMatrixOverride
                    val rxRad = (angX * (PI / 180.0)).toFloat()
                    val ryRad = (angY * (PI / 180.0)).toFloat()
                    val rzRad = (angZ * (PI / 180.0)).toFloat()
                    val cx = cos(rxRad); val sx = sin(rxRad)
                    val cy = cos(ryRad); val sy = sin(ryRad)
                    val cz = cos(rzRad); val sz = sin(rzRad)
                    // même layout que computeRotationMatrix dans GLRenderer (R row-major → GLSL voit R^T)
                    val ricaMod = floatArrayOf(
                        cy*cz,   sx*sy*cz - cx*sz,   cx*sy*cz + sx*sz,
                        cy*sz,   sx*sy*sz + cx*cz,   cx*sy*sz - sx*cz,
                        -sy,     sx*cy,              cx*cy
                    )

                    glSurfaceView.queueEvent {
                        renderer.setRotationMatrixColumnMajor(ricaMod)
                        glSurfaceView.requestRender()
                    }
                    runOnUiThread {
                        seekBarRotX.progress = progX
                        seekBarRotY.progress = progY
                        seekBarRotZ.progress = progZ
                        labelRotX.text = getString(R.string.angle_degrees, angX.toInt())
                        labelRotY.text = getString(R.string.angle_degrees, angY.toInt())
                        labelRotZ.text = getString(R.string.angle_degrees, angZ.toInt())
                        btnICA.setTextColor(android.graphics.Color.YELLOW)
                        Toast.makeText(this@MainActivity, "ICA applied", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "ICA error", e)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "ICA error: ${e.message}", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()
        }
    }

    /**
     * FastICA déflation (tanh) sur [bitmap] whitened par [mu]/[wLin].
     * Identique à Rascal.py _fastica_rotation() + apply_ica().
     * Retourne R (3×3, row-major).
     */
    @Suppress("UNUSED_PARAMETER")
    private fun computeICARotation(
        bitmap: Bitmap,
        mu: FloatArray,
        wLin: FloatArray,
        maxSamples: Int = 100_000,
        maxIter: Int = 200,
        tol: Double = 1e-6,
        mask: BooleanArray? = null
    ): FloatArray {
        val width  = bitmap.width
        val height = bitmap.height
        val total  = width * height
        // Candidats pixels (masque ROI ou image entière)
        val candidates: IntArray = if (mask != null && mask.size == total) {
            IntArray(mask.count { it }).also { arr ->
                var i = 0
                for (idx in 0 until total) if (mask[idx]) arr[i++] = idx
            }
        } else IntArray(total) { it }
        val n = minOf(candidates.size, maxSamples)
        // Stride sampling déterministe — identique à Python
        val indices = if (n < candidates.size) {
            val stride = candidates.size.toDouble() / n
            IntArray(n) { i -> candidates[(i * stride).toInt()] }
        } else candidates

        // Whitening : Z[s] = W * (x[s] - mu)  — stocké en 3 colonnes
        val allPixels = IntArray(total)
        bitmap.getPixels(allPixels, 0, width, 0, 0, width, height)
        val z0 = DoubleArray(n); val z1 = DoubleArray(n); val z2 = DoubleArray(n)
        for (s in 0 until n) {
            val p = allPixels[indices[s]]
            val rl = srgbToLin(((p shr 16) and 0xFF) / 255f).toDouble() - mu[0]
            val gl = srgbToLin(((p shr 8)  and 0xFF) / 255f).toDouble() - mu[1]
            val bl = srgbToLin((p           and 0xFF) / 255f).toDouble() - mu[2]
            z0[s] = wLin[0]*rl + wLin[1]*gl + wLin[2]*bl
            z1[s] = wLin[3]*rl + wLin[4]*gl + wLin[5]*bl
            z2[s] = wLin[6]*rl + wLin[7]*gl + wLin[8]*bl
        }

        val nd = n.toDouble()
        // W_ica[i] = vecteur démixage de la i-ème composante (row-major)
        val wIca = Array(3) { DoubleArray(3) }

        // Vecteurs initiaux exacts de numpy.random.default_rng(i).standard_normal(3) / norm
        // Obtenir avec : for i in range(3): w=np.random.default_rng(i).standard_normal(3); w/=np.linalg.norm(w); print(f"{w[0]:.17f}, {w[1]:.17f}, {w[2]:.17f}")
        @Suppress("FloatingPointLiteralPrecision")
        val numpyInitVectors = arrayOf(
            doubleArrayOf(0.18881711923692265, -0.19839032737660414, 0.96176367860637857),  // seed=0
            doubleArrayOf(0.36353656768131110,  0.86429948675750623, 0.34760259082636707),  // seed=1
            doubleArrayOf(0.27298068055600483, -0.75481445484455556, -0.59643665782788458)  // seed=2
        )

        // FastICA déflation — identique à Python _fastica_rotation()
        for (i in 0..2) {
            var w0 = numpyInitVectors[i][0]
            var w1 = numpyInitVectors[i][1]
            var w2 = numpyInitVectors[i][2]

            for (iter in 0 until maxIter) {
                // proj = Z @ w  (scalaire par sample)
                // g = tanh(proj), gp = 1 - g²
                // w_new = (Z.T @ g) / n - mean(gp) * w
                var gz0 = 0.0; var gz1 = 0.0; var gz2 = 0.0; var mgp = 0.0
                for (s in 0 until n) {
                    val proj = z0[s]*w0 + z1[s]*w1 + z2[s]*w2
                    val g = tanh(proj)
                    val gp = 1.0 - g*g
                    gz0 += g*z0[s]; gz1 += g*z1[s]; gz2 += g*z2[s]
                    mgp += gp
                }
                mgp /= nd
                var wn0 = gz0/nd - mgp*w0
                var wn1 = gz1/nd - mgp*w1
                var wn2 = gz2/nd - mgp*w2

                // Déflation : orthogonaliser par rapport aux composantes précédentes
                for (j in 0 until i) {
                    val dot = wn0*wIca[j][0] + wn1*wIca[j][1] + wn2*wIca[j][2]
                    wn0 -= dot*wIca[j][0]; wn1 -= dot*wIca[j][1]; wn2 -= dot*wIca[j][2]
                }

                val nm = sqrt(wn0*wn0 + wn1*wn1 + wn2*wn2)
                if (nm < 1e-12) break
                wn0 /= nm; wn1 /= nm; wn2 /= nm

                val conv = abs(abs(wn0*w0 + wn1*w1 + wn2*w2) - 1.0)
                w0 = wn0; w1 = wn1; w2 = wn2
                if (conv < tol) break
            }
            wIca[i][0] = w0; wIca[i][1] = w1; wIca[i][2] = w2
        }

        // SVD 3×3 sur wIca pour obtenir la matrice orthogonale la plus proche
        // R = U @ Vt  (même que Python)
        // SVD 3×3 via Jacobi sur wIca * wIca.T puis reconstruction
        // On utilise la décomposition via Jacobi sur A = wIca (pas symétrique)
        // Approche : one-sided Jacobi sur wIca directement
        // Pour rester simple et exact : copie wIca dans un tableau, applique Gram-Schmidt SVD via
        // la méthode des valeurs singulières (polar decomposition : R = M * (M^T M)^{-1/2})
        val m = Array(3) { i -> doubleArrayOf(wIca[i][0], wIca[i][1], wIca[i][2]) }

        // M^T * M (symétrique 3×3)
        fun mt(r: Int, c: Int) = m[0][r]*m[0][c] + m[1][r]*m[1][c] + m[2][r]*m[2][c]
        val mtm = doubleArrayOf(
            mt(0,0), mt(0,1), mt(0,2),
            mt(1,0), mt(1,1), mt(1,2),
            mt(2,0), mt(2,1), mt(2,2)
        )
        // Jacobi sur mtm pour obtenir eigenvalues/vectors → (M^T M)^{-1/2}
        val ev = doubleArrayOf(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
        for (it in 0 until 50) {
            var pr=0; var pc=1; var mx=abs(mtm[1])
            if (abs(mtm[2])>mx){mx=abs(mtm[2]);pr=0;pc=2}
            if (abs(mtm[5])>mx){pr=1;pc=2}
            if (abs(mtm[pr*3+pc])<1e-15) break
            val th=0.5*atan2(2.0*mtm[pr*3+pc], mtm[pc*3+pc]-mtm[pr*3+pr])
            val cc=cos(th); val ss=sin(th); val tau=ss/(1.0+cc)
            val hh=ss*mtm[pr*3+pc]
            mtm[pr*3+pr]-=hh; mtm[pc*3+pc]+=hh; mtm[pr*3+pc]=0.0; mtm[pc*3+pr]=0.0
            val r3=if(pr==0&&pc==1) 2 else if(pr==0) 1 else 0
            val t1=mtm[r3*3+pr]; val t2=mtm[r3*3+pc]
            mtm[r3*3+pr]=t1-ss*(t2+t1*tau); mtm[pr*3+r3]=mtm[r3*3+pr]
            mtm[r3*3+pc]=t2+ss*(t1-t2*tau); mtm[pc*3+r3]=mtm[r3*3+pc]
            for (k in 0..2) {
                val gv=ev[k*3+pr]; val hv=ev[k*3+pc]
                ev[k*3+pr]=gv-ss*(hv+gv*tau); ev[k*3+pc]=hv+ss*(gv-hv*tau)
            }
        }
        val d0=1.0/sqrt(maxOf(mtm[0],1e-10))
        val d1=1.0/sqrt(maxOf(mtm[4],1e-10))
        val d2=1.0/sqrt(maxOf(mtm[8],1e-10))
        // S = V * diag(d) * V^T = (M^T M)^{-1/2}
        fun sv(r: Int, c: Int) = ev[0*3+r]*d0*ev[0*3+c] + ev[1*3+r]*d1*ev[1*3+c] + ev[2*3+r]*d2*ev[2*3+c]
        // R = M * S  (polar decomposition)
        val rica = FloatArray(9)
        for (r in 0..2) for (c in 0..2) {
            var v = 0.0
            for (k in 0..2) v += m[r][k] * sv(k, c)
            rica[r*3+c] = v.toFloat()
        }

        // Correction det < 0 (identique à Python)
        val det = rica[0]*(rica[4]*rica[8]-rica[5]*rica[7]) -
                  rica[1]*(rica[3]*rica[8]-rica[5]*rica[6]) +
                  rica[2]*(rica[3]*rica[7]-rica[4]*rica[6])
        if (det < 0f) {
            rica[6] = -rica[6]; rica[7] = -rica[7]; rica[8] = -rica[8]
        }

        Log.d(TAG, "ICA R: [${rica[0]},${rica[1]},${rica[2]}] [${rica[3]},${rica[4]},${rica[5]}] [${rica[6]},${rica[7]},${rica[8]}] det=$det")
        return rica
    }

    // ── FIN ICA ───────────────────────────────────────────────────────────────

    private fun toggleRandomRotation() {
        if (isRandomRotating) stopRandomRotation() else startRandomRotation()
    }

    private fun startRandomRotation() {
        isRandomRotating = true
        btnRandom.setTextColor("#AAFF44".toColorInt())
        applyRandomRotation()
        val handler = android.os.Handler(mainLooper)
        randRotHandler = handler
        val runnable = object : Runnable {
            override fun run() {
                if (isRandomRotating) {
                    applyRandomRotation()
                    handler.postDelayed(this, 3000L)
                }
            }
        }
        randRotRunnable = runnable
        handler.postDelayed(runnable, 3000L)
    }

    private fun stopRandomRotation() {
        if (!isRandomRotating) return
        isRandomRotating = false
        randRotRunnable?.let { randRotHandler?.removeCallbacks(it) }
        randRotRunnable = null
        randRotHandler = null
        btnRandom.setTextColor(android.graphics.Color.WHITE)
    }

    private fun applyRandomRotation() {
        val q = randomOrthogonalMatrix()
        val ay = asin(q[6].coerceIn(-1f, 1f))
        val ax = atan2(-q[7], q[8])
        val az = atan2(-q[3], q[0])
        val angX = (ax * (180.0 / PI)).toFloat()
        val angY = (ay * (180.0 / PI)).toFloat()
        val angZ = (az * (180.0 / PI)).toFloat()
        seekBarRotX.progress = (angX + 180f).toInt().coerceIn(0, 360)
        seekBarRotY.progress = (angY + 180f).toInt().coerceIn(0, 360)
        seekBarRotZ.progress = (angZ + 180f).toInt().coerceIn(0, 360)
        labelRotX.text = getString(R.string.angle_degrees, angX.toInt())
        labelRotY.text = getString(R.string.angle_degrees, angY.toInt())
        labelRotZ.text = getString(R.string.angle_degrees, angZ.toInt())
        btnICA.setTextColor(android.graphics.Color.WHITE)
        glSurfaceView.queueEvent {
            renderer.setRotationAngles(angX, angY, angZ)
        }
        glSurfaceView.requestRender()
    }

    private fun randomOrthogonalMatrix(): FloatArray {
        val rand = java.util.Random()
        fun rg() = floatArrayOf(rand.nextGaussian().toFloat(), rand.nextGaussian().toFloat(), rand.nextGaussian().toFloat())
        fun dot(a: FloatArray, b: FloatArray) = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        fun sub(a: FloatArray, b: FloatArray) = floatArrayOf(a[0]-b[0], a[1]-b[1], a[2]-b[2])
        fun scale(s: Float, v: FloatArray) = floatArrayOf(s*v[0], s*v[1], s*v[2])
        fun normalize(v: FloatArray): FloatArray {
            val n = sqrt(dot(v, v)).coerceAtLeast(1e-10f)
            return floatArrayOf(v[0]/n, v[1]/n, v[2]/n)
        }
        val v1 = rg(); val v2 = rg(); val v3 = rg()
        val e1 = normalize(v1)
        val e2 = normalize(sub(v2, scale(dot(v2, e1), e1)))
        val e3 = normalize(sub(sub(v3, scale(dot(v3, e1), e1)), scale(dot(v3, e2), e2)))
        val det = e1[0]*(e2[1]*e3[2]-e2[2]*e3[1]) - e1[1]*(e2[0]*e3[2]-e2[2]*e3[0]) + e1[2]*(e2[0]*e3[1]-e2[1]*e3[0])
        val f1 = if (det < 0f) floatArrayOf(-e1[0], -e1[1], -e1[2]) else e1
        return floatArrayOf(f1[0], e2[0], e3[0], f1[1], e2[1], e3[1], f1[2], e2[2], e3[2])
    }

    private fun saveBitmapToGallery(bitmap: Bitmap, sourceUri: Uri?) {
        try {
            val timestamp = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.ROOT).format(java.util.Date())
            val filename = "RASCAL_${timestamp}_processed.jpg"
            
            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/Rascal")
            }
            
            val uri = contentResolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
            if (uri != null) {
                val outStream = contentResolver.openOutputStream(uri)
                outStream?.use { stream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, stream)
                }
                bitmap.recycle()
                if (sourceUri != null) copyExifData(sourceUri, uri)
                Toast.makeText(this, "Image saved to Pictures/Rascal", Toast.LENGTH_LONG).show()
                Log.d(TAG, "Processed image saved: $uri")
            } else {
                Toast.makeText(this, "Failed to create file", Toast.LENGTH_SHORT).show()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bitmap to gallery", e)
            Toast.makeText(this, "Error saving image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}
