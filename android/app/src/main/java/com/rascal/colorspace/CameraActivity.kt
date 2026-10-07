package com.rascal.colorspace

import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.location.Location
import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.tan

/**
 * Activité caméra CameraX — capture JPEG qualité maximale sans post-traitements IA.
 * 1. Viewfinder + bouton déclencheur (cercle blanc)
 * 2. Preview de la photo + boutons ✓ (confirmer) et ✗ (reprendre)
 */
class CameraActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "CameraActivity"
        const val EXTRA_PHOTO_URI = "photo_uri"
        const val EXTRA_GPS_LATITUDE  = "gps_latitude"
        const val EXTRA_GPS_LONGITUDE = "gps_longitude"
        const val EXTRA_GPS_ALTITUDE  = "gps_altitude"
        const val EXTRA_GPS_HAS_ALTITUDE = "gps_has_altitude"
    }

    private lateinit var imageCapture: ImageCapture
    private lateinit var previewView: PreviewView
    private lateinit var reviewGLView: GLSurfaceView
    private lateinit var reviewRenderer: CameraReviewRenderer
    private lateinit var viewfinderContainer: FrameLayout
    private lateinit var reviewContainer: LinearLayout

    private var lastPhotoFile: File? = null
    private var originalReviewBitmap: Bitmap? = null
    private var exposureValue: Float = 0f    // range -1..+1
    private var contrastValue: Float = 0f    // range -1..+1
    private var lastKnownLocation: Location? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Récupérer la localisation transmise par MainActivity
        if (intent.hasExtra(EXTRA_GPS_LATITUDE)) {
            lastKnownLocation = Location("intent").apply {
                latitude  = intent.getDoubleExtra(EXTRA_GPS_LATITUDE, 0.0)
                longitude = intent.getDoubleExtra(EXTRA_GPS_LONGITUDE, 0.0)
                if (intent.getBooleanExtra(EXTRA_GPS_HAS_ALTITUDE, false)) {
                    altitude = intent.getDoubleExtra(EXTRA_GPS_ALTITUDE, 0.0)
                }
            }
        }

        // ── Root layout ──────────────────────────────────────────────────────
        val root = FrameLayout(this).apply {
            setBackgroundColor(android.graphics.Color.BLACK)
        }
        setContentView(root)

        // ── Viewfinder screen ────────────────────────────────────────────────
        viewfinderContainer = FrameLayout(this)
        root.addView(viewfinderContainer, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
        viewfinderContainer.addView(previewView, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        // Bouton déclencheur — cercle blanc en bas au centre
        val shutterBtn = View(this).apply {
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                setColor(android.graphics.Color.WHITE)
                setStroke((4 * resources.displayMetrics.density).toInt(),
                    android.graphics.Color.LTGRAY)
            }
        }
        val shutterSize = (80 * resources.displayMetrics.density).toInt()
        val shutterParams = FrameLayout.LayoutParams(shutterSize, shutterSize).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            bottomMargin = (48 * resources.displayMetrics.density).toInt()
        }
        shutterBtn.setOnClickListener { capturePhoto() }
        viewfinderContainer.addView(shutterBtn, shutterParams)

        // ── Review screen (caché par défaut) ────────────────────────────────
        reviewContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }
        root.addView(reviewContainer, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        ))

        reviewRenderer = CameraReviewRenderer(this)
        reviewGLView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setEGLConfigChooser(8, 8, 8, 8, 0, 0)
            holder.setFormat(android.graphics.PixelFormat.RGBA_8888)
            setRenderer(reviewRenderer)
            renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        }
        reviewContainer.addView(reviewGLView, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
        ))

        // ── Sliders Exposure / Contrast ──────────────────────────────────────
        val slidersPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xCC000000.toInt())
            val pad = (12 * resources.displayMetrics.density).toInt()
            setPadding(pad * 2, pad, pad * 2, pad)
        }
        reviewContainer.addView(slidersPanel, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        fun addSliderRow(label: String, onProgress: (Int) -> Unit): SeekBar {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                val vpad = (4 * resources.displayMetrics.density).toInt()
                setPadding(0, vpad, 0, vpad)
            }
            val lbl = TextView(this).apply {
                text = label
                setTextColor(Color.WHITE)
                textSize = 13f
                width = (80 * resources.displayMetrics.density).toInt()
            }
            val seekBar = SeekBar(this).apply {
                max = 200
                progress = 100
            }
            val valueLbl = TextView(this).apply {
                text = getString(R.string.slider_value, 0f)
                setTextColor(Color.WHITE)
                textSize = 13f
                width = (52 * resources.displayMetrics.density).toInt()
                gravity = android.view.Gravity.END or android.view.Gravity.CENTER_VERTICAL
            }
            row.addView(lbl)
            row.addView(seekBar, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(valueLbl)
            slidersPanel.addView(row)
            seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    val v = (p - 100) / 100f
                    valueLbl.text = getString(R.string.slider_value, v)
                    if (fromUser) onProgress(p)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
            return seekBar
        }

        addSliderRow("Exposure") { p ->
            exposureValue = (p - 100) / 100f
            reviewRenderer.setExposure(exposureValue)
            reviewGLView.requestRender()
        }
        addSliderRow("Contrast") { p ->
            contrastValue = (p - 100) / 100f
            reviewRenderer.setContrast(contrastValue)
            reviewGLView.requestRender()
        }

        // Boutons ✓ et ✗
        val btnBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setBackgroundColor(0xCC000000.toInt())
            setPadding(0, (16 * resources.displayMetrics.density).toInt(),
                0, (32 * resources.displayMetrics.density).toInt())
        }
        reviewContainer.addView(btnBar, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ))

        val btnSize = (72 * resources.displayMetrics.density).toInt()
        val btnMargin = (32 * resources.displayMetrics.density).toInt()

        // ✗ Annuler — reprendre la photo
        val btnCancel = makeRoundButton("✗", android.graphics.Color.RED)
        val cancelParams = LinearLayout.LayoutParams(btnSize, btnSize).apply {
            marginEnd = btnMargin
        }
        btnCancel.setOnClickListener { showViewfinder() }
        btnBar.addView(btnCancel, cancelParams)

        // ✓ Confirmer — utiliser la photo
        val btnConfirm = makeRoundButton("✓", 0xFF4CAF50.toInt())
        val confirmParams = LinearLayout.LayoutParams(btnSize, btnSize).apply {
            marginStart = btnMargin
        }
        btnConfirm.setOnClickListener { confirmPhoto() }
        btnBar.addView(btnConfirm, confirmParams)

        startCamera()
    }

    private fun makeRoundButton(label: String, color: Int): android.widget.TextView {
        return android.widget.TextView(this).apply {
            text = label
            textSize = 28f
            gravity = Gravity.CENTER
            setTextColor(android.graphics.Color.WHITE)
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.OVAL
                setColor(color)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .build()

            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(100)
                .setResolutionSelector(resolutionSelector)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
                setResult(RESULT_CANCELED)
                finish()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun capturePhoto() {
        val photoFile = File(
            getExternalFilesDir(null),
            "RASCAL_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.jpg"
        )

        imageCapture.takePicture(
            ImageCapture.OutputFileOptions.Builder(photoFile).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    lastPhotoFile = photoFile
                    showReview(photoFile)
                }
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Capture failed", exception)
                }
            }
        )
    }

    private fun showReview(file: File) {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, opts)
        val screen = resources.displayMetrics.let { maxOf(it.widthPixels, it.heightPixels) }
        var s = 1
        while (maxOf(opts.outWidth, opts.outHeight) / (s * 2) > screen) s *= 2
        var bmp = BitmapFactory.decodeFile(file.absolutePath,
            BitmapFactory.Options().apply { inSampleSize = s })

        // Corriger l'orientation EXIF
        val exif = androidx.exifinterface.media.ExifInterface(file.absolutePath)
        val rotation = when (exif.getAttributeInt(
            androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
            androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
        )) {
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90  -> 90f
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (rotation != 0f && bmp != null) {
            val m = android.graphics.Matrix().apply { postRotate(rotation) }
            val rotated = android.graphics.Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            bmp.recycle()
            bmp = rotated
        }

        originalReviewBitmap?.recycle()
        originalReviewBitmap = bmp
        exposureValue = 0f
        contrastValue = 0f
        reviewRenderer.setExposure(0f)
        reviewRenderer.setContrast(0f)
        reviewGLView.queueEvent {
            reviewRenderer.loadBitmap(bmp!!)
            reviewGLView.requestRender()
        }
        viewfinderContainer.visibility = View.GONE
        reviewContainer.visibility = View.VISIBLE
    }

    private fun showViewfinder() {
        lastPhotoFile?.delete()
        lastPhotoFile = null
        originalReviewBitmap?.recycle()
        originalReviewBitmap = null
        reviewContainer.visibility = View.GONE
        viewfinderContainer.visibility = View.VISIBLE
    }

    private fun confirmPhoto() {
        val file = lastPhotoFile ?: return

        // Si expo/contraste modifiés, réécrire le JPEG avec le bitmap ajusté (pleine résolution)
        if (exposureValue != 0f || contrastValue != 0f) {
            applyAdjustmentsToFile(file)
        }

        // Écrire le GPS dans le fichier temporaire AVANT la copie en galerie
        writeGpsToFile(file)

        val galleryUri = saveToGallery(file)
        val resultUri = galleryUri ?: file.toUri()
        Log.d(TAG, "Photo confirmed: $resultUri (gallery=$galleryUri)")
        setResult(RESULT_OK, Intent().apply {
            putExtra(EXTRA_PHOTO_URI, resultUri.toString())
        })
        finish()
    }

    private fun writeGpsToFile(file: File) {
        val loc = lastKnownLocation ?: return
        try {
            val exif = androidx.exifinterface.media.ExifInterface(file.absolutePath)
            fun toDms(deg: Double): String {
                val d = deg.toInt()
                val mFull = (deg - d) * 60.0
                val m = mFull.toInt()
                val s = (mFull - m) * 60.0
                return "$d/1,$m/1,${(s * 1000).toInt()}/1000"
            }
            exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE,
                toDms(kotlin.math.abs(loc.latitude)))
            exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LATITUDE_REF,
                if (loc.latitude >= 0) "N" else "S")
            exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE,
                toDms(kotlin.math.abs(loc.longitude)))
            exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_LONGITUDE_REF,
                if (loc.longitude >= 0) "E" else "W")
            if (loc.hasAltitude()) {
                val alt = kotlin.math.abs(loc.altitude)
                exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE,
                    "${(alt * 100).toInt()}/100")
                exif.setAttribute(androidx.exifinterface.media.ExifInterface.TAG_GPS_ALTITUDE_REF,
                    if (loc.altitude >= 0) "0" else "1")
            }
            exif.saveAttributes()
            Log.d(TAG, "GPS written to temp file: ${loc.latitude}, ${loc.longitude}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write GPS to temp file", e)
        }
    }

    private fun applyAdjustmentsToFile(file: File) {
        try {
            var bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return

            // Conserver l'orientation EXIF avant modification
            val exif = androidx.exifinterface.media.ExifInterface(file.absolutePath)
            val rotation = when (exif.getAttributeInt(
                androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
            )) {
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90  -> 90f
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            if (rotation != 0f) {
                val m = android.graphics.Matrix().apply { postRotate(rotation) }
                val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                bmp.recycle(); bmp = rotated
            }

            val w = bmp.width; val h = bmp.height
            val pixels = IntArray(w * h)
            bmp.getPixels(pixels, 0, w, 0, 0, w, h)
            bmp.recycle()

            val expMul = 2.0.pow(exposureValue.toDouble()).toFloat()
            val cFactor = tan((contrastValue + 1f) * PI / 4.0).toFloat()

            for (i in pixels.indices) {
                val px = pixels[i]
                var r = ((px shr 16) and 0xFF) / 255f
                var g = ((px shr 8)  and 0xFF) / 255f
                var b = ( px         and 0xFF) / 255f
                r = ((r * expMul - 0.5f) * cFactor + 0.5f).coerceIn(0f, 1f)
                g = ((g * expMul - 0.5f) * cFactor + 0.5f).coerceIn(0f, 1f)
                b = ((b * expMul - 0.5f) * cFactor + 0.5f).coerceIn(0f, 1f)
                pixels[i] = (0xFF shl 24) or
                    ((r * 255f + 0.5f).toInt() shl 16) or
                    ((g * 255f + 0.5f).toInt() shl 8) or
                     (b * 255f + 0.5f).toInt()
            }

            val adjusted = Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
            java.io.FileOutputStream(file).use { out ->
                adjusted.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            adjusted.recycle()
            Log.d(TAG, "Adjustments applied to file: exp=$exposureValue contrast=$contrastValue")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply adjustments", e)
        }
    }

    private fun saveToGallery(file: File): android.net.Uri? {
        return try {
            val filename = file.name
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/Rascal")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { out ->
                        file.inputStream().use { it.copyTo(out) }
                    }
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    contentResolver.update(uri, values, null, null)
                    Log.d(TAG, "Photo saved to gallery: $uri")
                }
                uri
            } else {
                val dcim = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
                val dest = File(File(dcim, "Rascal").also { it.mkdirs() }, filename)
                file.copyTo(dest, overwrite = true)
                val destUri = dest.toUri()
                sendBroadcast(Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, destUri))
                Log.d(TAG, "Photo saved to gallery: ${dest.absolutePath}")
                destUri
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save to gallery", e)
            Toast.makeText(this, "Could not save to gallery", Toast.LENGTH_SHORT).show()
            null
        }
    }
}
