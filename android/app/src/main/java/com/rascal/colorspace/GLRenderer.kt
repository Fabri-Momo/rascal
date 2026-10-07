package com.rascal.colorspace

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class GLRenderer(private val context: Context) : GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "GLRenderer"
    }

    private var program: Int = 0
    private var vertexBuffer: FloatBuffer
    private var texCoordBuffer: FloatBuffer
    
    private var textureId: Int = 0
    private var hasTexture = false
    private var currentBitmap: Bitmap? = null   // Bitmap actuellement affiché (original ou ROI)
    private var originalBitmap: Bitmap? = null  // Bitmap original pour restauration après ROI
    
    @Volatile private var imageWidth: Int = 1
    @Volatile private var imageHeight: Int = 1
    @Volatile private var viewWidth: Int = 1
    @Volatile private var viewHeight: Int = 1
    
    private var muLin = FloatArray(3) { 0.5f }
    private var wLin = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
    private var wInvLin = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
    // Matrices ZCA initiales — sauvegardées au chargement, JAMAIS modifiées (pour le reset)
    private var muLinInit = FloatArray(3) { 0.5f }
    private var wLinInit = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
    private var wInvLinInit = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)
    
    private var rotX: Float = 0f
    private var rotY: Float = 0f
    private var rotZ: Float = 0f
    // Matrice de rotation courante (column-major pour GLSL). null = utiliser Euler.
    private var rotationMatrixOverride: FloatArray? = null
    private var pushFactor: Float = 1f
    private var varRestore: Float = 1f
    @Volatile private var zoomScale: Float = 1f
    @Volatile private var zoomOffsetX: Float = 0f
    @Volatile private var zoomOffsetY: Float = 0f
    private var channelMode: Int = 0  // 0=COLOR, 1=RED, 2=GREEN, 3=BLUE
    private var splitActive: Boolean = false
    private var splitPos: Float = 0.5f
    
    private val vertices = floatArrayOf(
        -1f, -1f,
        1f, -1f,
        -1f, 1f,
        1f, 1f
    )
    
    private val texCoords = floatArrayOf(
        0f, 1f,
        1f, 1f,
        0f, 0f,
        1f, 0f
    )

    init {
        vertexBuffer = ByteBuffer.allocateDirect(vertices.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(vertices)
        vertexBuffer.position(0)

        texCoordBuffer = ByteBuffer.allocateDirect(texCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(texCoords)
        texCoordBuffer.position(0)
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        Log.d(TAG, "onSurfaceCreated")
        // Use a dark gray background to distinguish from black images
        GLES20.glClearColor(0.2f, 0.2f, 0.2f, 1f)
        
        try {
            val vertexShader = ShaderUtils.loadShader(
                GLES20.GL_VERTEX_SHADER,
                ShaderUtils.loadShaderFromAssets(context, "vertex_shader.glsl")
            )
            val fragmentShader = ShaderUtils.loadShader(
                GLES20.GL_FRAGMENT_SHADER,
                ShaderUtils.loadShaderFromAssets(context, "fragment_shader.glsl")
            )
            
            program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vertexShader)
            GLES20.glAttachShader(program, fragmentShader)
            GLES20.glLinkProgram(program)
            
            val linkStatus = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0)
            if (linkStatus[0] == 0) {
                val error = GLES20.glGetProgramInfoLog(program)
                Log.e(TAG, "Error linking program: $error")
                GLES20.glDeleteProgram(program)
                program = 0
            } else {
                Log.d(TAG, "Program linked successfully: $program")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Shader compilation failed", e)
            program = 0
        }
        
        textureId = createTexture()
        hasTexture = false
        Log.d(TAG, "New texture created with ID: $textureId")
        
        // Re-upload bitmap if we have one (handles context loss)
        currentBitmap?.let {
            Log.d(TAG, "Re-uploading current bitmap after context loss...")
            uploadBitmapToTexture(it)
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewWidth = width
        viewHeight = height
        updateVertexBuffer()
        Log.d(TAG, "onSurfaceChanged: ${width}x${height}")
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        
        if (program == 0) return
        if (!hasTexture) return
        
        GLES20.glUseProgram(program)
        
        val positionHandle = GLES20.glGetAttribLocation(program, "a_position")
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)
        
        val texCoordHandle = GLES20.glGetAttribLocation(program, "a_texcoord")
        GLES20.glEnableVertexAttribArray(texCoordHandle)
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer)
        
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        val textureHandle = GLES20.glGetUniformLocation(program, "u_texture")
        GLES20.glUniform1i(textureHandle, 0)
        
        GLES20.glUniform3fv(GLES20.glGetUniformLocation(program, "u_mu_lin"), 1, muLin, 0)
        GLES20.glUniformMatrix3fv(GLES20.glGetUniformLocation(program, "u_W_lin"), 1, false, wLin, 0)
        GLES20.glUniformMatrix3fv(GLES20.glGetUniformLocation(program, "u_Winv_lin"), 1, false, wInvLin, 0)
        
        val rotationMatrix = rotationMatrixOverride ?: computeRotationMatrix(rotX, rotY, rotZ)
        GLES20.glUniformMatrix3fv(GLES20.glGetUniformLocation(program, "u_rotation"), 1, false, rotationMatrix, 0)
        
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_push_factor"), pushFactor)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_var_restore"), varRestore)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "u_channel_mode"), channelMode)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "u_split_active"), if (splitActive) 1 else 0)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_split_pos"), splitPos)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_split_line_w"), 0.003f)
        
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(texCoordHandle)
    }

    private fun createTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        val id = textures[0]
        
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        
        return id
    }

    fun loadBitmapWithPrecomputedZCA(bitmap: Bitmap, mu: FloatArray, w: FloatArray, wInv: FloatArray) {
        Log.d(TAG, "loadBitmapWithPrecomputedZCA: ${bitmap.width}x${bitmap.height}")
        originalBitmap?.recycle()
        val config = bitmap.config ?: Bitmap.Config.ARGB_8888
        originalBitmap = bitmap.copy(config, false)
        currentBitmap = originalBitmap
        imageWidth = bitmap.width
        imageHeight = bitmap.height

        muLin       = mu.copyOf()
        wLin        = w.copyOf()
        wInvLin     = wInv.copyOf()
        muLinInit   = mu.copyOf()
        wLinInit    = w.copyOf()
        wInvLinInit = wInv.copyOf()
        uploadBitmapToTexture(bitmap)
        updateVertexBuffer()
    }

    fun restoreOriginalBitmap() {
        val orig = originalBitmap ?: return
        // Restaurer les matrices initiales (jamais modifiées depuis le chargement)
        muLin       = muLinInit.copyOf()
        wLin        = wLinInit.copyOf()
        wInvLin     = wInvLinInit.copyOf()
        // currentBitmap repointe sur l'original
        currentBitmap?.let { if (it !== originalBitmap) it.recycle() }
        currentBitmap = orig
        uploadBitmapToTexture(orig)
        Log.d(TAG, "Original bitmap + initial ZCA restored")
    }

    private fun updateVertexBuffer() {
        if (viewWidth <= 0 || viewHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
            return
        }
        
        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()
        
        var scaleX = 1f
        var scaleY = 1f
        
        if (imageAspect > viewAspect) {
            // Image is wider than view - fit to width
            scaleY = viewAspect / imageAspect
        } else {
            // Image is taller than view - fit to height
            scaleX = imageAspect / viewAspect
        }
        
        // Apply zoom scale
        scaleX *= zoomScale
        scaleY *= zoomScale
        
        // Apply zoom offset to center zoom on focal point
        val adjustedVertices = floatArrayOf(
            -scaleX + zoomOffsetX, -scaleY + zoomOffsetY,
            scaleX + zoomOffsetX, -scaleY + zoomOffsetY,
            -scaleX + zoomOffsetX, scaleY + zoomOffsetY,
            scaleX + zoomOffsetX, scaleY + zoomOffsetY
        )
        
        vertexBuffer.clear()
        vertexBuffer.put(adjustedVertices)
        vertexBuffer.position(0)
        
        Log.d(TAG, "Aspect ratio updated: image=${imageAspect}, view=${viewAspect}, scale=($scaleX, $scaleY), zoom=$zoomScale, offset=($zoomOffsetX, $zoomOffsetY)")
    }

    private fun uploadBitmapToTexture(bitmap: Bitmap) {
        if (textureId == 0) {
            Log.e(TAG, "Cannot upload bitmap: textureId is 0")
            return
        }

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)

        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            Log.e(TAG, "OpenGL error after texImage2D: $error")
            hasTexture = false
        } else {
            Log.d(TAG, "Bitmap uploaded to texture $textureId")
            hasTexture = true
        }
    }

    private fun computeRotationMatrix(rx: Float, ry: Float, rz: Float): FloatArray {
        val cx = cos(rx); val sx = sin(rx)
        val cy = cos(ry); val sy = sin(ry)
        val cz = cos(rz); val sz = sin(rz)
        // R = Rz * Ry * Rx écrit row-major. Uniform envoyé avec transpose=false →
        // GLSL voit R^T : convention identique au web (uniformMatrix3fv(false, R_rowmajor))
        // et à Rascal.py (vispy uploade row-major → GLSL voit R^T).
        return floatArrayOf(
            cy*cz,   sx*sy*cz - cx*sz,   cx*sy*cz + sx*sz,
            cy*sz,   sx*sy*sz + cx*cz,   cx*sy*sz - sx*cz,
            -sy,     sx*cy,              cx*cy
        )
    }

    fun setRotationAngles(x: Float, y: Float, z: Float) {
        rotX = (x * (PI / 180.0)).toFloat()
        rotY = (y * (PI / 180.0)).toFloat()
        rotZ = (z * (PI / 180.0)).toFloat()
        rotationMatrixOverride = null
    }

    fun setRotationMatrix(rowMajor: FloatArray) {
        // rowMajor est R (row-major). On transpose pour envoyer en column-major,
        // comme computeRotationMatrix. Le shader applique R via mat * vec.
        rotationMatrixOverride = floatArrayOf(
            rowMajor[0], rowMajor[3], rowMajor[6],
            rowMajor[1], rowMajor[4], rowMajor[7],
            rowMajor[2], rowMajor[5], rowMajor[8]
        )
    }

    fun setRotationMatrixColumnMajor(colMajor: FloatArray) {
        // colMajor est déjà dans le même layout que computeRotationMatrix.
        // Stockage direct sans transposition → pas de saut quand les sliders prennent le relais.
        rotationMatrixOverride = colMajor.copyOf()
    }

    fun clearICABase() {
        rotationMatrixOverride = null
    }

    /**
     * Retourne le rectangle occupé par l'image en coordonnées pixels écran (top-left origin).
     * Retourne null si les dimensions ne sont pas encore connues.
     */
    fun getImageRectScreen(): android.graphics.RectF? {
        if (viewWidth <= 0 || viewHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) return null
        val imageAspect = imageWidth.toFloat() / imageHeight
        val viewAspect  = viewWidth.toFloat()  / viewHeight
        val baseScaleX = if (imageAspect > viewAspect) 1f else imageAspect / viewAspect
        val baseScaleY = if (imageAspect > viewAspect) viewAspect / imageAspect else 1f
        val sx = baseScaleX * zoomScale
        val sy = baseScaleY * zoomScale
        // NDC → pixel : px = (ndc + 1) / 2 * viewWidth,  py = (1 - ndc) / 2 * viewHeight
        val ndcLeft   = -sx + zoomOffsetX;  val ndcRight = sx + zoomOffsetX
        val ndcTop    =  sy + zoomOffsetY;  val ndcBot   = -sy + zoomOffsetY
        return android.graphics.RectF(
            (ndcLeft  + 1f) / 2f * viewWidth,
            (1f - ndcTop)   / 2f * viewHeight,
            (ndcRight + 1f) / 2f * viewWidth,
            (1f - ndcBot)   / 2f * viewHeight
        )
    }

    fun setPushFactor(factor: Float) {
        pushFactor = factor
    }

    fun setVarRestore(value: Float) {
        varRestore = value
    }

    fun updateZoomWithFocalPoint(scaleFactor: Float, focalX: Float, focalY: Float) {
        val oldScale = zoomScale
        zoomScale *= scaleFactor
        zoomScale = zoomScale.coerceIn(1f, 10f)

        val actualFactor = zoomScale / oldScale

        // focalX/focalY are in NDC screen space (-1..1).
        // zoomOffsetX/Y is a translation applied to the quad in NDC space.
        // To keep the focal point fixed: offset = focal - (focal - offset) * factor
        if (zoomScale <= 1f) {
            zoomScale = 1f
            zoomOffsetX = 0f
            zoomOffsetY = 0f
        } else {
            zoomOffsetX = focalX - (focalX - zoomOffsetX) * actualFactor
            zoomOffsetY = focalY - (focalY - zoomOffsetY) * actualFactor
        }

        updateVertexBuffer()
    }

    fun resetZoom() {
        zoomScale = 1f
        zoomOffsetX = 0f
        zoomOffsetY = 0f
        updateVertexBuffer()
    }

    fun panImage(ndcDx: Float, ndcDy: Float) {
        if (zoomScale <= 1f) return

        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspect  = viewWidth.toFloat()  / viewHeight.toFloat()

        // Base half-extents of the quad (at zoom=1)
        val baseScaleX = if (imageAspect > viewAspect) 1f else imageAspect / viewAspect
        val baseScaleY = if (imageAspect > viewAspect) viewAspect / imageAspect else 1f

        // Max offset = how much the zoomed quad overflows the NDC [-1,1] screen
        val maxOffsetX = baseScaleX * (zoomScale - 1f)
        val maxOffsetY = baseScaleY * (zoomScale - 1f)

        zoomOffsetX = (zoomOffsetX + ndcDx).coerceIn(-maxOffsetX, maxOffsetX)
        zoomOffsetY = (zoomOffsetY + ndcDy).coerceIn(-maxOffsetY, maxOffsetY)

        updateVertexBuffer()
    }

    fun setZoomScale(scale: Float) {
        zoomScale = scale
        updateVertexBuffer()
    }

    fun getZoomScale(): Float = zoomScale

    fun setChannelMode(mode: Int) {
        channelMode = mode
    }

    fun setSplitActive(active: Boolean) {
        splitActive = active
    }

    fun isSplitActive(): Boolean = splitActive

    fun setSplitPos(pos: Float) {
        splitPos = pos.coerceIn(0f, 1f)
    }

    fun getSplitPos(): Float = splitPos

    fun resetSplit() {
        splitActive = false
        splitPos = 0.5f
    }

    fun getRotationMatrixOverride(): FloatArray? = rotationMatrixOverride?.copyOf()

    fun getCenterSplitPos(): Float {
        if (zoomScale == 1f && zoomOffsetX == 0f) {
            return 0.5f // Pas de zoom, centre normal
        }
        
        // Le centre de l'écran en NDC est (0, 0)
        // On doit trouver quelle coordonnée de texture correspond à ce point
        
        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()
        
        val baseScaleX = if (imageAspect > viewAspect) 1f else imageAspect / viewAspect
        val baseScaleY = if (imageAspect > viewAspect) viewAspect / imageAspect else 1f
        
        val scaleX = baseScaleX * zoomScale
        val scaleY = baseScaleY * zoomScale
        
        // Transformation inverse: NDC → coordonnées de texture normalisées
        // Le centre NDC (0, 0) correspond à:
        // texX = (0 - zoomOffsetX) / (2 * scaleX) + 0.5
        val centerTexX = (0f - zoomOffsetX) / (2f * scaleX) + 0.5f
        
        return centerTexX.coerceIn(0f, 1f)
    }

    fun getCurrentZCAMatrices(): Triple<FloatArray, FloatArray, FloatArray> {
        return Triple(muLin.copyOf(), wLin.copyOf(), wInvLin.copyOf())
    }

    fun getCurrentBitmap(): Bitmap? {
        val bmp = currentBitmap ?: return null
        return bmp.copy(bmp.config ?: Bitmap.Config.ARGB_8888, false)
    }

    class RenderState(
        val bitmap: Bitmap?,
        val mu: FloatArray,
        val w: FloatArray,
        val wInv: FloatArray,
        val rotation: FloatArray,
        val pushFactor: Float,
        val varRestore: Float,
        val channelMode: Int
    )

    fun getFullRenderState(): RenderState {
        val bmp = getCurrentBitmap()
        val rot = (rotationMatrixOverride ?: computeRotationMatrix(rotX, rotY, rotZ)).copyOf()
        return RenderState(bmp, muLin.copyOf(), wLin.copyOf(), wInvLin.copyOf(), rot, pushFactor, varRestore, channelMode)
    }

    fun restorePresetWithROI(bitmap: Bitmap, mu: FloatArray, w: FloatArray, wInv: FloatArray) {
        currentBitmap?.let { if (it !== originalBitmap) it.recycle() }
        val config = bitmap.config ?: Bitmap.Config.ARGB_8888
        currentBitmap = bitmap.copy(config, false)
        muLin = mu.copyOf()
        wLin = w.copyOf()
        wInvLin = wInv.copyOf()
        uploadBitmapToTexture(currentBitmap!!)
    }

    fun loadBitmapAsROIWithMatrices(bitmap: Bitmap, mu: FloatArray, w: FloatArray, wInv: FloatArray) {
        // Recycle old ROI if it's not the original
        currentBitmap?.let { if (it !== originalBitmap) it.recycle() }
        
        val config = bitmap.config ?: Bitmap.Config.ARGB_8888
        currentBitmap = bitmap.copy(config, false)
        
        muLin = mu.copyOf()
        wLin = w.copyOf()
        wInvLin = wInv.copyOf()
        
        uploadBitmapToTexture(currentBitmap!!)
    }

}
