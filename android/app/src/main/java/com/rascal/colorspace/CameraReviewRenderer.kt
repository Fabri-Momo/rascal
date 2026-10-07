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

class CameraReviewRenderer(private val context: Context) : GLSurfaceView.Renderer {

    companion object {
        private const val TAG = "CameraReviewRenderer"
    }

    private var program: Int = 0
    private var textureId: Int = 0
    private var hasTexture = false

    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var viewWidth: Int = 1
    private var viewHeight: Int = 1

    private var exposure: Float = 0f
    private var contrast: Float = 0f

    private var pendingBitmap: Bitmap? = null

    private lateinit var vertexBuffer: FloatBuffer
    private lateinit var texCoordBuffer: FloatBuffer

    private val texCoords = floatArrayOf(
        0f, 1f,
        1f, 1f,
        0f, 0f,
        1f, 0f
    )

    init {
        texCoordBuffer = ByteBuffer.allocateDirect(texCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(texCoords)
        texCoordBuffer.position(0)

        // Placeholder vertex buffer (updated in onSurfaceChanged)
        val verts = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        vertexBuffer = ByteBuffer.allocateDirect(verts.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(verts)
        vertexBuffer.position(0)
    }

    fun setExposure(value: Float) { exposure = value }
    fun setContrast(value: Float) { contrast = value }

    fun loadBitmap(bitmap: Bitmap) {
        pendingBitmap = bitmap
        imageWidth = bitmap.width
        imageHeight = bitmap.height
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        try {
            val vertSrc = ShaderUtils.loadShaderFromAssets(context, "vertex_shader.glsl")
            val fragSrc = ShaderUtils.loadShaderFromAssets(context, "camera_review_fragment.glsl")
            val vert = ShaderUtils.loadShader(GLES20.GL_VERTEX_SHADER, vertSrc)
            val frag = ShaderUtils.loadShader(GLES20.GL_FRAGMENT_SHADER, fragSrc)
            program = GLES20.glCreateProgram()
            GLES20.glAttachShader(program, vert)
            GLES20.glAttachShader(program, frag)
            GLES20.glLinkProgram(program)
            val status = IntArray(1)
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0)
            if (status[0] == 0) {
                Log.e(TAG, "Link error: ${GLES20.glGetProgramInfoLog(program)}")
                GLES20.glDeleteProgram(program); program = 0
            }
        } catch (e: Exception) {
            Log.e(TAG, "Shader error", e); program = 0
        }

        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        // Upload bitmap if already set before surface was ready
        pendingBitmap?.let { uploadTexture(it); pendingBitmap = null }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewWidth = width
        viewHeight = height
        updateVertexBuffer()
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        if (program == 0) return

        // Upload pending bitmap before checking hasTexture
        pendingBitmap?.let { bmp ->
            uploadTexture(bmp)
            pendingBitmap = null
        }

        if (!hasTexture) return

        GLES20.glUseProgram(program)

        val posHandle = GLES20.glGetAttribLocation(program, "a_position")
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)

        val tcHandle = GLES20.glGetAttribLocation(program, "a_texcoord")
        GLES20.glEnableVertexAttribArray(tcHandle)
        GLES20.glVertexAttribPointer(tcHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "u_texture"), 0)

        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_exposure"), exposure)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "u_contrast"), contrast)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(tcHandle)
    }

    private fun uploadTexture(bitmap: Bitmap) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
        hasTexture = GLES20.glGetError() == GLES20.GL_NO_ERROR
        imageWidth = bitmap.width
        imageHeight = bitmap.height
        updateVertexBuffer()
    }

    private fun updateVertexBuffer() {
        if (viewWidth <= 0 || viewHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) return
        val imgAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspect = viewWidth.toFloat() / viewHeight.toFloat()
        val scaleX = if (imgAspect > viewAspect) 1f else imgAspect / viewAspect
        val scaleY = if (imgAspect > viewAspect) viewAspect / imgAspect else 1f
        val verts = floatArrayOf(
            -scaleX, -scaleY,
             scaleX, -scaleY,
            -scaleX,  scaleY,
             scaleX,  scaleY
        )
        vertexBuffer.clear()
        vertexBuffer.put(verts)
        vertexBuffer.position(0)
    }
}
