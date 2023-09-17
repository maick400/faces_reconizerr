package com.example.facesreconnizerr


import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.drawable.BitmapDrawable
import android.media.Image
import android.media.ThumbnailUtils
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.facesreconnizerr.ml.ModelUnquant
import com.google.android.gms.tasks.OnFailureListener
import com.google.android.gms.tasks.OnSuccessListener
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.DecimalFormat
import java.util.ArrayList
import java.util.Collections
import java.util.List

class MainActivity : AppCompatActivity(), OnSuccessListener<Text>, OnFailureListener {



    private var tamanio_imagen = 224
    private var permisosNoAprobados: ArrayList<String>? = null
    private lateinit var txtResults: TextView
    private lateinit var mImageView: ImageView
    private var mSelectedImage: Bitmap? = null
    private lateinit var btnCamara: Button
    private lateinit var btnGaleria: Button

    companion object {
        const val REQUEST_CAMERA = 111
        const val REQUEST_GALLERY = 222
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        txtResults = findViewById(R.id.txtresults)
        mImageView = findViewById(R.id.image_view)

        btnCamara = findViewById(R.id.btCamera)
        btnGaleria = findViewById(R.id.btGallery)

        val permisos_requeridos = ArrayList<String>()
        permisos_requeridos.add(Manifest.permission.CAMERA)
        permisos_requeridos.add(Manifest.permission.MANAGE_EXTERNAL_STORAGE)
        permisos_requeridos.add(Manifest.permission.READ_EXTERNAL_STORAGE)

        permisosNoAprobados = getPermisosNoAprobados(permisos_requeridos)
        requestPermissions(permisosNoAprobados?.toTypedArray(), 100)
    }

    fun abrirGaleria(view: View) {
        val i = Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(i, REQUEST_GALLERY)
    }

    fun abrirCamera(view: View) {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, REQUEST_CAMERA)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && null != data) {
            try {
                if (requestCode == REQUEST_CAMERA)
                    mSelectedImage = data.extras?.get("data") as Bitmap
                else
                    mSelectedImage = MediaStore.Images.Media.getBitmap(contentResolver, data.data)

                mImageView.setImageBitmap(mSelectedImage)

                // Llamada a función para reconocer el rostro apenas cargue la imagen seleccionada.
                val imagen = Bitmap.createScaledBitmap(mSelectedImage, tamanio_imagen, tamanio_imagen, false)
                Reconocer(imagen)
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        for (i in permissions.indices) {
            if (permissions[i] == Manifest.permission.CAMERA) {
                // btnCamara.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            } else if (permissions[i] == Manifest.permission.MANAGE_EXTERNAL_STORAGE ||
                permissions[i] == Manifest.permission.READ_EXTERNAL_STORAGE
            ) {
                // btnGaleria.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            }
        }
    }

    fun btn_Reconocer(view: View) {
        // Instanciar el modelo basado en el creado (TensorFlow Lite)
        val imagenBase = mImageView.drawable as BitmapDrawable

        // Definir un bitmap para enviarlo al modelo creado.
        var image = imagenBase.bitmap.copy(imagenBase.bitmap.config, false)
        image = Bitmap.createScaledBitmap(image, tamanio_imagen, tamanio_imagen, true)

        Reconocer(image)
    }

    fun Reconocer(imagen: Bitmap) {
        try {
            // Instanciar el modelo basado en el creado (TensorFlow Lite)
            val model = ModelUnquant.newInstance(applicationContext)
            val imagenBase = mImageView.drawable as BitmapDrawable

            // Definir un bitmap para enviarlo al modelo creado.
            var imagen = Bitmap.createScaledBitmap(imagen, tamanio_imagen, tamanio_imagen, true)

            // Establecer las dimensiones que tend




            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, tamanio_imagen, tamanio_imagen, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * tamanio_imagen * tamanio_imagen * 3)
            byteBuffer.order(ByteOrder.nativeOrder())

            val intValues = IntArray(tamanio_imagen * tamanio_imagen)
            imagen.getPixels(intValues, 0, imagen.width, 0, 0, imagen.width, imagen.height)

            var pixel = 0

            for (i in 0 until imagen.height) {
                for (j in 0 until imagen.width) {
                    val valInt = intValues[pixel++]
                    byteBuffer.putFloat(((valInt shr 16) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat(((valInt shr 8) and 0xFF) * (1f / 255f))
                    byteBuffer.putFloat((valInt and 0xFF) * (1f / 255f))
                }
            }

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.getOutputFeature0AsTensorBuffer()

            val confidences = outputFeature0.floatArray

            val classes = arrayOf("ELONK", "MARK", )

            var resultados = ""
            val df = DecimalFormat("0.00")
            for (i in classes.indices) {
                resultados += "${classes[i]}, ${df.format(confidences[i] * 100)}%\n"
            }

            txtResults.text = "Resultados\n$resultados"

            model.close()
        } catch (e: Exception) {
            txtResults.text = e.message
        }
    }

    fun getPermisosNoAprobados(listaPermisos: ArrayList<String>): ArrayList<String> {
        val list = ArrayList<String>()

        if (Build.VERSION.SDK_INT >= 23) {
            for (permiso in listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso)
                }
            }
        }

        return list
    }

    fun OCRfx(v: View) {
        val image = InputImage.fromBitmap(mSelectedImage, 0)
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        recognizer.process(image)
            .addOnSuccessListener(this)
            .addOnFailureListener(this)
    }

    override fun onFailure(e: Exception) {
        txtResults.text = "Error de proceso"
    }

    override fun onSuccess(text: Text) {
        val blocks = text.textBlocks
        var resultados = ""
        if (blocks.isEmpty()) {
            resultados = "_____________"
        } else {
            for (i in blocks.indices) {
                val lines = blocks[i].lines
                for (j in lines.indices) {
                    val elements = lines[j].elements
                    for (k in elements.indices) {
                        resultados = "$resultados${elements[k].text} "
                    }
                }
            }
            resultados += "\n"
        }
        txtResults.text = resultados
    }

    fun Rostrosfx(v: View) {
        val image = InputImage.fromBitmap(mSelectedImage, 0)
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)
        detector.process(image)
            .addOnSuccessListener(object : OnSuccessListener<List<Face>> {
                override fun onSuccess(faces: List<Face>) {
                    if (faces.isEmpty()) {
                        txtResults.text = "No Hay rostros"
                    } else {
                        txtResults.text = "Hay ${faces.size} rostro(s)"
                    }

                    val drawable = mImageView.drawable as BitmapDrawable
                    val bitmap = drawable.bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = Canvas(bitmap)
                    val paint = Paint()
                    paint.color = Color.RED
                    paint.textSize = 70f
                    paint.strokeWidth = 20f
                    paint.style = Paint.Style.STROKE

                    for (face in faces) {
                        canvas.drawRect(face.boundingBox, paint)
                    }

                    mImageView.setImageBitmap(bitmap)
                }
            })
            .addOnFailureListener(this)
    }
}
