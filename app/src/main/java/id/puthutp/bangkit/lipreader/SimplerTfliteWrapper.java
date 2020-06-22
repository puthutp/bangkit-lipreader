package id.puthutp.bangkit.lipreader;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Debug;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.classification.R;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.w3c.dom.Text;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class SimplerTfliteWrapper {
    private Activity activity;
    private Interpreter tflite;

    private static final Logger LOGGER = new Logger();

    private String[] labels = {"Begin", "Choose", "Connection", "Navigation", "Next", "Previous", "Start", "Stop", "Hello", "Web"};

//    private final TensorBuffer outputProbabilityBuffer;

    public SimplerTfliteWrapper(Activity activity)
    {
        this.activity = activity;
        Initialize();
    }

    public void Initialize()
    {
        try{
            Interpreter.Options options = new Interpreter.Options();
            options.setUseNNAPI(true);
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(activity,
                    "modelsimple.tflite");

            tflite = new Interpreter(tfliteModel, options);
        } catch (IOException e){
            LOGGER.e("tfliteSupport", "Error reading model", e);
        }

        int[] outputShape = tflite.getOutputTensor(0).shape();
        LOGGER.e("output shape " + outputShape.length);
        int[] inputShape = tflite.getInputTensor(0).shape();
        LOGGER.e("input shape " + inputShape.length);
        for (int i = 0; i < inputShape.length; i ++)
        {
            LOGGER.e("dim " + i + " " + inputShape[i]);
        }

        LOGGER.d("output count " + tflite.getOutputTensorCount() + " " + tflite.getInputTensorCount());
        for (int t = 0; t < tflite.getOutputTensorCount(); t++)
        {
            Tensor output = tflite.getOutputTensor(t);
            LOGGER.d("output " + output.name() + " " + output.shape().length);
        }
        for (int u = 0; u < tflite.getInputTensorCount(); u++)
        {
            Tensor input = tflite.getInputTensor(u);
            LOGGER.d("input " + input.name() + " " + input.shape().length);
        }

//        Tensor outputTensor = tflite.getOutputTensor(tflite.getOutputTensorCount() - 1);
//        LOGGER.d("output " + outputTensor.name() + " " + outputTensor.shape().length);
//        int[] outputShape = outputTensor.shape();
    }

    private MappedByteBuffer loadModelFile(String path) throws IOException {
        AssetManager assets = activity.getAssets();
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = assets.openFd(path);
        } catch (IOException e) {
            e.printStackTrace();
        }

        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void Run(ArrayList<Bitmap> bitmaps)
    {
        ByteBuffer byteBuffer = convertBitmapsToByteBuffer(bitmaps);

        float[][] output = new float[1][10];
        tflite.run(byteBuffer, output);
        LOGGER.e(""+ output);

        float[] result = output[0];
        float maxRes = 0;
        int maxIdx = 0;
        for (int i = 0; i < 10; i++)
        {
            LOGGER.d("res " + i + " " + result[i]);
            if (result[i] > maxRes)
            {
                maxRes = result[i];
                maxIdx = i;
            }
        }

        TextView resultView = (TextView) activity.findViewById(R.id.result);
        resultView.setText(labels[maxIdx]);
    }

    private ByteBuffer convertBitmapsToByteBuffer(ArrayList<Bitmap> bitmaps) {
        //byte size of float times total dimension of model input
        int modelInputSize = 4 * (1 * 22 * 100 * 100 * 1);

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        int sampleCount = bitmaps.size();
        for (int i = 0; i < 22; i++) { //up to 22
            if (i < sampleCount) {
                Bitmap original = bitmaps.get(i);
                Bitmap bitmap = Bitmap.createScaledBitmap(
                        original,
                        100,
                        100,
                        true
                );
//            LOGGER.d(original.getWidth() + " " + bitmap.getHeight());
//            LOGGER.d(bitmap.getWidth() + " " + bitmap.getHeight());
//            LOGGER.d("byte count " + bitmap.getByteCount());

//            //BEGIN SAMPLING TEST
//            int testPx = bitmap.getPixel(0, 0);
//            int r = (testPx >> 16 & 0xFF);
//            int g = (testPx >> 8 & 0xFF);
//            int b = (testPx & 0xFF);
//            float normalized = (r + g + b) / 3.0f / 255.0f;
//            LOGGER.d("pix " + testPx + " " + normalized + " " + Float.BYTES);
//            //END SAMPLING

                int[] pixels = new int[100 * 100];
                bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

                for (int pixelValue : pixels) {
                    int r = (pixelValue >> 16 & 0xFF);
                    int g = (pixelValue >> 8 & 0xFF);
                    int b = (pixelValue & 0xFF);

                    // Convert RGB to grayscale and normalize pixel value to [0..1].
                    float normalizedPixelValue = (r + g + b) / 3.0f / 255.0f;
                    byteBuffer.putFloat(normalizedPixelValue);
                }
            }
            else
            {
                byteBuffer.putFloat(0f);
            }
        }

        return byteBuffer;
    }
}
