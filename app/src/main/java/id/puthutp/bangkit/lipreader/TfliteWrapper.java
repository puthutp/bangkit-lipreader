package id.puthutp.bangkit.lipreader;

import android.app.Activity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class TfliteWrapper {
    private Activity activity;
    private Interpreter tflite;

    private static final Logger LOGGER = new Logger();

    public TfliteWrapper(Activity activity)
    {
        this.activity = activity;
        Initialize();
    }

    public void Initialize()
    {
        try{
            MappedByteBuffer tfliteModel
                    = FileUtil.loadMappedFile(activity,
                    "mobilenet_v1_1.0_224_quant.tflite");
            tflite = new Interpreter(tfliteModel);
        } catch (IOException e){
            LOGGER.e("tfliteSupport", "Error reading model", e);
        }
    }

    public void Run(TensorImage tImage)
    {
        TensorBuffer probabilityBuffer =
                TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);

        if(null != tflite) {
            tflite.run(tImage.getBuffer(), probabilityBuffer.getBuffer());
        }

        float[] results =probabilityBuffer.getFloatArray();
        String output = "";
        for (int i = 0; i < results.length; i++)
        {
            output += results[i] + " ";
        }
        LOGGER.d("output" + output);
    }
}
