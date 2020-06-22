package id.puthutp.bangkit.lipreader;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.ImageView;

import com.esafirm.imagepicker.model.Image;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.classification.R;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.util.ArrayList;

public class ImagesHandler
{
    private Activity activity;
    private ArrayList<Bitmap> bitmaps;

    private static final Logger LOGGER = new Logger();

    private int modelWidth = 100;
    private int modelHeight = 100;
    private SimplerTfliteWrapper tfliteWrapper;

    public ImagesHandler(Activity activity)
    {
        this.activity = activity;
        bitmaps = new ArrayList<>();
        tfliteWrapper = new SimplerTfliteWrapper(activity);
    }

    public void Handle(ArrayList<Image> images)
    {
        bitmaps.clear();
        for (int i = 0; i < images.size(); i++)
        {
            String path = images.get(i).getPath();
            LOGGER.d("image " + path);
            Bitmap bitmap = BitmapFactory.decodeFile(path);
            bitmaps.add(bitmap);

            LOGGER.d("info " + bitmap.getWidth() + " " + bitmap.getHeight());
            //ProcessImage(bitmap);
        }
        Show(bitmaps.get(0));

        tfliteWrapper.Run(bitmaps);
    }

    private void Show(Bitmap imageBitmap)
    {
        ImageView imageView = (ImageView) activity.findViewById(R.id.preview);
        imageView.setImageBitmap(imageBitmap);
    }

    private void ProcessImage(Bitmap bitmap)
    {
        LOGGER.d(bitmap.getByteCount() + " ");
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(modelWidth, modelHeight, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        TensorImage tImage = new TensorImage(DataType.UINT8);
        tImage.load(bitmap);

        tImage = imageProcessor.process(tImage);

        LOGGER.d(tImage.getWidth() + " " + tImage.getHeight());
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        //int[] intValues = new int[];
        //bitmap.getPixels();
//        if (imgData == null) {
//            return;
//        }
//        imgData.rewind();
//
//
//        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//
//        long startTime = SystemClock.uptimeMillis();
//
//
//        Mat bufmat = new Mat(197,197,CV_8UC3);
//        Mat newmat = new Mat(197,197,CV_32FC3);
//
//
//        Utils.bitmapToMat(bitmap,bufmat);
//        Imgproc.cvtColor(bufmat,bufmat,Imgproc.COLOR_RGBA2RGB);
//
//        List<Mat> sp_im = new ArrayList<Mat>(3);
//
//
//        Core.split(bufmat,sp_im);
//
//        sp_im.get(0).convertTo(sp_im.get(0),CV_32F,1.0/255/0);
//        sp_im.get(1).convertTo(sp_im.get(1),CV_32F,1.0/255.0);
//        sp_im.get(2).convertTo(sp_im.get(2),CV_32F,1.0/255.0);
//
//        Core.merge(sp_im,newmat);
//
//
//
//        //bufmat.convertTo(newmat,CV_32FC3,1.0/255.0);
//        float buf[] = new float[197*197*3];
//
//
//        newmat.get(0,0,buf);
//
//        //imgData.wrap(buf).order(ByteOrder.nativeOrder()).getFloat();
//        imgData.order(ByteOrder.nativeOrder()).asFloatBuffer().put(buf);
//
//
//        long endTime = SystemClock.uptimeMillis();
//        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }
}
