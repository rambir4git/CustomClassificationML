package com.example.customclassificationml;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.otaliastudios.cameraview.BitmapCallback;
import com.otaliastudios.cameraview.CameraListener;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.PictureResult;
import com.otaliastudios.cameraview.controls.Mode;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Calendar;

import dmax.dialog.SpotsDialog;

public class MainActivity extends AppCompatActivity {
    private Button startBtn;
    private TextView textView;
    private CameraView cameraView;
    private ImageView imageView;
    private AlertDialog alertDialog;
    private final int inputImageSize = 299;
    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private final int classifications = 5;
    private final int imageChannels = 3;
    private final int batchSize = 1;
    private JSONObject labels;
    private float[] outputProbabilities;
    private String outputResult;
    private String TAG = "CustomModel: ";

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        startBtn = findViewById(R.id.startBtn);
        textView = findViewById(R.id.textView);
        imageView = findViewById(R.id.imageView);
        cameraView = findViewById(R.id.cameraView);
        cameraView.setMode(Mode.PICTURE);
        cameraView.setLifecycleOwner(this);
        alertDialog = new SpotsDialog.Builder()
                .setContext(this)
                .setMessage("Processing...")
                .setCancelable(false)
                .build();
        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick: Requested to take a picture at: " + Calendar.getInstance().getTime());
                cameraView.takePicture();
            }
        });
        cameraView.addCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(@NonNull PictureResult result) {
                Log.d(TAG, "onPictureTaken: Picture taken at: " + Calendar.getInstance().getTime());
                super.onPictureTaken(result);
                result.toBitmap(new BitmapCallback() {
                    @Override
                    public void onBitmapReady(@Nullable Bitmap bitmap) {
                        bitmap = Bitmap.createScaledBitmap(bitmap, inputImageSize, inputImageSize, true);
                        imageView.setImageBitmap(bitmap);
                        alertDialog.show();
                        normalizeInput(bitmap);
                    }
                });
            }
        });
        try {
            initializeModel();
            Log.d(TAG, "onCreate: CustomModel and Interpreter are successfully registered !!");
        } catch (FirebaseMLException e) {
            Log.d(TAG, "onCreate: Failed to register CustomModel and Interpreter !!");
        }
    }

    private void getLabels() {
        String line = "";
        String results = "";
        try {
            InputStream inputStream = this.getAssets().open("flower_labels.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            while (line != null) {
                results += line;
                line = reader.readLine();
            }
            labels = new JSONObject(results);
            Log.d(TAG, "getLabels: Successfully got all labels !!");
        } catch (IOException e) {
            Log.d(TAG, "getLabels: Failed to get labels due to IOError !!");
            e.printStackTrace();
        } catch (JSONException e) {
            Log.d(TAG, "getLabels: File is not in JSON format, labels are not created !!");
            e.printStackTrace();
        }
    }

    private void initializeModel() throws FirebaseMLException {
        getLabels();
        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath("flower_model_optimized.tflite")
                .build();
        FirebaseModelInterpreterOptions options =
                new FirebaseModelInterpreterOptions.Builder(localModel).build();
        interpreter = FirebaseModelInterpreter.getInstance(options);
        inputOutputOptions =
                new FirebaseModelInputOutputOptions.Builder()
                        .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{batchSize, inputImageSize, inputImageSize, imageChannels})
                        .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{batchSize, classifications})
                        .build();
    }

    private void normalizeInput(Bitmap bitmap) {
        Log.d(TAG, "normalizeInput: Bitmap normalization started at: " + Calendar.getInstance().getTime());
        int batchNum = 0; //single image index
        float[][][][] input = new float[batchSize][inputImageSize][inputImageSize][imageChannels];
        for (int x = 0; x < inputImageSize; x++) {
            for (int y = 0; y < inputImageSize; y++) {
                int pixel = bitmap.getPixel(x, y);
                // Normalize channel values to [0.0f, 1.0f]
                input[batchNum][x][y][0] = (Color.red(pixel)) / 255.0f;
                input[batchNum][x][y][1] = (Color.green(pixel)) / 255.0f;
                input[batchNum][x][y][2] = (Color.blue(pixel)) / 255.0f;
            }
        }
        Log.d(TAG, "processClassification: Bitmap normalization completed at: " + Calendar.getInstance().getTime());
        startInference(input);
    }

    private void startInference(float[][][][] input) {
        Log.d(TAG, "processClassification: Inferences started at: " + Calendar.getInstance().getTime());
        FirebaseModelInputs inputs = null;
        try {
            inputs = new FirebaseModelInputs.Builder()
                    .add(input)  // add() as many input arrays as your model requires
                    .build();

        } catch (FirebaseMLException e) {
            Log.d(TAG, "startInference: CustomModel rejected the input, maybe mismatch of DataType !!");
            e.printStackTrace();
        }
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                float[][] output = result.getOutput(0);
                                outputProbabilities = output[0];
                                outputResult = "Confidence\n";
                                for (int i = 0; i < classifications; i++) {
                                    outputResult += outputProbabilities[i] + "\n";
                                }
                                Log.d(TAG, "onSuccess: Inferences were successful !!");
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                outputResult = "Inference Failed :(\n";
                                Log.d(TAG, "onFailure: Inferences failed !!");
                            }
                        })
                .addOnCompleteListener(
                        new OnCompleteListener<FirebaseModelOutputs>() {
                            @Override
                            public void onComplete(@NonNull Task<FirebaseModelOutputs> task) {
                                Log.d(TAG, "onComplete: Inferences completed at: " + Calendar.getInstance().getTime());
                                textView.setText(outputResult);
                                alertDialog.dismiss();
                            }
                        }
                );
    }

}

