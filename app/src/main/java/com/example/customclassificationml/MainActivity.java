package com.example.customclassificationml;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
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
import com.otaliastudios.cameraview.CameraListener;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.PictureResult;
import com.otaliastudios.cameraview.controls.Mode;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Calendar;

import dmax.dialog.SpotsDialog;
import im.delight.android.location.SimpleLocation;

public class MainActivity extends AppCompatActivity {
    public Button startBtn;
    private TextView textView;
    private CameraView cameraView;
    private ImageView imageView;
    private ProgressBar progressBar1;
    private ProgressBar progressBar2;
    private AlertDialog alertDialog;
    private final int inputImageSize = 299;
    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private final int classifications = 10;
    private final int imageChannels = 3;
    private final int batchSize = 1;
    private JSONObject labels;
    private String outputResult;
    private SimpleLocation location;
    private String TAG = "CustomModel";

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        startBtn = findViewById(R.id.startBtn);
        textView = findViewById(R.id.textView);
        imageView = findViewById(R.id.imageView);
        progressBar1 = findViewById(R.id.progressBar1);
        progressBar2 = findViewById(R.id.progressBar2);
        cameraView = findViewById(R.id.cameraView);
        cameraView.setMode(Mode.PICTURE);
        cameraView.setLifecycleOwner(this);
        try {
            location = new SimpleLocation(MainActivity.this);
            Log.d(TAG, "onCreate: Location services started successfully !!");
            initializeModel();
        } catch (SecurityException e) {
            e.printStackTrace();
            Toast.makeText(this, "Location permissions are not provided :(", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "onCreate: Failed to start location services !!");
        } catch (FirebaseMLException e) {
            Log.d(TAG, "onCreate: Failed to register CustomModel and Interpreter !!");
        }

        alertDialog = new SpotsDialog.Builder()
                .setContext(MainActivity.this)
                .setMessage("Processing...")
                .setCancelable(false)
                .build();
        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick: Requested to take a picture");
                cameraView.takePicture();
            }
        });
        cameraView.addCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(@NonNull PictureResult result) {
                Log.d(TAG, "onPictureTaken: Picture taken");
                super.onPictureTaken(result);
                new SavePicture().execute(result);
                new RunModel().execute(result);
            }
        });
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Start Gallery or History activity
                Toast.makeText(MainActivity.this, "IMGBTN", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void initializeModel() throws FirebaseMLException {
        getLabels();
        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath("crop_model.tflite")
                .build();
        if (localModel == null)
            throw new FirebaseMLException("Failed to load model. Either it doesn't exist or corrupted !!", 0);
        FirebaseModelInterpreterOptions options =
                new FirebaseModelInterpreterOptions.Builder(localModel).build();
        interpreter = FirebaseModelInterpreter.getInstance(options);
        inputOutputOptions =
                new FirebaseModelInputOutputOptions.Builder()
                        .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{batchSize, inputImageSize, inputImageSize, imageChannels})
                        .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{batchSize, classifications})
                        .build();
    }

    private void getLabels() {
        String line = "";
        String results = "";
        try {
            InputStream inputStream = this.getAssets().open("crop_labels.json");
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

    private void normalizeInput(Bitmap bitmap) {
        Log.d(TAG, "normalizeInput: Bitmap normalization started");
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
        Log.d(TAG, "processClassification: Bitmap normalization completed");
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
                                outputResult = "Confidence: \n";
                                float[][] output = result.getOutput(0);
                                for (int i = 1; i < classifications; i++) {
                                    outputResult += output[0][i] + "\n";
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
                                progressBar2.setVisibility(View.GONE);
                                textView.setVisibility(View.VISIBLE);
                                alertDialog.dismiss();
                            }
                        }
                );
    }

    private class SavePicture extends AsyncTask<PictureResult, Void, Void> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            if (!location.hasLocationEnabled()) {
                Toast.makeText(MainActivity.this, "Turn on the Location !!", Toast.LENGTH_SHORT).show();
                SimpleLocation.openSettings(MainActivity.this);
            }
        }

        @Override
        protected Void doInBackground(PictureResult... results) {
            String imageName = "PIC_" + Calendar.getInstance().getTimeInMillis() + ".jpg";
            String locationFileData = "ID: " + imageName +
                    "\nLatLang: " + location.getLatitude() + "," + location.getLongitude() + "\n\n";
            File imageFile = new File(getExternalFilesDir(null), imageName);
            File locationFile = new File(getExternalFilesDir(null), "test_location.txt");
            try {
                FileOutputStream fileOutputStream = new FileOutputStream(imageFile);
                fileOutputStream.write(results[0].getData());
                fileOutputStream.close();

                OutputStream outputStream = new FileOutputStream(locationFile, true);
                OutputStreamWriter writer = new OutputStreamWriter(outputStream);
                writer.write(locationFileData);
                writer.flush();
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);

        }
    }

    private class RunModel extends AsyncTask<PictureResult, Bitmap, Void> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            alertDialog.show();
            textView.setVisibility(View.GONE);
            progressBar1.setVisibility(View.VISIBLE);
            progressBar2.setVisibility(View.VISIBLE);
        }

        @Override
        protected Void doInBackground(PictureResult... results) {
            Bitmap bitmap;
            Matrix matrix = new Matrix();
            matrix.preRotate(results[0].getRotation());
            bitmap = BitmapFactory.decodeByteArray(results[0].getData(), 0, results[0].getData().length);
            bitmap = Bitmap.createScaledBitmap(bitmap, inputImageSize, inputImageSize, true);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, inputImageSize, inputImageSize, matrix, true);
            publishProgress(bitmap);
            normalizeInput(bitmap);
            return null;
        }

        @Override
        protected void onProgressUpdate(Bitmap... values) {
            super.onProgressUpdate(values);
            imageView.setImageBitmap(values[0]);
            progressBar1.setVisibility(View.GONE);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        location.beginUpdates();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        location.endUpdates();
        if (alertDialog != null) {
            alertDialog.dismiss();
            alertDialog = null;
        }
    }
}

