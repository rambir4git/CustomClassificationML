package com.example.customclassificationml;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

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

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import dmax.dialog.SpotsDialog;

public class MainActivity extends AppCompatActivity {
    private Button startBtn;
    private TextView textView;
    private CameraView cameraView;
    private ImageView imageView;
    private AlertDialog alertDialog;
    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        startBtn = findViewById(R.id.startBtn);
        textView = findViewById(R.id.textView);
        imageView = findViewById(R.id.imageView);
        cameraView = findViewById(R.id.cameraView);
        cameraView.setMode(Mode.PICTURE);
        cameraView.setLifecycleOwner(this);
        initialize_model();
        alertDialog = new SpotsDialog.Builder()
                .setContext(this)
                .setMessage("Processing...")
                .setCancelable(false)
                .build();
        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.takePicture();
                alertDialog.show();
            }
        });
        cameraView.addCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(@NonNull PictureResult result) {
                super.onPictureTaken(result);
                result.toBitmap(result.getSize().getWidth(), result.getSize().getWidth(), new BitmapCallback() {
                    @Override
                    public void onBitmapReady(@Nullable Bitmap bitmap) {
                        processClassification(bitmap);
                    }
                });
            }
        });
        check();
    }

    private void check() {
        String results = null;
        try {
            InputStream inputStream = new BufferedInputStream(this.getAssets().open("reverse_labels.json"));
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            while (reader.readLine() != null) {
                results += reader.readLine();
            }
            JSONObject jsonObject = new JSONObject(results);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }

    }

    private void initialize_model() {
        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath("converted_model.tflite")
                .build();
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);
        } catch (FirebaseMLException e) {
            Toast.makeText(this, "Interpreter not instantiated", Toast.LENGTH_SHORT).show();
        }
        try {
            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 299, 299, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 38})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    private void processClassification(Bitmap bitmap) {
        int batchNum = 0;
        bitmap = Bitmap.createScaledBitmap(bitmap, 299, 299, true);
        float[][][][] input = new float[1][299][299][3];
        for (int x = 0; x < 299; x++) {
            for (int y = 0; y < 299; y++) {
                int pixel = bitmap.getPixel(x, y);
                // Normalize channel values to [0.0f, 1.0f]
                input[batchNum][x][y][0] = (Color.red(pixel)) / 255.0f;
                input[batchNum][x][y][1] = (Color.green(pixel)) / 255.0f;
                input[batchNum][x][y][2] = (Color.blue(pixel)) / 255.0f;
            }
        }
        imageView.setImageBitmap(bitmap);
        startInference(input);
    }

    private void startInference(float[][][][] input) {
        FirebaseModelInputs inputs = null;
        try {
            inputs = new FirebaseModelInputs.Builder()
                    .add(input)  // add() as many input arrays as your model requires
                    .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                float[][] output = result.getOutput(0);
                                float[] probabilities = output[0];
                                classification_results(probabilities);
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                Log.d("inference status ", "onFailure: ");
                                Toast.makeText(MainActivity.this, "Inference failed", Toast.LENGTH_SHORT).show();
                            }
                        })
                .addOnCompleteListener(
                        new OnCompleteListener<FirebaseModelOutputs>() {
                            @Override
                            public void onComplete(@NonNull Task<FirebaseModelOutputs> task) {
                                alertDialog.dismiss();
                            }
                        }
                );

    }

    private void classification_results(float[] probabilities) {
        String results = "Confidence\n";
        for (int i = 0; i < 38; i++) {
            results += probabilities[i] + "\n";
        }
        textView.setText(results);
    }

    private class ThumbnailHandler extends AsyncTask<Bitmap, Void, Void> {
        @Override
        protected Void doInBackground(Bitmap... bitmaps) {
            return null;
        }
    }
}

