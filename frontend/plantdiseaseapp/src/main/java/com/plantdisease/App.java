package com.plantdisease;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.stage.Stage;
import javafx.stage.FileChooser;
import javafx.scene.Scene;
import javafx.scene.layout.*;
import javafx.scene.control.*;
import javafx.scene.image.*;
import javafx.geometry.Pos;
import javafx.geometry.Insets;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import javafx.scene.paint.Color;
import javafx.concurrent.Task;

import java.io.File;
import java.io.IOException;

import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.entity.ContentType;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class App extends Application {

    private Label resultLabel = new Label("Prediction will appear here");
    private Label loadingLabel = new Label("");
    private ProgressIndicator progressIndicator = new ProgressIndicator();
    private ImageView imageView = new ImageView();

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("CROPxPERT");

        // Header Section
        Label headingLabel = new Label("CROPxPERT: A One Stop Solution for Farmers");
        headingLabel.setFont(new Font("Arial", 24));
        headingLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #333333;");
        headingLabel.setTextAlignment(TextAlignment.CENTER);

        Label subHeadingLabel = new Label("Input infected crop image to know the disease");
        subHeadingLabel.setFont(new Font("Arial", 16));
        subHeadingLabel.setStyle("-fx-font-style: italic; -fx-text-fill: #666666;");
        subHeadingLabel.setTextAlignment(TextAlignment.CENTER);

        VBox headerBox = new VBox(5, headingLabel, subHeadingLabel);
        headerBox.setAlignment(Pos.CENTER);
        headerBox.setPadding(new Insets(10, 0, 20, 0));

        // Image Display
        imageView.setFitHeight(200);
        imageView.setFitWidth(200);
        imageView.setPreserveRatio(true);
        imageView.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.2), 10, 0, 0, 0);"
                + "-fx-border-color: #ccc; -fx-border-width: 2;");
        VBox imageBox = new VBox(imageView);
        imageBox.setAlignment(Pos.CENTER);

        // Upload Button
        Button uploadButton = new Button("Upload Leaf Image");
        uploadButton.setStyle("""
            -fx-background-color: linear-gradient(to right, #4CAF50, #2E7D32);
            -fx-text-fill: white;
            -fx-font-size: 16px;
            -fx-padding: 10px 20px;
            -fx-border-radius: 5px;
            -fx-background-radius: 5px;
            -fx-cursor: hand;
        """);

        uploadButton.setOnMouseEntered(e -> uploadButton.setStyle("""
            -fx-background-color: linear-gradient(to right, #66BB6A, #388E3C);
            -fx-text-fill: white;
            -fx-font-size: 16px;
            -fx-padding: 10px 20px;
            -fx-border-radius: 5px;
            -fx-background-radius: 5px;
            -fx-cursor: hand;
        """));

        uploadButton.setOnMouseExited(e -> uploadButton.setStyle("""
            -fx-background-color: linear-gradient(to right, #4CAF50, #2E7D32);
            -fx-text-fill: white;
            -fx-font-size: 16px;
            -fx-padding: 10px 20px;
            -fx-border-radius: 5px;
            -fx-background-radius: 5px;
            -fx-cursor: hand;
        """));

        uploadButton.setTooltip(new Tooltip("Click to upload a crop leaf image for disease prediction"));

        uploadButton.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select Image");
            File selectedFile = fileChooser.showOpenDialog(primaryStage);

            if (selectedFile != null) {
                imageView.setImage(new Image(selectedFile.toURI().toString()));
                sendImageToFlask(selectedFile);
            }
        });

        // Result and Loading Display
        resultLabel.setFont(new Font("Arial", 18));
        resultLabel.setStyle("-fx-text-fill: #2E86C1; -fx-font-weight: bold;");
        resultLabel.setTextAlignment(TextAlignment.CENTER);

        loadingLabel.setFont(new Font("Arial", 14));
        loadingLabel.setStyle("-fx-text-fill: #E74C3C; -fx-font-style: italic;");
        loadingLabel.setTextAlignment(TextAlignment.CENTER);

        progressIndicator.setVisible(false);
        progressIndicator.setPrefSize(30, 30);

        HBox loadingBox = new HBox(10, loadingLabel, progressIndicator);
        loadingBox.setAlignment(Pos.CENTER);

        // Main Layout
        VBox mainLayout = new VBox(20, headerBox, imageBox, uploadButton, loadingBox, resultLabel);
        mainLayout.setAlignment(Pos.CENTER);
        mainLayout.setPadding(new Insets(20));
        mainLayout.setStyle("-fx-background-color: #F4F6F7;"); // Light background

        // Scene Configuration
        Scene scene = new Scene(mainLayout, 600, 700);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void sendImageToFlask(File imageFile) {
        loadingLabel.setText("Uploading and processing image...");
        progressIndicator.setVisible(true);
        resultLabel.setText("");

        Task<Void> task = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                try (CloseableHttpClient client = HttpClients.createDefault()) {
                    HttpPost post = new HttpPost("http://127.0.0.1:5000/predict");

                    HttpEntity entity = MultipartEntityBuilder.create()
                            .addBinaryBody("file", imageFile, ContentType.DEFAULT_BINARY, imageFile.getName())
                            .build();

                    post.setEntity(entity);

                    try (CloseableHttpResponse response = client.execute(post)) {
                        String jsonResponse = EntityUtils.toString(response.getEntity());
                        ObjectMapper objectMapper = new ObjectMapper();
                        JsonNode jsonNode = objectMapper.readTree(jsonResponse);

                        String predictedClass = jsonNode.path("predicted_class").asText();
                        String advice = jsonNode.path("advice").asText();

                        Platform.runLater(() -> {
                            resultLabel.setText("Prediction: " + predictedClass + "\nAdvice: " + advice);
                            loadingLabel.setText("");
                            progressIndicator.setVisible(false);
                        });

                    } catch (IOException ex) {
                        Platform.runLater(() -> {
                            resultLabel.setText("Error in response: " + ex.getMessage());
                            loadingLabel.setText("");
                            progressIndicator.setVisible(false);
                        });
                    }
                } catch (IOException ex) {
                    Platform.runLater(() -> {
                        resultLabel.setText("Error: " + ex.getMessage());
                        loadingLabel.setText("");
                        progressIndicator.setVisible(false);
                    });
                }
                return null;
            }
        };

        new Thread(task).start();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
