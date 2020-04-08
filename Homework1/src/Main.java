import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException,
            InterruptedException {
        String path = "C:/Users/blackbird/JetBrains/IdeaProjects/ImageProcessing/image/";
        String name = "brain.jpg";

        GreyProcessor greyProcessor = new GreyProcessor(path, name);
        greyProcessor.GreyProcessing();
        greyProcessor.WriteGrayImage();

        int width = greyProcessor.getWidth();
        int height = greyProcessor.getHeight();
        int[] greyPixels = greyProcessor.getGreyPixels();

        HistogramEqualizer histogramEqualizer = new HistogramEqualizer(
                path, name, greyPixels, width, height);
        histogramEqualizer.HistogramEqualization();
        histogramEqualizer.WriteEqualizationImage();

        Transformer transformer = new Transformer(
                path, name, greyPixels, width, height);
        transformer.LinearTransformation();
        transformer.WriteLinearTransformImage();

        transformer.LinearStretch();
        transformer.WriteLinearStretchImage();
    }
}
