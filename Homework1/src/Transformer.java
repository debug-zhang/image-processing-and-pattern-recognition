import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Transformer {
    private String path;
    private String name;

    private int width;
    private int height;

    private int[] greyPixels;
    private BufferedImage linearTransformImage;
    private BufferedImage linearStretchImage;

    public Transformer(String path, String name, int[] greyPixels,
                       int width, int height) {
        this.path = path;
        this.name = name;

        this.greyPixels = greyPixels;

        this.width = width;
        this.height = height;
    }

    public void LinearTransformation() {
        int a = 0;
        int b = 255;
        int c = 64;
        int d = 255;
        double k = (double) (d - c) / (b - a);

        this.linearTransformImage = new BufferedImage(this.width, this.height,
                BufferedImage.TYPE_3BYTE_BGR);
        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                int grey = this.greyPixels[this.height * i + j] & 0xff;
                grey = (int) (k * (grey - a) + c);
                grey = grey << 16 | grey << 8 | grey;
                this.linearTransformImage.setRGB(i, j, grey);
            }
        }
    }

    public void LinearStretch() {
        int M = 255;
        int a = 32;
        int b = 128;
        int c = 8;
        int d = 192;
        int e = 255;
        double k1 = (double) c / a;
        double k2 = (double) (d - c) / (b - a);
        double k3 = (double) (e - d) / (M - b);

        this.linearStretchImage = new BufferedImage(this.width, this.height,
                BufferedImage.TYPE_3BYTE_BGR);
        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                int grey = this.greyPixels[this.height * i + j] & 0xff;

                if (grey < a) {
                    grey = (int) (k1 * grey);
                } else if (grey < b) {
                    grey = (int) (k2 * (grey - a) + c);
                } else {
                    grey = (int) (k3 * (grey - b) + d);
                }

                grey = grey << 16 | grey << 8 | grey;
                this.linearStretchImage.setRGB(i, j, grey);
            }
        }
    }

    public void WriteLinearTransformImage() throws IOException {
        File file = new File(path + "linear_transform_" + name);
        ImageIO.write(this.linearTransformImage, "jpeg", file);
    }

    public void WriteLinearStretchImage() throws IOException {
        File file = new File(path + "linear_stretch_" + name);
        ImageIO.write(this.linearStretchImage, "jpeg", file);
    }
}

