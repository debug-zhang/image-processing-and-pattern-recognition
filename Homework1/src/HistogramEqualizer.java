import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class HistogramEqualizer {
    private String path;
    private String name;

    private int width;
    private int height;

    private int[] greyPixels;
    private BufferedImage equalizationImage;

    public HistogramEqualizer(String path, String name, int[] greyPixels,
                              int width, int height) {
        this.path = path;
        this.name = name;

        this.greyPixels = greyPixels;

        this.width = width;
        this.height = height;
    }

    public void HistogramEqualization() {
        int[] greyFrequency = new int[256];
        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                int grey = this.greyPixels[this.height * i + j] & 0xff;
                greyFrequency[grey]++;
            }
        }

        double[] p = new double[256];
        double[] s = new double[256];
        int sum = this.width * this.height;
        for (int i = 0; i < 256; i++) {
            p[i] = (double) greyFrequency[i] / sum;

            if (i > 0) {
                s[i] = s[i - 1] + p[i];
            } else {
                s[i] = p[i];
            }
        }
        for (int i = 0; i < 256; i++) {
            s[i] *= 255;
        }

        int[] equalizationPixels = new int[this.width * this.height];
        equalizationImage = new BufferedImage(this.width, this.height,
                BufferedImage.TYPE_3BYTE_BGR);

        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                int index = this.height * i + j;
                int level = (int) s[this.greyPixels[index] & 0xff];
                equalizationPixels[index] = level << 16 | level << 8 | level;
                equalizationImage.setRGB(i, j, equalizationPixels[index]);
            }
        }
    }

    public void WriteEqualizationImage() throws IOException {
        File file = new File(path + "equalization_" + name);
        ImageIO.write(this.equalizationImage, "jpeg", file);
    }
}
