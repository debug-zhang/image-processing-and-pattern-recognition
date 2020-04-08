import javax.imageio.ImageIO;
import javax.swing.JFrame;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class GreyProcessor extends JFrame {
    private String path;
    private String name;

    private int width;
    private int height;

    private BufferedImage image;
    private BufferedImage greyImage;

    private int[] greyPixels;

    public GreyProcessor(String path, String name) throws IOException {
        this.path = path;
        this.name = name;

        File file = new File(path + name);
        this.image = ImageIO.read(file);

        this.width = this.image.getWidth();
        this.height = this.image.getHeight();
    }

    public void GreyProcessing() throws InterruptedException {
        this.greyImage = new BufferedImage(this.width, this.height,
                BufferedImage.TYPE_3BYTE_BGR);
        this.greyPixels = new int[this.width * this.height];
        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                int argb = image.getRGB(i, j);
                int a = (argb >> 24) & 0xff;
                int r = (argb >> 16) & 0xff;
                int g = (argb >> 8) & 0xff;
                int b = (argb) & 0xff;
                int grey = (int) (0.229 * r + 0.587 * g + 0.114 * b);
                argb = (a << 24) | (grey << 16) | (grey << 8) | grey;
                this.greyImage.setRGB(i, j, argb);
                this.greyPixels[this.height * i + j] = argb;
            }
        }
    }

    public void WriteGrayImage() throws IOException {
        File file = new File(path + "grey_" + name);
        ImageIO.write(this.greyImage, "jpeg", file);
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    public int[] getGreyPixels() {
        return greyPixels;
    }
}
