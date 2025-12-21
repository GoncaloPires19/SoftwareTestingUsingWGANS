package utils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.util.List;

public class JUnitCSVLogger {

    private static final String OUTPUT_DIR = "python/data";
    private static final String FILE_NAME = "junit_data.csv";
    private static FileWriter writer;

    static {
        try {
            File dir = new File(OUTPUT_DIR);
            if (!dir.exists()) dir.mkdirs();

            writer = new FileWriter(OUTPUT_DIR + "/" + FILE_NAME);
        } catch (IOException e) {
            throw new RuntimeException("Erro a criar CSV: " + e.getMessage());
        }
    }

    public static synchronized void log(List<Double> values) {
        try {
            for (int i = 0; i < values.size(); i++) {
                writer.append(values.get(i).toString());
                if (i < values.size() - 1) writer.append(",");
            }
            writer.append("\n");
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
