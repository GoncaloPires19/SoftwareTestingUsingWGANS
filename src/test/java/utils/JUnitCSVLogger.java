package utils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

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

    /**
     * Logs a 3-feature tuple to CSV.
     *
     * The column names methods/branches/statements are preserved for backward
     * compatibility with existing Python scripts, but the values are generic
     * test-derived numeric features produced by each test suite.
     */
    public static synchronized void log(double methods, double branches, double statements) {
        try {
            writer.append(String.valueOf(methods))
                  .append(",")
                  .append(String.valueOf(branches))
                  .append(",")
                  .append(String.valueOf(statements))
                  .append("\n");
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException("Failed to log test data: " + e.getMessage(), e);
        }
    }
}
