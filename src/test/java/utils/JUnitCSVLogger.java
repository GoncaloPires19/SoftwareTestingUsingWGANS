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
     * Logs test metrics to CSV file with exactly 3 columns: methods, branches, statements.
     * 
     * @param methods Number of methods covered/tested
     * @param branches Number of branches covered/tested
     * @param statements Number of statements covered/tested
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
