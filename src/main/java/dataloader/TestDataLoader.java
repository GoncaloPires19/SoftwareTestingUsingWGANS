package dataloader;


import java.io.*;
import java.util.*;

public class TestDataLoader {

    private List<String> collectedData = new ArrayList<>();

    public void collectTestData(Object... values) {
        // Junta valores separados por vírgula
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            sb.append(values[i]);
            if (i < values.length - 1) sb.append(",");
        }
        collectedData.add(sb.toString());
    }

    public void exportToCSV(String outputFile) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        for (String line : collectedData) {
            writer.write(line + "\n");
        }
        writer.close();
    }

    public int getCollectedDataSize() {
        return collectedData.size();
    }

    public String getCollectedData(int index) {
        if (index >= 0 && index < collectedData.size()) {
            return collectedData.get(index);
        }
        return null;
    }

    public List<String> getAllCollectedData() {
        return new ArrayList<>(collectedData);
    }
}
