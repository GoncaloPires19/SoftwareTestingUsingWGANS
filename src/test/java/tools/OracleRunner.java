package tools;

import dataloader.MathOperations;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tarefa 5, passo 3 — regression oracle for GAN/random generated test inputs.
 *
 * <p>Reads a CSV of encoded inputs, invokes the ORIGINAL (unmutated)
 * {@link MathOperations} for every method of the matching family, and records
 * the observed outcome. The emitted JUnit tests then assert against these
 * recorded values as compile-time literals.
 *
 * <p>Baking the expected value in as a literal is what makes the generated
 * suite able to kill mutants: a test shaped like
 * {@code assertEquals(MathOperations.sum(x), MathOperations.sum(x))} would
 * mutate on both sides and never fail.
 *
 * <p>Deliberately placed in package {@code tools} (test sources), NOT in
 * {@code dataloader} — the PIT config mutates {@code dataloader.*}, so keeping
 * this harness outside that package stops it from polluting the mutation score.
 *
 * <p>Input CSV columns:  arr_len,v0..v7,s0,s1,s2   (arr_len == -1 means null array)
 * <p>Output CSV columns: method,arr_len,v0..v7,s0,s1,s2,outcome,value
 *   <ul>
 *     <li>outcome = VALUE     -&gt; value holds the result (arrays as "1;2;3")</li>
 *     <li>outcome = EXCEPTION -&gt; value holds the exception's simple name</li>
 *   </ul>
 *
 * <p>Usage: {@code java -cp target/classes:target/test-classes tools.OracleRunner <in.csv> <out.csv>}
 */
public final class OracleRunner {

    private static final String[] ARRAY_METHODS = {
            "sum", "max", "min", "average", "median", "reverseArray", "bubbleSort"
    };

    private OracleRunner() {
    }

    /** One decoded row of the input CSV. */
    private static final class Input {
        int arrLen;
        int[] v = new int[8];
        int s0, s1, s2;

        /** @return the int[] argument, or null when arrLen == -1 */
        int[] array() {
            if (arrLen < 0) {
                return null;
            }
            return Arrays.copyOfRange(v, 0, arrLen);
        }

        String encode() {
            StringBuilder sb = new StringBuilder();
            sb.append(arrLen);
            for (int x : v) {
                sb.append(',').append(x);
            }
            sb.append(',').append(s0).append(',').append(s1).append(',').append(s2);
            return sb.toString();
        }
    }

    private static String arrayToString(int[] arr) {
        if (arr == null) {
            return "null";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) {
                sb.append(';');
            }
            sb.append(arr[i]);
        }
        return sb.toString();
    }

    /** Invokes one method, converting both normal results and exceptions into a CSV cell. */
    private static String[] invoke(String method, Input in) {
        try {
            switch (method) {
                case "sum":
                    return value(String.valueOf(MathOperations.sum(in.array())));
                case "max":
                    return value(String.valueOf(MathOperations.max(in.array())));
                case "min":
                    return value(String.valueOf(MathOperations.min(in.array())));
                case "average":
                    return value(String.valueOf(MathOperations.average(in.array())));
                case "median":
                    return value(String.valueOf(MathOperations.median(in.array())));
                case "reverseArray":
                    return value(arrayToString(MathOperations.reverseArray(in.array())));
                case "bubbleSort":
                    return value(arrayToString(MathOperations.bubbleSort(in.array())));
                case "countOccurrences":
                    return value(String.valueOf(MathOperations.countOccurrences(in.array(), in.s0)));
                case "isPrime":
                    return value(String.valueOf(MathOperations.isPrime(in.s0)));
                case "factorial":
                    return value(String.valueOf(MathOperations.factorial(in.s0)));
                case "fibonacci":
                    return value(String.valueOf(MathOperations.fibonacci(in.s0)));
                case "isPerfectSquare":
                    return value(String.valueOf(MathOperations.isPerfectSquare(in.s0)));
                case "gcd":
                    return value(String.valueOf(MathOperations.gcd(in.s0, in.s1)));
                case "clamp":
                    return value(String.valueOf(MathOperations.clamp(in.s0, in.s1, in.s2)));
                default:
                    throw new IllegalArgumentException("Unknown method: " + method);
            }
        } catch (Throwable t) {
            return new String[]{"EXCEPTION", t.getClass().getSimpleName()};
        }
    }

    private static String[] value(String v) {
        return new String[]{"VALUE", v};
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: OracleRunner <inputs.csv> <oracle.csv>");
            System.exit(2);
        }

        List<Input> inputs = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(args[0]))) {
            String line = reader.readLine(); // header
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 12) {
                    continue;
                }
                Input in = new Input();
                in.arrLen = Integer.parseInt(parts[0].trim());
                for (int i = 0; i < 8; i++) {
                    in.v[i] = Integer.parseInt(parts[1 + i].trim());
                }
                in.s0 = Integer.parseInt(parts[9].trim());
                in.s1 = Integer.parseInt(parts[10].trim());
                in.s2 = Integer.parseInt(parts[11].trim());
                inputs.add(in);
            }
        }

        int rows = 0;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(args[1]))) {
            writer.write("method,arr_len,v0,v1,v2,v3,v4,v5,v6,v7,s0,s1,s2,outcome,value\n");
            for (Input in : inputs) {
                List<String> methods = new ArrayList<>(Arrays.asList(ARRAY_METHODS));
                methods.add("countOccurrences");
                methods.add("isPrime");
                methods.add("factorial");
                methods.add("fibonacci");
                methods.add("isPerfectSquare");
                methods.add("gcd");
                methods.add("clamp");

                for (String method : methods) {
                    String[] outcome = invoke(method, in);
                    writer.write(method + "," + in.encode() + "," + outcome[0] + "," + outcome[1] + "\n");
                    rows++;
                }
            }
        }

        System.out.println("Inputs lidos : " + inputs.size());
        System.out.println("Linhas oracle: " + rows);
        System.out.println("Output       : " + args[1]);
    }
}
