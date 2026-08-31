package tools;

import dataloader.ArithmeticUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Oráculo de regressão do segundo alvo experimental de E3.
 *
 * <p>Espelha o {@link OracleRunner} do primeiro alvo: lê entradas codificadas,
 * invoca a implementação ORIGINAL para cada método, e regista o resultado
 * observado. Os testes emitidos afirmam contra esses valores como literais em
 * tempo de compilação, e não por reexecução — de outra forma a mutação atingiria
 * os dois lados da igualdade e nenhum mutante morreria.
 *
 * <p>Está em {@code tools} pela mesma razão que o primeiro: a configuração do
 * PIT muta {@code dataloader.*}, e um oráculo mutável deixaria de ser oráculo.
 *
 * <p>A codificação do segundo alvo tem duas dimensões, {@code [s0, s1]}, e não
 * as doze do primeiro. Os seis métodos são todos {@code (int, int)}, pelo que
 * reutilizar o vetor longo deixaria dez dimensões mortas a dominar as métricas
 * de fidelidade — o defeito que a canonicalização corrigiu no primeiro alvo.
 *
 * <p>As exceções são resultado esperado e não falha: metade destes métodos
 * existe precisamente para as lançar quando há <em>overflow</em>. São
 * registadas pelo nome e a emissão converte-as em {@code assertThrows}.
 */
public final class ArithmeticOracleRunner {

    /** Os seis métodos (int, int) que a codificação representa. */
    private static final String[] METHODS = {
        "gcd", "lcm", "pow", "addAndCheck", "subAndCheck", "mulAndCheck",
        // As mesmas seis operacoes na sobrecarga long. Sao invocadas com os
        // mesmos pares (a, b) promovidos, e existem aqui porque de outra forma
        // dois tercos das mutacoes da classe ficariam permanentemente sem
        // cobertura: as sobrecargas long partilham o nome com as int e o PIT
        // conta-as na mesma classe.
        "gcdL", "lcmL", "powL", "addAndCheckL", "subAndCheckL", "mulAndCheckL",
    };

    private ArithmeticOracleRunner() {
    }

    private static String[] value(String v) {
        return new String[]{"VALUE", v};
    }

    private static String[] evaluate(String method, int a, int b) {
        try {
            switch (method) {
                case "gcd":
                    return value(String.valueOf(ArithmeticUtils.gcd(a, b)));
                case "lcm":
                    return value(String.valueOf(ArithmeticUtils.lcm(a, b)));
                case "pow":
                    return value(String.valueOf(ArithmeticUtils.pow(a, b)));
                case "addAndCheck":
                    return value(String.valueOf(ArithmeticUtils.addAndCheck(a, b)));
                case "subAndCheck":
                    return value(String.valueOf(ArithmeticUtils.subAndCheck(a, b)));
                case "mulAndCheck":
                    return value(String.valueOf(ArithmeticUtils.mulAndCheck(a, b)));
                case "gcdL":
                    return value(String.valueOf(ArithmeticUtils.gcd((long) a, (long) b)));
                case "lcmL":
                    return value(String.valueOf(ArithmeticUtils.lcm((long) a, (long) b)));
                case "powL":
                    return value(String.valueOf(ArithmeticUtils.pow((long) a, (long) b)));
                case "addAndCheckL":
                    return value(String.valueOf(ArithmeticUtils.addAndCheck((long) a, (long) b)));
                case "subAndCheckL":
                    return value(String.valueOf(ArithmeticUtils.subAndCheck((long) a, (long) b)));
                case "mulAndCheckL":
                    return value(String.valueOf(ArithmeticUtils.mulAndCheck((long) a, (long) b)));
                default:
                    throw new IllegalArgumentException("Unknown method: " + method);
            }
        } catch (Throwable t) {
            return new String[]{"EXCEPTION", t.getClass().getSimpleName()};
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("uso: ArithmeticOracleRunner <inputs.csv> <oracle.csv>");
            System.exit(2);
        }

        List<int[]> inputs = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(args[0]))) {
            String line = r.readLine();          // cabecalho
            while ((line = r.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] parts = line.split(",");
                inputs.add(new int[]{
                    Integer.parseInt(parts[0].trim()),
                    Integer.parseInt(parts[1].trim()),
                });
            }
        }

        int rows = 0;
        try (BufferedWriter w = new BufferedWriter(new FileWriter(args[1]))) {
            w.write("method,s0,s1,outcome,value\n");
            for (int[] in : inputs) {
                for (String m : METHODS) {
                    String[] res = evaluate(m, in[0], in[1]);
                    w.write(m + "," + in[0] + "," + in[1] + ","
                            + res[0] + "," + res[1] + "\n");
                    rows++;
                }
            }
        }
        System.out.println("[oraculo] " + rows + " linhas escritas em " + args[1]);
    }
}
