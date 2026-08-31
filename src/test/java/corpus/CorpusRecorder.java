package corpus;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Tarefa 5, segundo alvo — recolha do corpus por instrumentação em execução.
 *
 * <p>O primeiro alvo constrói o corpus lendo literais do código-fonte das
 * suites. Isso funciona no {@code MathOperations}, cujos testes foram escritos
 * com chamadas do tipo {@code gcd(12, 18)}, e falha em bibliotecas reais: no
 * {@code ArithmeticUtilsTest} do Commons Math há 154 invocações e apenas 8 com
 * argumentos literais. As restantes usam variáveis locais e ciclos, que uma
 * análise de texto não resolve.
 *
 * <p>Esta classe substitui a análise pela observação: regista os argumentos
 * realmente passados durante a execução da suite verdadeira, o que capta também
 * os valores gerados em ciclos.
 *
 * <p>Vive em {@code src/test/java} e no pacote {@code corpus} de propósito. A
 * configuração do PIT muta {@code dataloader.*}, pelo que nem esta classe nem o
 * delegador que a invoca podem ser mutados ou contaminar o
 * <em>mutation score</em>.
 *
 * <p>Guarda entradas distintas por ordem de primeira ocorrência. A duplicação é
 * abundante — a suite invoca {@code gcd} 43 vezes, muitas com os mesmos
 * argumentos — e o corpus interessa como conjunto de entradas distintas.
 */
public final class CorpusRecorder {

    /** Ordem de inserção preservada para que o ficheiro seja estável entre execuções. */
    private static final Set<String> ROWS = new LinkedHashSet<>();

    private CorpusRecorder() {
    }

    /** Regista uma invocação de dois argumentos inteiros. */
    public static void record(String method, int a, int b) {
        ROWS.add(method + "," + a + "," + b);
    }

    /** Regista uma invocação de um argumento; o segundo fica a zero. */
    public static void record(String method, int a) {
        ROWS.add(method + "," + a + ",0");
    }

    /**
     * Escreve o que foi acumulado.
     *
     * <p>Chamado a partir de um {@code shutdown hook} e não do fim de cada
     * teste: a suite tem vários métodos de teste e o Surefire pode reordená-los,
     * pelo que só no fim do processo é que o conjunto está completo.
     */
    public static void dump(String outPath) {
        try {
            Path path = Paths.get(outPath);
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            try (BufferedWriter w = new BufferedWriter(new FileWriter(path.toFile()))) {
                w.write("method,a,b\n");
                for (String row : ROWS) {
                    w.write(row);
                    w.write("\n");
                }
            }
            System.out.println("[corpus] " + ROWS.size()
                    + " invocacoes distintas escritas em " + outPath);
        } catch (IOException e) {
            // Falhar em silencio aqui daria um corpus vazio sem explicacao.
            throw new RuntimeException("falha a escrever o corpus: " + outPath, e);
        }
    }

    static {
        Runtime.getRuntime().addShutdownHook(new Thread(
                () -> dump("outputs/task5b/corpus_raw.csv")));
    }
}
