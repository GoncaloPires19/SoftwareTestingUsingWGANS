package corpus;

import dataloader.ArithmeticUtils;

import java.math.BigInteger;

/**
 * Delegador instrumentado para a recolha do corpus do segundo alvo.
 *
 * <p>Cada método regista os argumentos e delega no {@link ArithmeticUtils} real.
 * Não há lógica duplicada, pelo que esta classe não pode divergir da classe sob
 * teste por muito que esta mude.
 *
 * <p>A alternativa seria acrescentar chamadas ao registador dentro do próprio
 * {@code ArithmeticUtils}. Seria mais simples e estaria errado por duas razões:
 * o PIT mutaria essas chamadas junto com o resto, e a classe medida deixaria de
 * ser a classe publicada pela biblioteca. Com o delegador, o que é medido
 * mantém-se literalmente igual ao original, à excepção da linha de pacote.
 *
 * <p>Só estão aqui os métodos que a codificação {@code [s0, s1]} representa:
 * seis operações {@code (int, int)}. As sobrecargas {@code long} e
 * {@code BigInteger} delegam sem registar, porque a suite do Commons Math
 * invoca-as e têm de compilar, mas não entram no corpus.
 */
public final class ArithmeticUtilsProbe {

    private ArithmeticUtilsProbe() {
    }

    // --- as seis operacoes (int, int) que formam o corpus -----------------

    public static int gcd(int p, int q) {
        CorpusRecorder.record("gcd", p, q);
        return ArithmeticUtils.gcd(p, q);
    }

    public static int lcm(int a, int b) {
        CorpusRecorder.record("lcm", a, b);
        return ArithmeticUtils.lcm(a, b);
    }

    public static int pow(final int k, final int e) {
        CorpusRecorder.record("pow", k, e);
        return ArithmeticUtils.pow(k, e);
    }

    public static int addAndCheck(int x, int y) {
        CorpusRecorder.record("addAndCheck", x, y);
        return ArithmeticUtils.addAndCheck(x, y);
    }

    public static int subAndCheck(int x, int y) {
        CorpusRecorder.record("subAndCheck", x, y);
        return ArithmeticUtils.subAndCheck(x, y);
    }

    public static int mulAndCheck(int x, int y) {
        CorpusRecorder.record("mulAndCheck", x, y);
        return ArithmeticUtils.mulAndCheck(x, y);
    }

    // --- sobrecargas que a suite invoca mas que ficam fora do corpus ------
    // A codificacao e de dois inteiros; long e BigInteger nao sao
    // representaveis e nao devem poluir o corpus com valores que o gerador
    // nunca podera emitir.

    public static long gcd(final long p, final long q) {
        return ArithmeticUtils.gcd(p, q);
    }

    public static long lcm(long a, long b) {
        return ArithmeticUtils.lcm(a, b);
    }

    public static long pow(final long k, final long e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static long pow(final long k, final int e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static int pow(final int k, long e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static BigInteger pow(final BigInteger k, int e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static BigInteger pow(final BigInteger k, long e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static BigInteger pow(final BigInteger k, BigInteger e) {
        return ArithmeticUtils.pow(k, e);
    }

    public static long addAndCheck(long a, long b) {
        return ArithmeticUtils.addAndCheck(a, b);
    }

    public static long subAndCheck(long a, long b) {
        return ArithmeticUtils.subAndCheck(a, b);
    }

    public static long mulAndCheck(long a, long b) {
        return ArithmeticUtils.mulAndCheck(a, b);
    }

    public static long factorial(final int n) {
        return ArithmeticUtils.factorial(n);
    }

    public static double factorialDouble(final int n) {
        return ArithmeticUtils.factorialDouble(n);
    }

    public static double factorialLog(final int n) {
        return ArithmeticUtils.factorialLog(n);
    }

    public static long binomialCoefficient(final int n, final int k) {
        return ArithmeticUtils.binomialCoefficient(n, k);
    }

    public static double binomialCoefficientDouble(final int n, final int k) {
        return ArithmeticUtils.binomialCoefficientDouble(n, k);
    }

    public static double binomialCoefficientLog(final int n, final int k) {
        return ArithmeticUtils.binomialCoefficientLog(n, k);
    }

    public static long stirlingS2(final int n, final int k) {
        return ArithmeticUtils.stirlingS2(n, k);
    }

    public static boolean isPowerOfTwo(long n) {
        return ArithmeticUtils.isPowerOfTwo(n);
    }
}
