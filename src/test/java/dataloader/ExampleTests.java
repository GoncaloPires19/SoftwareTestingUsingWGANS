package dataloader;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import utils.JUnitCSVLogger;
import java.util.List;
import java.io.*;
import java.nio.file.*;

class ExampleTests {

    private TestDataLoader dataLoader;

    @BeforeEach
    void setUp() {
        dataLoader = new TestDataLoader();
    }

    @Test
    void testSum() {
        int[][] testCases = {
            {5,3},{2,7},{10,4},{1,6},{8,2},
            {15,5},{3,14},{6,6},{9,7},{12,8}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = a + b;

            assertEquals(expected, a + b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testMultiply() {
        int[][] testCases = {
            {2,3},{4,5},{1,10},{6,7},{3,3},
            {5,5},{7,8},{9,2},{10,4},{12,6}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = a * b;

            assertEquals(expected, a * b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testCollectTestDataSingleValue() {
        dataLoader.collectTestData(42);
        assertEquals(1, dataLoader.getCollectedDataSize());
    }

    @Test
    void testCollectTestDataMultipleValues() {
        dataLoader.collectTestData(10, 20, 30);
        assertEquals(1, dataLoader.getCollectedDataSize());
        assertEquals("10,20,30", dataLoader.getCollectedData(0));
    }

    @Test
    void testCollectTestDataStringValues() {
        dataLoader.collectTestData("test", "data", "value");
        assertEquals(1, dataLoader.getCollectedDataSize());
        assertEquals("test,data,value", dataLoader.getCollectedData(0));
    }

    @Test
    void testCollectTestDataMixedTypes() {
        dataLoader.collectTestData(1, "two", 3.0, true);
        assertEquals(1, dataLoader.getCollectedDataSize());
        assertEquals("1,two,3.0,true", dataLoader.getCollectedData(0));
    }

    @Test
    void testCollectMultipleDataLines() {
        dataLoader.collectTestData(1, 2);
        dataLoader.collectTestData(3, 4);
        dataLoader.collectTestData(5, 6);
        assertEquals(3, dataLoader.getCollectedDataSize());
        assertEquals("1,2", dataLoader.getCollectedData(0));
        assertEquals("3,4", dataLoader.getCollectedData(1));
        assertEquals("5,6", dataLoader.getCollectedData(2));
    }

    @Test
    void testGetCollectedDataOutOfBounds() {
        dataLoader.collectTestData(10, 20, 30);

        assertNull(dataLoader.getCollectedData(-1));
        assertNull(dataLoader.getCollectedData(1));
    }

    @Test
    void testGetAllCollectedDataReturnsDefensiveCopy() {
        dataLoader.collectTestData("a", "b");
        dataLoader.collectTestData("c", "d");

        List<String> exported = dataLoader.getAllCollectedData();
        assertEquals(2, exported.size());
        assertEquals("a,b", exported.get(0));
        assertEquals("c,d", exported.get(1));

        exported.clear();
        assertEquals(2, dataLoader.getCollectedDataSize());
    }

    @Test
    void testExportToCSV() throws IOException {
        String testFile = "test_export.csv";
        dataLoader.collectTestData(100, 200, 300);
        dataLoader.collectTestData(400, 500, 600);
        dataLoader.exportToCSV(testFile);

        assertTrue(Files.exists(Paths.get(testFile)));
        
        List<String> lines = Files.readAllLines(Paths.get(testFile));
        assertEquals(2, lines.size());
        assertEquals("100,200,300", lines.get(0));
        assertEquals("400,500,600", lines.get(1));

        Files.delete(Paths.get(testFile));
    }

    @Test
    void testExportToCSVEmptyData() throws IOException {
        String testFile = "test_empty_export.csv";
        dataLoader.exportToCSV(testFile);

        assertTrue(Files.exists(Paths.get(testFile)));
        
        List<String> lines = Files.readAllLines(Paths.get(testFile));
        assertEquals(0, lines.size());

        Files.delete(Paths.get(testFile));
    }

    @Test
    void testDivision() {
        int[][] testCases = {
            {10,2},{20,4},{15,3},{100,10},{25,5},
            {36,6},{49,7},{64,8},{81,9},{144,12}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = a / b;

            assertEquals(expected, a / b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testSubtraction() {
        int[][] testCases = {
            {10,3},{20,7},{15,4},{100,20},{50,15},
            {30,10},{45,15},{60,25},{80,35},{120,45}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = a - b;

            assertEquals(expected, a - b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testModulo() {
        int[][] testCases = {
            {10,3},{20,7},{15,4},{100,30},{50,15},
            {17,5},{23,8},{31,10},{47,12},{99,25}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = a % b;

            assertEquals(expected, a % b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testPower() {
        int[][] testCases = {
            {2,3},{2,5},{3,4},{5,2},{10,2},
            {2,8},{4,3},{6,2},{7,2},{3,5}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            double expected = Math.pow(a, b);

            assertEquals(expected, Math.pow(a, b));

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                expected
            );
        }
    }

    @Test
    void testSquareRoot() {
        double[] testCases = {4, 9, 16, 25, 36, 49, 64, 81, 100, 144};

        for (double tc : testCases) {
            double expected = Math.sqrt(tc);

            assertEquals(expected, Math.sqrt(tc));

            JUnitCSVLogger.log(
                tc,
                expected,
                0.0
            );
        }
    }

    @Test
    void testAbsoluteValue() {
        int[][] testCases = {
            {-5,5},{-10,10},{-1,1},{-100,100},{-42,42},
            {5,5},{10,10},{1,1},{100,100},{42,42}
        };

        for (int[] tc : testCases) {
            int value = tc[0];
            int expected = tc[1];

            assertEquals(expected, Math.abs(value));

            JUnitCSVLogger.log(
                (double) value,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testMinValue() {
        int[][] testCases = {
            {5,3,3},{10,7,7},{20,15,15},{100,50,50},{42,8,8},
            {1,1,1},{99,88,88},{50,40,40},{200,150,150},{777,333,333}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, Math.min(a, b));

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testMaxValue() {
        int[][] testCases = {
            {5,3,5},{10,7,10},{20,15,20},{100,50,100},{42,8,42},
            {1,1,1},{99,88,99},{50,40,50},{200,150,200},{777,333,777}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, Math.max(a, b));

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testCubicPower() {
        int[][] testCases = {
            {1,1},{2,8},{3,27},{4,64},{5,125},
            {6,216},{7,343},{8,512},{9,729},{10,1000}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int expected = tc[1];

            assertEquals(expected, a * a * a);

            JUnitCSVLogger.log(
                (double) a,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testSquare() {
        int[][] testCases = {
            {1,1},{2,4},{3,9},{4,16},{5,25},
            {6,36},{7,49},{8,64},{9,81},{10,100}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int expected = tc[1];

            assertEquals(expected, a * a);

            JUnitCSVLogger.log(
                (double) a,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testAverage() {
        double[][] testCases = {
            {2,4,3},{10,20,15},{5,5,5},{1,9,5},{100,200,150},
            {3,3,3},{25,75,50},{11,13,12},{7,8,7.5},{99,1,50}
        };

        for (double[] tc : testCases) {
            double a = tc[0];
            double b = tc[1];
            double expected = tc[2];

            assertEquals(expected, (a + b) / 2, 0.0001);

            JUnitCSVLogger.log(
                a,
                b,
                expected
            );
        }
    }

    @Test
    void testPercentage() {
        double[][] testCases = {
            {50,100,50},{25,100,25},{75,100,75},{10,50,20},{200,400,50},
            {30,60,50},{15,30,50},{5,100,5},{80,200,40},{99,99,100}
        };

        for (double[] tc : testCases) {
            double part = tc[0];
            double total = tc[1];
            double expected = tc[2];

            assertEquals(expected, (part / total) * 100, 0.0001);

            JUnitCSVLogger.log(
                part,
                total,
                expected
            );
        }
    }

    @Test
    void testBitwiseAnd() {
        int[][] testCases = {
            {5,3,1},{10,6,2},{15,7,7},{8,4,0},{12,10,8},
            {7,3,3},{11,5,1},{14,10,10},{9,6,0},{16,8,0}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, a & b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testBitwiseOr() {
        int[][] testCases = {
            {5,3,7},{10,6,14},{15,7,15},{8,4,12},{12,10,14},
            {7,3,7},{11,5,15},{14,10,14},{9,6,15},{16,8,24}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, a | b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testBitwiseXor() {
        int[][] testCases = {
            {5,3,6},{10,6,12},{15,7,8},{8,4,12},{12,10,6},
            {7,3,4},{11,5,14},{14,10,4},{9,6,15},{16,8,24}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, a ^ b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testLeftShift() {
        int[][] testCases = {
            {1,1,2},{1,2,4},{1,3,8},{2,1,4},{2,2,8},
            {3,1,6},{5,2,20},{7,3,56},{2,4,32},{1,5,32}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, a << b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testRightShift() {
        int[][] testCases = {
            {8,1,4},{8,2,2},{16,1,8},{16,2,4},{32,1,16},
            {32,2,8},{64,2,16},{100,2,25},{128,3,16},{256,4,16}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, a >> b);

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testFactorial() {
        int[][] testCases = {
            {0,1},{1,1},{2,2},{3,6},{4,24},
            {5,120},{6,720},{7,5040},{8,40320},{9,362880}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1];

            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }

            assertEquals(expected, result);

            JUnitCSVLogger.log(
                (double) n,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testIsEven() {
        int[][] testCases = {
            {2,1},{4,1},{6,1},{8,1},{10,1},
            {1,0},{3,0},{5,0},{7,0},{9,0}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1]; // 1 = true/even, 0 = false/odd

            int result = n % 2 == 0 ? 1 : 0;
            assertEquals(expected, result);

            JUnitCSVLogger.log(
                (double) n,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testIsPrime() {
        int[][] testCases = {
            {2,1},{3,1},{5,1},{7,1},{11,1},
            {13,1},{17,1},{19,1},{23,1},{1,0},
            {4,0},{6,0},{8,0},{9,0},{10,0}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1]; // 1 = prime, 0 = not prime

            int result = isPrime(n) ? 1 : 0;
            assertEquals(expected, result);

            JUnitCSVLogger.log(
                (double) n,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testGCD() {
        int[][] testCases = {
            {12,8,4},{15,25,5},{20,30,10},{48,18,6},{100,50,50},
            {7,13,1},{21,14,7},{36,24,12},{99,66,33},{144,60,12}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, gcd(a, b));

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testLCM() {
        int[][] testCases = {
            {4,6,12},{8,12,24},{5,7,35},{10,15,30},{12,18,36},
            {3,5,15},{6,9,18},{4,4,4},{7,11,77},{20,30,60}
        };

        for (int[] tc : testCases) {
            int a = tc[0];
            int b = tc[1];
            int expected = tc[2];

            assertEquals(expected, lcm(a, b));

            JUnitCSVLogger.log(
                (double) a,
                (double) b,
                (double) expected
            );
        }
    }

    @Test
    void testFibonacci() {
        int[][] testCases = {
            {0,0},{1,1},{2,1},{3,2},{4,3},
            {5,5},{6,8},{7,13},{8,21},{9,34}
        };

        for (int[] tc : testCases) {
            int n = tc[0];
            int expected = tc[1];

            assertEquals(expected, fibonacci(n));

            JUnitCSVLogger.log(
                (double) n,
                (double) expected,
                0.0
            );
        }
    }

    @Test
    void testIsPalindrome() {
        String[] palindromes = {"a","aba","abba","racecar","level","noon","deed"};
        String[] nonPalindromes = {"ab","abc","hello","world","test"};

        for (String str : palindromes) {
            int result = isPalindrome(str) ? 1 : 0;
            assertEquals(1, result);
            JUnitCSVLogger.log((double) result, 0.0, 0.0);
        }

        for (String str : nonPalindromes) {
            int result = isPalindrome(str) ? 1 : 0;
            assertEquals(0, result);
            JUnitCSVLogger.log((double) result, 0.0, 0.0);
        }
    }

    @Test
    void testStringLength() {
        String[] testCases = {"a","ab","abc","hello","world","testing"};
        int[] expectedLengths = {1,2,3,5,5,7};

        for (int i = 0; i < testCases.length; i++) {
            String str = testCases[i];
            int expected = expectedLengths[i];

            assertEquals(expected, str.length());

            JUnitCSVLogger.log(
                (double) expected,
                0.0,
                0.0
            );
        }
    }

    @Test
    void testMutationAlignmentMathOperationsExample() {
        assertEquals(0, MathOperations.sum(null));
        assertEquals(6, MathOperations.sum(new int[]{1, 2, 3}));

        assertEquals(9, MathOperations.max(new int[]{1, 5, 9, 3}));
        assertEquals(1, MathOperations.min(new int[]{1, 5, 9, 3}));

        assertTrue(MathOperations.isPrime(13));
        assertFalse(MathOperations.isPrime(12));

        assertEquals(120, MathOperations.factorial(5));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.factorial(-1));

        assertEquals(8, MathOperations.fibonacci(6));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.fibonacci(-1));

        assertEquals(3, MathOperations.gcd(9, 6));

        assertTrue(MathOperations.isPerfectSquare(49));
        assertFalse(MathOperations.isPerfectSquare(50));

        assertTrue(MathOperations.isPalindrome("A man, a plan, a canal: Panama"));
        assertFalse(MathOperations.isPalindrome("software testing"));

        assertArrayEquals(new int[]{3, 2, 1}, MathOperations.reverseArray(new int[]{1, 2, 3}));
        assertArrayEquals(new int[]{1, 2, 3}, MathOperations.bubbleSort(new int[]{3, 2, 1}));
    }

    @Test
    void testMutationAlignmentTestDataLoaderExample() {
        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData(10, 20, 30);
        loader.collectTestData("alpha", true);

        assertEquals(2, loader.getCollectedDataSize());
        assertEquals("10,20,30", loader.getCollectedData(0));
        assertEquals("alpha,true", loader.getCollectedData(1));
        assertNull(loader.getCollectedData(-1));
        assertNull(loader.getCollectedData(2));

        List<String> copy = loader.getAllCollectedData();
        copy.clear();
        assertEquals(2, loader.getCollectedDataSize());
    }

    // Helper methods
    private boolean isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    private int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }

    private int fibonacci(int n) {
        if (n <= 1) return n;
        int prev = 0, curr = 1;
        for (int i = 2; i <= n; i++) {
            int next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }

    private boolean isPalindrome(String str) {
        return str.equals(new StringBuilder(str).reverse().toString());
    }
}
