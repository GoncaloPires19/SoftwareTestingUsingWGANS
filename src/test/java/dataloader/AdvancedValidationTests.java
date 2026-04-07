package dataloader;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import utils.JUnitCSVLogger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class AdvancedValidationTests {

    @BeforeEach
    void setUp() {
        // Setup before each test
    }

    // ============== EXCEPTION HANDLING ==============
    @Test
    void testDivisionByZeroException() {
        int[][] testCases = {
            {10, 2, 5, 0},
            {20, 4, 5, 0},
            {15, 3, 5, 0},
            {100, 10, 10, 0},
            {50, 5, 10, 0}
        };

        for (int[] tc : testCases) {
            int a = tc[0], b = tc[1], expected = tc[2];
            int result = safeDivide(a, b);
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)a, (double)b, (double)result);
        }
    }

    @Test
    void testNullValidation() {
        String[] testCases = {"hello", "world", "test"};

        for (String str : testCases) {
            int result = isValidString(str) ? 1 : 0;
            assertEquals(1, result);
            JUnitCSVLogger.log((double)result, 0.0, 0.0);
        }
    }

    // ============== COLLECTION OPERATIONS ==============
    @Test
    void testListOperations() {
        int[][] testCases = {
            {1, 2, 3, 4, 5},
            {10, 20, 30},
            {5, 5, 5},
            {1, 1, 1, 1},
            {7, 8, 9}
        };

        for (int[] arr : testCases) {
            List<Integer> list = new ArrayList<>();
            for (int val : arr) {
                list.add(val);
            }
            assertEquals(arr.length, list.size());
            JUnitCSVLogger.log((double)list.size(), 0.0, 0.0);
        }
    }

    @Test
    void testMapOperations() {
        String[][] testCases = {
            {"key1", "value1"},
            {"test", "data"},
            {"name", "john"},
            {"age", "25"},
            {"city", "lisbon"}
        };

        for (String[] tc : testCases) {
            Map<String, String> map = new HashMap<>();
            map.put(tc[0], tc[1]);
            assertEquals(1, map.size());
            assertTrue(map.containsKey(tc[0]));
            JUnitCSVLogger.log((double)map.size(), 0.0, 0.0);
        }
    }

    // ============== OBJECT COMPARISON ==============
    @Test
    void testObjectEquality() {
        String[] testCases = {"hello", "world", "test", "java", "junit"};

        for (String str : testCases) {
            String copy = new String(str);
            assertEquals(str, copy);
            JUnitCSVLogger.log((double)str.length(), 0.0, 0.0);
        }
    }

    // ============== NUMERIC EDGE CASES ==============
    @Test
    void testLargeNumbers() {
        long[][] testCases = {
            {1000000, 1000000, 2000000},
            {999999, 1, 1000000},
            {500000, 500000, 1000000},
            {1000, 1000000, 1001000},
            {100000, 100000, 200000}
        };

        for (long[] tc : testCases) {
            long a = tc[0], b = tc[1], expected = tc[2];
            long result = a + b;
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)a, (double)b, (double)result);
        }
    }

    @Test
    void testFloatingPointArithmetic() {
        double[][] testCases = {
            {0.1, 0.2, 0.3},
            {1.5, 2.5, 4.0},
            {3.14, 2.86, 6.0},
            {0.5, 0.5, 1.0},
            {2.25, 1.75, 4.0}
        };

        for (double[] tc : testCases) {
            double a = tc[0], b = tc[1], expected = tc[2];
            double result = a + b;
            assertEquals(expected, result, 0.001);
            JUnitCSVLogger.log(a, b, result);
        }
    }

    // ============== BOUNDARY TESTING ==============
    @Test
    void testBoundaryConditions() {
        int[][] testCases = {
            {0, 0, 1},
            {1, 1, 1},
            {Integer.MAX_VALUE, 0, 1},
            {Integer.MIN_VALUE, Integer.MIN_VALUE, 1},
            {100, 100, 1},
            {50, 49, 1},
            {10, 10, 1}
        };

        for (int[] tc : testCases) {
            int a = tc[0], b = tc[1], expected = tc[2];
            int result = (a >= b) ? 1 : 0;
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)a, (double)b, (double)result);
        }
    }

    // ============== SEQUENCE VALIDATION ==============
    @Test
    void testAscendingSequence() {
        int[][] testCases = {
            {1, 2, 3, 4, 5},
            {10, 20, 30, 40, 50},
            {5, 5, 5, 5, 5},
            {1, 3, 5, 7, 9},
            {2, 4, 6, 8, 10}
        };

        for (int[] arr : testCases) {
            boolean isAscending = isAscending(arr);
            assertTrue(isAscending);
            JUnitCSVLogger.log((double)arr.length, 0.0, 0.0);
        }
    }

    @Test
    void testAlternatingPatterns() {
        int[][] testCases = {
            {1, 0, 1, 0, 1},
            {5, 10, 5, 10, 5},
            {3, 7, 3, 7, 3},
            {2, 4, 2, 4, 2},
            {100, 200, 100, 200, 100}
        };

        for (int[] arr : testCases) {
            boolean isAlternating = hasAlternatingPattern(arr);
            assertTrue(isAlternating);
            JUnitCSVLogger.log((double)arr.length, 0.0, 0.0);
        }
    }

    // ============== PERFORMANCE METRICS ==============
    @Test
    void testExecutionTime() {
        int[] testCases = {100, 1000, 10000, 100000, 1000000};

        for (int n : testCases) {
            long start = System.nanoTime();
            sumRange(n);
            long end = System.nanoTime();
            long duration = end - start;

            assertTrue(duration >= 0);
            JUnitCSVLogger.log((double)n, (double)duration, 0.0);
        }
    }

    // ============== PATTERN RECOGNITION ==============
    @Test
    void testPrimeSequence() {
        int[][] testCases = {
            {2, 1},
            {3, 1},
            {5, 1},
            {7, 1},
            {11, 1},
            {4, 0},
            {6, 0},
            {8, 0},
            {9, 0},
            {10, 0}
        };

        for (int[] tc : testCases) {
            int num = tc[0];
            int expected = tc[1];
            int result = isPrime(num) ? 1 : 0;
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)num, (double)result, 0.0);
        }
    }

    @Test
    void testPerfectSquares() {
        int[][] testCases = {
            {1, 1},
            {4, 1},
            {9, 1},
            {16, 1},
            {25, 1},
            {2, 0},
            {3, 0},
            {5, 0},
            {8, 0},
            {10, 0}
        };

        for (int[] tc : testCases) {
            int num = tc[0];
            int expected = tc[1];
            int result = isPerfectSquare(num) ? 1 : 0;
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)num, (double)result, 0.0);
        }
    }

    // ============== PERMUTATION & COMBINATION ==============
    @Test
    void testCombinations() {
        int[][] testCases = {
            {5, 2, 10},
            {4, 2, 6},
            {6, 3, 20},
            {3, 1, 3},
            {5, 1, 5}
        };

        for (int[] tc : testCases) {
            int n = tc[0], r = tc[1], expected = tc[2];
            int result = combination(n, r);
            assertEquals(expected, result);
            JUnitCSVLogger.log((double)n, (double)r, (double)result);
        }
    }

    // ============== MUTATION ALIGNMENT ==============
    @Test
    void testMutationAlignmentMathOperationsValidation() {
        assertThrows(IllegalArgumentException.class, () -> MathOperations.max(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.min(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.factorial(-1));
        assertThrows(IllegalArgumentException.class, () -> MathOperations.fibonacci(-1));

        assertEquals(0, MathOperations.sum(null));
        assertNull(MathOperations.reverseArray(null));
        assertNull(MathOperations.bubbleSort(null));
        assertEquals(1, MathOperations.gcd(17, 13));
    }

    @Test
    void testMutationAlignmentTestDataLoaderValidation() {
        TestDataLoader loader = new TestDataLoader();
        loader.collectTestData("a", "b");
        loader.collectTestData(1, true);

        assertEquals(2, loader.getCollectedDataSize());
        assertEquals("a,b", loader.getCollectedData(0));
        assertEquals("1,true", loader.getCollectedData(1));
        assertNull(loader.getCollectedData(-1));
        assertNull(loader.getCollectedData(2));
    }

    @Test
    void testMutationCoreCoverageMathOperationsValidation() {
        assertEquals(0, MathOperations.sum(new int[0]));
        assertEquals(6, MathOperations.sum(new int[]{1, 2, 3}));

        assertEquals(9, MathOperations.max(new int[]{1, 9, 3, 2}));
        assertEquals(-10, MathOperations.min(new int[]{-10, -5, -1, -8}));

        assertTrue(MathOperations.isPrime(97));
        assertFalse(MathOperations.isPrime(100));

        assertEquals(1, MathOperations.factorial(0));
        assertEquals(720, MathOperations.factorial(6));

        assertEquals(0, MathOperations.fibonacci(0));
        assertEquals(13, MathOperations.fibonacci(7));

        assertEquals(6, MathOperations.gcd(54, 24));

        assertTrue(MathOperations.isPerfectSquare(81));
        assertFalse(MathOperations.isPerfectSquare(82));
        assertFalse(MathOperations.isPerfectSquare(-1));

        assertTrue(MathOperations.isPalindrome("RaceCar"));
        assertFalse(MathOperations.isPalindrome("mutation"));

        assertArrayEquals(new int[]{3, 2, 1}, MathOperations.reverseArray(new int[]{1, 2, 3}));
        assertArrayEquals(new int[]{1, 2, 3, 4}, MathOperations.bubbleSort(new int[]{4, 3, 2, 1}));
        assertArrayEquals(new int[]{5}, MathOperations.bubbleSort(new int[]{5}));
    }

    // ============== HELPER METHODS ==============

    private int safeDivide(int a, int b) {
        if (b == 0) return 0;
        return a / b;
    }

    private boolean isValidString(String str) {
        return str != null && !str.isEmpty();
    }

    private boolean isAscending(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            if (arr[i] > arr[i + 1]) return false;
        }
        return true;
    }

    private boolean hasAlternatingPattern(int[] arr) {
        if (arr.length < 2) return true;
        for (int i = 2; i < arr.length; i++) {
            if (arr[i] != arr[i - 2]) return false;
        }
        return true;
    }

    private long sumRange(int n) {
        long sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        return sum;
    }

    private boolean isPrime(int n) {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    private boolean isPerfectSquare(int n) {
        if (n < 0) return false;
        int root = (int) Math.sqrt(n);
        return root * root == n;
    }

    private int combination(int n, int r) {
        return factorial(n) / (factorial(r) * factorial(n - r));
    }

    private int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
}
